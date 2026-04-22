"""RotorQuantizer — Clifford rotor KV cache quantization.

Replaces TurboQuant's dense d×d rotation with block-diagonal 3×3 rotations
implemented via unit quaternions (Cl(3,0) rotors). This gives equivalent
information-theoretic quality at O(d) rotation cost vs O(d²) for TurboQuant.

Algorithm:
  1. Store L2 norm as FP16.
  2. Normalize to unit sphere.
  3. Chunk the d-dim vector into ⌊d/3⌋ groups of 3; apply a per-group unit
     quaternion rotation (rotor sandwich product R v R̃). Any remainder
     dimensions (d % 3 ∈ {1, 2}) are left unrotated.
  4. Quantize each coordinate via precomputed Lloyd-Max codebook (same
     codebooks as PolarQuantizer — distribution is still ≈ N(0, 1/d)).
  5. Store indices as uint8.

Dequantization: lookup centroids → apply inverse rotors → scale by norm.

Reference: Pope (2026), "RotorQuant: Clifford Algebra Vector Quantization
           for LLM KV Cache Compression." https://www.scrya.com/rotorquant/
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from tqai.backend import get_backend
from tqai.codebook import load_codebook


class RotorQuantizer:
    """Block-diagonal Clifford rotor quantizer for KV cache compression.

    Each ⌊d/3⌋ group of 3 dimensions is independently rotated by a random
    unit quaternion before Lloyd-Max scalar quantization. The quaternions are
    generated once at construction time and reused for all vectors.

    Args:
        head_dim: Dimension of vectors to quantize.
        bits: Bits per coordinate (2, 3, 4, 6, or 8).
        seed: RNG seed for rotor generation.
        ops: Backend ops object (auto-detected if None).
    """

    def __init__(
        self,
        head_dim: int,
        bits: int,
        seed: int = 42,
        ops: Any | None = None,
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.seed = seed
        self.ops = ops or get_backend()

        self._n_full = head_dim // 3       # number of complete 3-dim groups
        self._remainder = head_dim % 3     # 0, 1, or 2 unrotated tail dims
        self._rotated_dim = self._n_full * 3

        # Random unit quaternions for each group: shape (n_full, 4)
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((max(self._n_full, 1), 4)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=-1, keepdims=True)
        self._quats: np.ndarray = raw / (norms + 1e-10)

        # Precompute forward and inverse 3×3 rotation matrices
        # Forward: R  (column-vector convention: v' = R @ v, row-vector: v' = v @ R.T)
        # Inverse: R.T (orthogonal matrix)
        self._mats: np.ndarray = self._build_matrices(self._quats)         # (n, 3, 3)
        self._mats_inv: np.ndarray = self._mats.transpose(0, 2, 1).copy() # (n, 3, 3)

        # Lloyd-Max codebook (shared with PolarQuantizer)
        centroids_np, _ = load_codebook(head_dim, bits)
        self._centroids = self.ops.from_numpy(centroids_np)

        # Fused Metal kernel path (MLX backend + Metal GPU only)
        self._use_metal = False
        if hasattr(self.ops, "quantize_fused"):  # MLX backend sentinel
            try:
                from tqai.kernels import metal_available
                if metal_available():
                    import mlx.core as mx
                    # Flat (n_full * 9,) float32 array of rotation matrices
                    self._block_mats_mlx = mx.array(
                        self._mats.reshape(-1).astype(np.float32)
                    )
                    self._centroids_mlx = mx.array(centroids_np)
                    self._use_metal = True
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Rotor construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_matrices(quats: np.ndarray) -> np.ndarray:
        """Build 3×3 rotation matrices from unit quaternions.

        Args:
            quats: (n, 4) array of unit quaternions (w, x, y, z).

        Returns:
            (n, 3, 3) orthogonal rotation matrices.
        """
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        n = len(quats)
        R = np.zeros((n, 3, 3), dtype=np.float32)
        R[:, 0, 0] = w*w + x*x - y*y - z*z
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)
        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = w*w - x*x + y*y - z*z
        R[:, 1, 2] = 2*(y*z - w*x)
        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = w*w - x*x - y*y + z*z
        return R

    # ------------------------------------------------------------------
    # Block-diagonal rotation (vectorized via einsum)
    # ------------------------------------------------------------------

    def _rotate(self, x_np: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Apply block-diagonal quaternion rotations to a batch of vectors.

        Args:
            x_np: (..., head_dim) float32 array.
            inverse: If True, apply the inverse (conjugate) rotations.

        Returns:
            Rotated array, same shape as input.
        """
        if self._n_full == 0:
            return x_np

        batch_shape = x_np.shape[:-1]
        x_flat = x_np.reshape(-1, self.head_dim)   # (N, d)

        mats = self._mats_inv if inverse else self._mats  # (n_full, 3, 3)

        # Slice the rotated portion and reshape into groups: (N, n_full, 3)
        grouped = x_flat[:, :self._rotated_dim].reshape(-1, self._n_full, 3)

        # Vectorized block rotation:
        # For each group g and batch element n:
        #   out[n,g,:] = grouped[n,g,:] @ mats[g].T   (row-vector convention)
        # Einsum: 'ngi,gji -> ngj'  (contract on i=input dim, g=group)
        rotated = np.einsum("ngi,gji->ngj", grouped, mats)  # (N, n_full, 3)

        out = x_flat.copy()
        out[:, :self._rotated_dim] = rotated.reshape(-1, self._rotated_dim)
        return out.reshape(*batch_shape, self.head_dim)

    # ------------------------------------------------------------------
    # Quantize / Dequantize
    # ------------------------------------------------------------------

    def quantize(self, x: Any) -> Tuple:
        """Quantize vectors via block-diagonal rotor rotation + Lloyd-Max.

        Args:
            x: shape ``(..., head_dim)``, float tensors.

        Returns:
            ``(indices, norms)``
              - indices: ``(..., head_dim)`` uint8 codebook indices.
              - norms:   ``(..., 1)`` FP16 L2 norms.
        """
        # Fast path: fused Metal kernel (single GPU dispatch, no CPU round-trip)
        if self._use_metal:
            from tqai.kernels import metal_rotor_quantize
            return metal_rotor_quantize(
                x, self._block_mats_mlx, self._centroids_mlx, self._n_full
            )

        # 1. Norm + normalize
        norms = self.ops.norm(x, dim=-1, keepdim=True)
        x_unit = self.ops.float32(x) / (self.ops.float32(norms) + 1e-10)

        # 2. Block-diagonal rotor rotation (numpy round-trip)
        x_np = self.ops.to_numpy(x_unit).astype(np.float32)
        y_np = self._rotate(x_np, inverse=False)
        y = self.ops.from_numpy(y_np)

        # 3. Per-coordinate nearest-centroid quantization
        y_exp = self.ops.unsqueeze(y, -1)                    # (..., d, 1)
        diffs = self.ops.abs(y_exp - self._centroids)        # (..., d, n_levels)
        indices = self.ops.uint8(self.ops.argmin(diffs, dim=-1))
        norms = self.ops.float16(norms)

        return indices, norms

    def dequantize(self, indices: Any, norms: Any, qjl_data=None) -> Any:
        """Reconstruct vectors from quantized representation.

        Args:
            indices: ``(..., head_dim)`` uint8 codebook indices.
            norms:   ``(..., 1)`` FP16 L2 norms.
            qjl_data: Unused; accepted for API compatibility with PolarQuantizer.

        Returns:
            Reconstructed vectors ``(..., head_dim)`` float32.
        """
        # Fast path: fused Metal kernel
        if self._use_metal:
            from tqai.kernels import metal_rotor_dequantize
            return metal_rotor_dequantize(
                indices, norms, self._block_mats_mlx, self._centroids_mlx, self._n_full
            )

        # 1. Centroid lookup
        y_hat = self.ops.float32(self.ops.index_select(self._centroids, indices))

        # 2. Inverse block-diagonal rotor rotation
        y_np = self.ops.to_numpy(y_hat).astype(np.float32)
        x_np = self._rotate(y_np, inverse=True)
        x_unit_hat = self.ops.from_numpy(x_np)

        # 3. Scale by norm
        return x_unit_hat * self.ops.float32(norms)
