"""PolarQuantizer — core TurboQuant Stage 1 algorithm.

Implements data-oblivious vector quantization via:
  1. Store L2 norm as FP16
  2. Normalize to unit sphere
  3. Rotate by a fixed random orthogonal matrix (Haar-distributed, seeded)
  4. Quantize each coordinate independently via precomputed Lloyd-Max codebook
  5. Store coordinate indices as uint8

Dequantization reverses: lookup centroids -> inverse rotate -> scale by norm.

Reference: TurboQuant (arXiv:2504.19874), PolarQuant (arXiv:2502.02617).
"""

from __future__ import annotations

from typing import Any, Tuple

from tqai.backend import get_backend
from tqai.codebook import load_codebook


class PolarQuantizer:
    """Data-oblivious vector quantizer using random rotation + Lloyd-Max codebooks."""

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

        self._rotation = self._build_rotation_matrix()
        centroids_np, _ = load_codebook(head_dim, bits)
        self._centroids = self.ops.from_numpy(centroids_np)

    def _build_rotation_matrix(self) -> Any:
        """Generate Haar-distributed orthogonal matrix via QR of Gaussian."""
        G = self.ops.randn((self.head_dim, self.head_dim), seed=self.seed)
        Q, R = self.ops.qr(G)
        # Fix sign ambiguity: multiply columns of Q by sign(diag(R))
        diag_sign = self.ops.sign(self._diag(R))
        Q = self.ops.matmul(Q, self._diag_matrix(diag_sign))
        return Q

    def _diag(self, matrix: Any) -> Any:
        """Extract diagonal of a 2D matrix."""
        import numpy as np

        mat_np = self.ops.to_numpy(matrix)
        return self.ops.from_numpy(np.diag(mat_np).copy().astype(np.float32))

    def _diag_matrix(self, vec: Any) -> Any:
        """Create diagonal matrix from vector."""
        import numpy as np

        v_np = self.ops.to_numpy(vec)
        return self.ops.from_numpy(np.diag(v_np).astype(np.float32))

    def quantize(self, x: Any) -> Tuple[Any, Any]:
        """Quantize vectors.

        Args:
            x: shape ``(..., head_dim)``, float tensors.

        Returns:
            indices: shape ``(..., head_dim)``, uint8 codebook indices.
            norms: shape ``(..., 1)``, FP16 L2 norms.
        """
        # 1. Extract and store norm
        norms = self.ops.norm(x, dim=-1, keepdim=True)

        # 2. Normalize
        safe_norms = norms + 1e-10
        x_unit = self.ops.float32(x) / self.ops.float32(safe_norms)

        # 3. Rotate: y = x_unit @ R^T
        y = self.ops.matmul(x_unit, self.ops.transpose(self._rotation))

        # 4. Per-coordinate quantization: find nearest centroid
        # y shape: (..., d), centroids shape: (n_levels,)
        # Expand for broadcasting: y -> (..., d, 1), centroids -> (n_levels,)
        y_expanded = self.ops.unsqueeze(y, -1)  # (..., d, 1)
        diffs = self.ops.abs(y_expanded - self._centroids)  # (..., d, n_levels)
        indices = self.ops.argmin(diffs, dim=-1)  # (..., d)
        indices = self.ops.uint8(indices)
        norms = self.ops.float16(norms)

        return indices, norms

    def dequantize(self, indices: Any, norms: Any) -> Any:
        """Reconstruct vectors from quantized representation.

        Args:
            indices: shape ``(..., head_dim)``, uint8 codebook indices.
            norms: shape ``(..., 1)``, FP16 L2 norms.

        Returns:
            Reconstructed vectors, shape ``(..., head_dim)``, float32.
        """
        # 1. Lookup centroids
        y_hat = self.ops.index_select(self._centroids, indices)  # (..., d)
        y_hat = self.ops.float32(y_hat)

        # 2. Inverse rotation: x_unit_hat = y_hat @ R
        x_unit_hat = self.ops.matmul(y_hat, self._rotation)

        # 3. Scale by norm
        x_hat = x_unit_hat * self.ops.float32(norms)

        return x_hat
