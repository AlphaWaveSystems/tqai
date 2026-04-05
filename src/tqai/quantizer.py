"""PolarQuantizer — TurboQuant Stage 1 + optional Stage 2 (QJL).

Stage 1 (PolarQuant):
  1. Store L2 norm as FP16
  2. Normalize to unit sphere
  3. Rotate by a fixed random orthogonal matrix (Haar-distributed, seeded)
  4. Quantize each coordinate via precomputed Lloyd-Max codebook
  5. Store coordinate indices as uint8

Stage 2 (QJL, optional, default off):
  Adds a 1-bit Johnson-Lindenstrauss sketch of the quantization residual.
  The sketch allows recovering a correction term at dequantization time,
  reducing systematic inner-product bias at the cost of slight variance.
  NOTE: independent research found QJL degrades softmax attention on
  average. Enable via ``use_qjl=True`` only for research or non-softmax use.

Dequantization reverses: lookup centroids -> (optional JL correction) ->
inverse rotate -> scale by norm.

Reference: TurboQuant (arXiv:2504.19874), PolarQuant (arXiv:2502.02617),
           QJL (AAAI 2025, dl.acm.org/doi/10.1609/aaai.v39i24.34773).
"""

from __future__ import annotations

import math
from typing import Any, Tuple

from tqai.backend import get_backend
from tqai.codebook import load_codebook


class PolarQuantizer:
    """Data-oblivious vector quantizer using random rotation + Lloyd-Max codebooks.

    Args:
        head_dim: Dimension of vectors to quantize.
        bits: Bits per coordinate (2, 3, 4, 6, or 8).
        seed: RNG seed for the rotation matrix.
        ops: Backend ops object (auto-detected if None).
        use_qjl: If True, compute and store a QJL residual sketch during
            quantization (Stage 2). The sketch is returned as a third element
            and, if passed to ``dequantize``, adds a correction to the
            reconstructed vector.
        qjl_sketch_size: Number of 1-bit JL projections (default 64).
    """

    def __init__(
        self,
        head_dim: int,
        bits: int,
        seed: int = 42,
        ops: Any | None = None,
        use_qjl: bool = False,
        qjl_sketch_size: int = 64,
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.seed = seed
        self.ops = ops or get_backend()
        self.use_qjl = use_qjl
        self.qjl_sketch_size = qjl_sketch_size

        self._rotation = self._build_rotation_matrix()
        centroids_np, _ = load_codebook(head_dim, bits)
        self._centroids = self.ops.from_numpy(centroids_np)

        if use_qjl:
            self._G = self._build_jl_matrix()

    # ------------------------------------------------------------------
    # Matrix construction
    # ------------------------------------------------------------------

    def _build_rotation_matrix(self) -> Any:
        """Generate Haar-distributed orthogonal matrix via QR of Gaussian."""
        G = self.ops.randn((self.head_dim, self.head_dim), seed=self.seed)
        Q, R = self.ops.qr(G)
        # Fix sign ambiguity: multiply columns of Q by sign(diag(R))
        diag_sign = self.ops.sign(self._diag(R))
        Q = self.ops.matmul(Q, self._diag_matrix(diag_sign))
        return Q

    def _build_jl_matrix(self) -> Any:
        """Generate random Gaussian JL projection matrix G of shape (m, head_dim).

        Uses a deterministic seed offset from the rotation seed so that G
        is always the same for a given quantizer instance.
        """
        import numpy as np

        rng = np.random.default_rng(self.seed + 999999)
        G_np = rng.standard_normal((self.qjl_sketch_size, self.head_dim)).astype(np.float32)
        G_np /= math.sqrt(self.qjl_sketch_size)
        return self.ops.from_numpy(G_np)

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

    # ------------------------------------------------------------------
    # Quantize
    # ------------------------------------------------------------------

    def quantize(self, x: Any) -> Tuple:
        """Quantize vectors.

        Args:
            x: shape ``(..., head_dim)``, float tensors.

        Returns:
            Without QJL: ``(indices, norms)``
              - indices: shape ``(..., head_dim)``, uint8 codebook indices.
              - norms: shape ``(..., 1)``, FP16 L2 norms.

            With QJL (``use_qjl=True``): ``(indices, norms, qjl_data)``
              - qjl_data: ``(sketch, residual_norm)`` where
                - sketch shape ``(..., qjl_sketch_size)``, int8 (±1 signs)
                - residual_norm shape ``(..., 1)``, FP16 L2 norm of residual
        """
        # 1. Extract and store norm
        norms = self.ops.norm(x, dim=-1, keepdim=True)

        # 2. Normalize
        safe_norms = norms + 1e-10
        x_unit = self.ops.float32(x) / self.ops.float32(safe_norms)

        # 3. Rotate: y = x_unit @ R^T
        y = self.ops.matmul(x_unit, self.ops.transpose(self._rotation))

        # 4. Per-coordinate quantization: find nearest centroid
        y_expanded = self.ops.unsqueeze(y, -1)  # (..., d, 1)
        diffs = self.ops.abs(y_expanded - self._centroids)  # (..., d, n_levels)
        indices = self.ops.argmin(diffs, dim=-1)  # (..., d)
        indices = self.ops.uint8(indices)
        norms = self.ops.float16(norms)

        if not self.use_qjl:
            return indices, norms

        # Stage 2 (QJL): compute 1-bit sketch of the quantization residual
        # Residual: r = y (rotated unit vector) - centroid_lookup(indices)
        y_hat_unit = self.ops.index_select(self._centroids, indices)  # (..., d)
        y_hat_unit = self.ops.float32(y_hat_unit)
        residual = y - y_hat_unit  # (..., d)

        # Residual norm (for scaling at dequantize time)
        residual_norm = self.ops.norm(residual, dim=-1, keepdim=True)
        residual_norm = self.ops.float16(residual_norm)

        # JL sketch: sign(G @ r) for each vector
        # G shape: (m, d), residual shape: (..., d) → projection: (..., m)
        proj = self.ops.matmul(residual, self.ops.transpose(self._G))  # (..., m)
        sketch = self.ops.int8(self.ops.sign(proj))  # (..., m), values ∈ {-1, 0, +1}

        return indices, norms, (sketch, residual_norm)

    # ------------------------------------------------------------------
    # Dequantize
    # ------------------------------------------------------------------

    def dequantize(self, indices: Any, norms: Any, qjl_data=None) -> Any:
        """Reconstruct vectors from quantized representation.

        Args:
            indices: shape ``(..., head_dim)``, uint8 codebook indices.
            norms: shape ``(..., 1)``, FP16 L2 norms.
            qjl_data: Optional ``(sketch, residual_norm)`` from Stage 2.
                If provided, adds a JL correction to the unit-sphere
                reconstruction before scaling by norm.

        Returns:
            Reconstructed vectors, shape ``(..., head_dim)``, float32.
        """
        # 1. Lookup centroids
        y_hat = self.ops.index_select(self._centroids, indices)  # (..., d)
        y_hat = self.ops.float32(y_hat)

        # 2. QJL correction (applied on unit sphere before inverse rotation)
        if qjl_data is not None:
            sketch, residual_norm = qjl_data
            # Reconstruct correction direction: G^T @ s  shape (..., d)
            correction_dir = self.ops.matmul(
                self.ops.float32(sketch), self._G
            )  # (..., m) @ (m, d) → (..., d)
            # E[G^T @ sign(G @ r)] ≈ r * sqrt(2m/π) for G with entries N(0,1/m).
            # Normalize to recover r: multiply by sqrt(π/(2m)).
            # Then scale by residual_norm so correction ≈ r * ||r|| in unit-sphere space.
            correction_norm = math.sqrt(math.pi / (2 * self.qjl_sketch_size))
            scale = self.ops.float32(residual_norm) * correction_norm
            y_hat = y_hat + correction_dir * scale

        # 3. Inverse rotation: x_unit_hat = y_hat @ R
        x_unit_hat = self.ops.matmul(y_hat, self._rotation)

        # 4. Scale by norm
        x_hat = x_unit_hat * self.ops.float32(norms)

        return x_hat
