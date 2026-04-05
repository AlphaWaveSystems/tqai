"""Unified codebook solver dispatcher.

Provides a single entry point for codebook generation that routes to the
appropriate solver: Lloyd-Max (fast, default runtime fallback), CMA-ES
(evolutionary refinement), or fuzzy C-means (soft assignment with annealing).

References:
    - Lloyd-Max: TurboQuant arXiv:2504.19874, PolarQuant arXiv:2502.02617
    - CMA-ES: arXiv:1710.05311 (IDE-LBG), arXiv:2106.07611 (NEMO)
    - Fuzzy: arXiv:1908.05033 (DSQ), arXiv:1710.05311 (fuzzy C-means)
"""

from __future__ import annotations

import numpy as np


def solve_codebook(
    d: int,
    bits: int,
    solver: str = "auto",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate optimal codebook centroids and boundaries.

    Args:
        d: Head dimension (coordinate variance = 1/d).
        bits: Number of quantization bits.
        solver: Solver name:
            - ``'auto'``: tries CMA-ES, falls back to Lloyd-Max
            - ``'lloyd_max'``: classical Lloyd-Max (fastest, no extra deps)
            - ``'cmaes'``: CMA-ES evolutionary refinement (requires ``cma``)
            - ``'fuzzy'``: fuzzy C-means with temperature annealing
        **kwargs: Passed through to the selected solver.

    Returns:
        centroids: float32 array ``(2^bits,)``, sorted ascending.
        boundaries: float32 array ``(2^bits - 1,)``.
    """
    if solver == "auto":
        try:
            import cma  # noqa: F401
            solver = "cmaes"
        except ImportError:
            solver = "lloyd_max"

    if solver == "lloyd_max":
        from tqai.codebook.lloyd_max import solve_lloyd_max
        return solve_lloyd_max(d, bits, **kwargs)

    if solver == "cmaes":
        from tqai.codebook.cmaes_solver import solve_cmaes
        return solve_cmaes(d, bits, **kwargs)

    if solver == "fuzzy":
        from tqai.codebook.fuzzy_solver import solve_fuzzy
        return solve_fuzzy(d, bits, **kwargs)

    raise ValueError(f"Unknown solver: {solver!r}. Choose from: lloyd_max, cmaes, fuzzy")
