"""CMA-ES evolutionary codebook optimizer.

Uses Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to refine
Lloyd-Max codebooks.  CMA-ES is initialized from the Lloyd-Max solution and
searches for globally better centroid positions by evolving a population of
candidate codebooks.

For 2-4 bit codebooks (4-16 centroids), the search space is tiny and
CMA-ES converges in seconds.  For 8-bit (256 centroids), Lloyd-Max is
likely near-optimal already.

References:
    - CMA-ES: Hansen & Ostermeier (2001), "Completely Derandomized
      Self-Adaptation in Evolution Strategies"
    - IDE-LBG (evolutionary VQ): arXiv:1710.05311
    - GA-VQ: arXiv:2102.08893
    - NEMO (multi-objective evolutionary quantization): arXiv:2106.07611

Requires: ``cma>=3.0`` (optional ``codegen`` dependency).
"""

from __future__ import annotations

import math

import numpy as np


def solve_cmaes(
    d: int,
    bits: int,
    objective: str = "mse",
    n_samples: int = 100_000,
    max_generations: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimize codebook centroids via CMA-ES.

    Args:
        d: Head dimension (determines coordinate variance = 1/d).
        bits: Number of quantization bits.
        objective: Objective function (``'mse'`` or ``'attention'``).
        n_samples: Monte Carlo sample size for objective evaluation.
        max_generations: Maximum CMA-ES generations.
        seed: RNG seed for reproducibility.

    Returns:
        centroids: float32 array of shape ``(2^bits,)``, sorted ascending.
        boundaries: float32 array of shape ``(2^bits - 1,)``.
    """
    import cma

    from tqai.codebook.lloyd_max import solve_lloyd_max
    from tqai.codebook.objectives import mse_objective

    n_levels = 1 << bits
    sigma = 1.0 / math.sqrt(d)

    # Generate fixed Monte Carlo samples from N(0, 1/d)
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, sigma, size=n_samples).astype(np.float64)

    # Initialize from Lloyd-Max (CMA-ES refines, never degrades)
    lm_centroids, _ = solve_lloyd_max(d, bits)
    x0 = lm_centroids.astype(np.float64)

    # Build objective
    if objective == "mse":
        def fitness(x):
            return mse_objective(np.sort(np.asarray(x)), samples)
    elif objective == "attention":
        from tqai.codebook.objectives import attention_score_objective

        # Generate random rotation and vector samples for attention objective
        G = rng.standard_normal((d, d)).astype(np.float64)
        Q, R_mat = np.linalg.qr(G)
        diag_sign = np.sign(np.diag(R_mat))
        rotation = Q * diag_sign[None, :]

        samples_q = rng.standard_normal((32, d)).astype(np.float64) / math.sqrt(d)
        samples_k = rng.standard_normal((64, d)).astype(np.float64) / math.sqrt(d)

        def fitness(x):
            return attention_score_objective(
                np.sort(np.asarray(x)), samples_q, samples_k, d, rotation
            )
    else:
        raise ValueError(f"Unknown objective: {objective!r}")

    # CMA-ES options
    opts = cma.CMAOptions()
    opts.set("maxiter", max_generations)
    opts.set("seed", seed)
    opts.set("verbose", -9)  # suppress output
    opts.set("tolfun", 1e-14)

    # Initial step size proportional to centroid spacing
    sigma0 = 0.1 * sigma

    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)

    while not es.stop():
        solutions = es.ask()
        # Sort each candidate to maintain ordering invariant
        sorted_solutions = [np.sort(s).tolist() for s in solutions]
        fitnesses = [fitness(s) for s in sorted_solutions]
        es.tell(sorted_solutions, fitnesses)

    best = np.sort(np.asarray(es.result.xbest)).astype(np.float32)

    # Compute boundaries as midpoints
    boundaries = np.array(
        [(best[i] + best[i + 1]) / 2.0 for i in range(n_levels - 1)],
        dtype=np.float32,
    )

    return best, boundaries
