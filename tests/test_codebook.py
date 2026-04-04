from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest

HEAD_DIMS = [64, 96, 128, 256]
BITS = [2, 3, 4]


@pytest.mark.parametrize("d", HEAD_DIMS)
@pytest.mark.parametrize("b", BITS)
def test_load_shipped_codebook(d, b):
    from tqai.codebook import load_codebook

    centroids, boundaries = load_codebook(d, b)
    n_levels = 1 << b
    assert centroids.shape == (n_levels,)
    assert boundaries.shape == (n_levels - 1,)
    assert centroids.dtype == np.float32
    assert boundaries.dtype == np.float32


@pytest.mark.parametrize("d", HEAD_DIMS)
@pytest.mark.parametrize("b", BITS)
def test_codebook_symmetry(d, b):
    """Centroids should be symmetric around 0 for a symmetric distribution."""
    from tqai.codebook import load_codebook

    centroids, _ = load_codebook(d, b)
    npt.assert_allclose(centroids, -centroids[::-1], atol=1e-6)


@pytest.mark.parametrize("d", HEAD_DIMS)
@pytest.mark.parametrize("b", BITS)
def test_centroids_sorted(d, b):
    from tqai.codebook import load_codebook

    centroids, boundaries = load_codebook(d, b)
    assert np.all(np.diff(centroids) > 0), "Centroids must be strictly ascending"
    assert np.all(np.diff(boundaries) > 0), "Boundaries must be strictly ascending"


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("b", BITS)
def test_distortion_bound(d, b):
    """Verify MSE is within the theoretical TurboQuant bound."""
    from scipy.integrate import quad

    from tqai.codebook.lloyd_max import gaussian_pdf, solve_lloyd_max

    centroids, boundaries = solve_lloyd_max(d, b)
    n_levels = 1 << b
    full_bounds = np.concatenate([[-np.inf], boundaries, [np.inf]])

    distortion = 0.0
    for j in range(n_levels):
        c = centroids[j]
        d_j, _ = quad(
            lambda x, c=c: (x - c) ** 2 * gaussian_pdf(x, d),
            full_bounds[j],
            full_bounds[j + 1],
        )
        distortion += d_j

    # Per-coordinate distortion. The theoretical bound from the paper is
    # D_mse <= (sqrt(3)*pi/2) / 4^b for normalised MSE.
    # Our distortion is already per-coordinate with variance 1/d,
    # so compare against (1/d) * (sqrt(3)*pi/2) / 4^b.
    theoretical_bound = (1.0 / d) * (math.sqrt(3) * math.pi / 2.0) / (4.0 ** b)
    assert distortion < theoretical_bound, (
        f"Distortion {distortion:.6e} exceeds bound {theoretical_bound:.6e} "
        f"for d={d}, b={b}"
    )


@pytest.mark.parametrize("d", [64, 128])
def test_lloyd_max_convergence(d):
    """Lloyd-Max should converge and produce valid codebooks."""
    from tqai.codebook.lloyd_max import solve_lloyd_max

    centroids, boundaries = solve_lloyd_max(d, bits=3, max_iter=200)
    assert len(centroids) == 8
    assert len(boundaries) == 7
    # Centroids should be within a reasonable range
    sigma = 1.0 / math.sqrt(d)
    assert np.all(np.abs(centroids) < 5.0 * sigma)
