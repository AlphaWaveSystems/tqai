from __future__ import annotations

import logging
import warnings
from importlib.resources import files
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class CodebookRegistry:
    """Manages loading and caching of precomputed Lloyd-Max codebooks."""

    def __init__(self):
        self._cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def codebook_filename(head_dim: int, bits: int) -> str:
        return f"d{head_dim:03d}_b{bits}.npz"

    def load(self, head_dim: int, bits: int) -> tuple[np.ndarray, np.ndarray]:
        """Load codebook from package data, falling back to runtime generation."""
        key = (head_dim, bits)
        if key in self._cache:
            return self._cache[key]

        # Try loading from shipped package data
        try:
            data_files = files("tqai.codebook") / "data"
            npz_path = data_files / self.codebook_filename(head_dim, bits)
            with npz_path.open("rb") as f:
                data = np.load(f)
                centroids = data["centroids"]
                boundaries = data["boundaries"]
                self._cache[key] = (centroids, boundaries)
                return centroids, boundaries
        except (FileNotFoundError, TypeError, KeyError):
            pass

        # Fallback: generate at runtime
        warnings.warn(
            f"Codebook for d={head_dim}, b={bits} not found in package data. "
            f"Generating at runtime (requires scipy). Consider running: "
            f"python -m scripts.generate_codebooks",
            stacklevel=2,
        )
        from tqai.codebook.solvers import solve_codebook

        centroids, boundaries = solve_codebook(head_dim, bits, solver="lloyd_max")
        self._cache[key] = (centroids, boundaries)
        return centroids, boundaries

    def save(self, head_dim: int, bits: int, path: Path) -> None:
        """Save a codebook to an .npz file."""
        centroids, boundaries = self.load(head_dim, bits)
        np.savez(
            path,
            centroids=centroids,
            boundaries=boundaries,
            head_dim=head_dim,
            bits=bits,
        )
        logger.info("Saved codebook d=%d b=%d to %s", head_dim, bits, path)
