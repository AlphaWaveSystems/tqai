from __future__ import annotations

import logging
import warnings
from importlib.resources import files
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class CodebookRegistry:
    """Manages loading and caching of precomputed Lloyd-Max codebooks.

    Supports per-head-type codebooks (copresheaf-inspired) via the
    ``head_type`` parameter.  When ``head_type`` is ``"generic"`` (default),
    uses standard codebooks.  Other types (``"spatial"``, ``"temporal"``,
    ``"cross_attn"``) fall back to generic if no specialized codebook exists.
    """

    # Recognized head types for copresheaf codebook support
    HEAD_TYPES = ("generic", "spatial", "temporal", "cross_attn")

    def __init__(self):
        self._cache: dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def codebook_filename(head_dim: int, bits: int, head_type: str = "generic") -> str:
        if head_type == "generic":
            return f"d{head_dim:03d}_b{bits}.npz"
        return f"d{head_dim:03d}_b{bits}_{head_type}.npz"

    def load(
        self, head_dim: int, bits: int, head_type: str = "generic",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load codebook from package data, falling back to runtime generation.

        When ``head_type`` is not ``"generic"``, first tries the specialized
        codebook, then falls back to the generic one.
        """
        key = (head_dim, bits, head_type)
        if key in self._cache:
            return self._cache[key]

        # Try specialized codebook first, then fall back to generic
        types_to_try = [head_type]
        if head_type != "generic":
            types_to_try.append("generic")

        for ht in types_to_try:
            result = self._try_load(head_dim, bits, ht)
            if result is not None:
                self._cache[key] = result
                return result

        # Not found at all — generate generic at runtime
        return self._generate_fallback(head_dim, bits, key)

    def _try_load(
        self, head_dim: int, bits: int, head_type: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        cache_key = (head_dim, bits, head_type)
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            data_files = files("tqai.codebook") / "data"
            npz_path = data_files / self.codebook_filename(head_dim, bits, head_type)
            with npz_path.open("rb") as f:
                data = np.load(f)
                result = (data["centroids"], data["boundaries"])
                self._cache[cache_key] = result
                return result
        except (FileNotFoundError, TypeError, KeyError):
            return None

    def _generate_fallback(self, head_dim, bits, key):
        key_generic = (head_dim, bits, "generic")
        if key_generic in self._cache:
            self._cache[key] = self._cache[key_generic]
            return self._cache[key]

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
