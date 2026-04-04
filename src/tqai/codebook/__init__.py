from __future__ import annotations

from tqai.codebook.registry import CodebookRegistry

_registry = CodebookRegistry()


def load_codebook(head_dim: int, bits: int):
    """Load precomputed Lloyd-Max codebook (centroids, boundaries) as numpy arrays."""
    return _registry.load(head_dim, bits)
