"""Generate precomputed Lloyd-Max codebooks for all standard configurations.

Usage:
    python -m scripts.generate_codebooks

Generates .npz files in src/tqai/codebook/data/ for:
    head_dims: [64, 96, 128, 256]
    bits: [2, 3, 4]
"""

from __future__ import annotations

from pathlib import Path

from tqai.codebook.lloyd_max import solve_lloyd_max

import numpy as np

HEAD_DIMS = [64, 96, 128, 256]
BITS = [2, 3, 4]

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "tqai" / "codebook" / "data"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for d in HEAD_DIMS:
        for b in BITS:
            filename = f"d{d:03d}_b{b}.npz"
            path = DATA_DIR / filename
            print(f"Generating d={d}, bits={b} ... ", end="", flush=True)
            centroids, boundaries = solve_lloyd_max(d, b)
            np.savez(path, centroids=centroids, boundaries=boundaries, head_dim=d, bits=b)
            n_levels = 1 << b
            print(f"done ({n_levels} levels, centroids range [{centroids[0]:.4f}, {centroids[-1]:.4f}])")

    print(f"\nAll codebooks saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
