from __future__ import annotations

import functools
import platform


def detect_backend() -> str:
    """Auto-detect the best available backend."""
    if platform.processor() == "arm" or platform.machine() == "arm64":
        try:
            import mlx.core  # noqa: F401

            return "mlx"
        except ImportError:
            pass
    try:
        import torch  # noqa: F401

        return "torch"
    except ImportError:
        pass
    raise RuntimeError(
        "No backend available. Install torch or mlx: "
        "pip install tqai[torch] or tqai[mlx]"
    )


@functools.lru_cache(maxsize=8)
def get_backend(name: str | None = None, device: str | None = None):
    """Return a cached backend ops instance."""
    name = name or detect_backend()
    if name == "torch":
        from tqai.backend._torch import TorchOps

        return TorchOps(device=device or "cpu")
    elif name == "mlx":
        from tqai.backend._mlx import MLXOps

        return MLXOps()
    raise ValueError(f"Unknown backend: {name!r}")
