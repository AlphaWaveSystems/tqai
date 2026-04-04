"""tqai — TurboQuant KV cache compression for local LLM inference."""

from __future__ import annotations

__version__ = "0.1.0"

from tqai.config import TurboQuantConfig


def patch(
    model,
    bits_k: int = 4,
    bits_v: int = 2,
    sink_tokens: int = 0,
    backend: str | None = None,
    device: str | None = None,
    config_path: str | None = None,
):
    """Enable TurboQuant KV cache compression on a model.

    Args:
        model: HuggingFace model or mlx-lm model.
        bits_k: Bits per key coordinate (default 4).
        bits_v: Bits per value coordinate (default 2).
        sink_tokens: Number of initial tokens to keep uncompressed.
        backend: ``'torch'``, ``'mlx'``, or ``None`` (auto-detect).
        device: PyTorch device string (ignored for MLX).
        config_path: Path to a pre-converted tqai config directory
            (from ``tqai convert``). When provided, ``bits_k`` and ``bits_v``
            are read from the saved config.

    Returns:
        For HuggingFace: a ``TurboQuantDynamicCache`` to pass as ``past_key_values``.
        For mlx-lm: ``None`` (model is patched in-place).
    """
    from tqai._patch import _patch

    if config_path is not None:
        from tqai.convert import load_converted

        converted = load_converted(config_path)
        cfg = converted["config"]
        bits_k = cfg["bits_k"]
        bits_v = cfg["bits_v"]

    config = TurboQuantConfig(
        bits_k=bits_k,
        bits_v=bits_v,
        seed=42,
        sink_tokens=sink_tokens,
        backend=backend,
        device=device,
        config_path=config_path,
    )
    return _patch(model, config)


def unpatch(model):
    """Restore original cache behaviour."""
    from tqai._patch import _unpatch

    _unpatch(model)


__all__ = ["TurboQuantConfig", "patch", "unpatch", "__version__"]
