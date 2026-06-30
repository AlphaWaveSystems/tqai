"""Registration of tqai with LMCache's v1 serde factory.

LMCache v1 uses a proper plugin registry (register_serde_factory) rather
than a hard-coded if/elif dispatch, so no monkey-patching is needed.

Usage::

    import lmcache_turbo_quant_serde
    lmcache_turbo_quant_serde.register()

    # Then in your L2 adapter serde config:
    #   {
    #     "type": "tqai",
    #     "head_dim": 128,
    #     "bits": 4,
    #     "seed": 42,
    #     "use_qjl": false,
    #     "max_workers": 1
    #   }
"""

from __future__ import annotations

_REGISTERED: dict[str, str] = {}  # serde_name → module path


def register(serde_name: str = "tqai") -> None:
    """Register tqai with LMCache's v1 serde factory.

    Safe to call multiple times; subsequent calls with the same name are
    no-ops (register_serde_factory raises on duplicate, so we guard).

    Args:
        serde_name: The ``"type"`` field in the LMCache serde config dict.
                    Default ``"tqai"``.

    Raises:
        ImportError: If ``lmcache`` is not installed.
    """
    if serde_name in _REGISTERED:
        return

    try:
        from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
        from lmcache.v1.distributed.serde.factory import register_serde_factory
    except ImportError as exc:
        raise ImportError(
            "lmcache is required for registration: pip install lmcache"
        ) from exc

    from ._v1_codec import TqaiDeserializer, TqaiSerializer

    def _factory(kwargs: dict) -> AsyncSerdeProcessor:
        head_dim = int(kwargs.get("head_dim", 128))
        bits = int(kwargs.get("bits", 4))
        seed = int(kwargs.get("seed", 42))
        use_qjl = bool(kwargs.get("use_qjl", False))
        qjl_sketch_size = int(kwargs.get("qjl_sketch_size", 64))
        max_workers = int(kwargs.get("max_workers", 1))

        s = TqaiSerializer(
            head_dim=head_dim,
            bits=bits,
            seed=seed,
            use_qjl=use_qjl,
            qjl_sketch_size=qjl_sketch_size,
        )
        d = TqaiDeserializer()
        return AsyncSerdeProcessor(s, d, max_workers=max_workers)

    register_serde_factory(serde_name, _factory)
    _REGISTERED[serde_name] = __name__
