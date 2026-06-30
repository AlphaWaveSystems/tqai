"""lmcache-turbo-quant-serde — tqai KV-cache compression for LMCache.

Standalone (no LMCache required)::

    from lmcache_turbo_quant_serde import TurboQuantSerializer, TurboQuantDeserializer

    ser = TurboQuantSerializer(bits=4)
    des = TurboQuantDeserializer()
    compressed = ser.to_bytes(kv_tensor)       # → bytes
    recovered   = des.from_bytes(compressed)   # → tensor

LMCache v1 integration::

    import lmcache_turbo_quant_serde
    lmcache_turbo_quant_serde.register()       # call once at startup

    # In your L2 adapter serde config:
    #   {"type": "tqai", "head_dim": 128, "bits": 4}
"""

from ._codec import TurboQuantDeserializer, TurboQuantSerializer
from ._register import register
from ._v1_codec import TqaiDeserializer, TqaiSerializer

__version__ = "0.1.0"
__all__ = [
    # Standalone (old-style to_bytes / from_bytes)
    "TurboQuantSerializer",
    "TurboQuantDeserializer",
    # LMCache v1 (serialize / deserialize / estimate_serialized_size)
    "TqaiSerializer",
    "TqaiDeserializer",
    # Registration
    "register",
    "__version__",
]
