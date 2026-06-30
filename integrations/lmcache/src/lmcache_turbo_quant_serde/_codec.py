"""TurboQuant Serializer / Deserializer conforming to LMCache's SERDE interface.

Drop-in replacement for lmcache's "torch" or "safetensor" codecs that uses
tqai's PolarQuantizer to compress KV tensors before storage.

Compression path  (Serializer.to_bytes):
  tensor (bfloat16)
    → quantize  → (uint8 indices, float16 norms)
    → pack      → bit-packed indices  (tqai.packing)
    → encode    → self-describing bytes  (_wire.encode)

Decompression path  (Deserializer.from_bytes):
  bytes
    → decode    → WireHeader + packed_indices + norms  (_wire.decode)
    → unpack    → uint8 indices
    → dequantize → float32 tensor
    → cast      → original dtype
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch

from ._wire import WireHeader, decode, encode

try:
    from tqai.backend import get_backend
    from tqai.packing import pack, packed_size, unpack
    from tqai.quantizer import PolarQuantizer
except ImportError as exc:
    raise ImportError(
        "tqai is required: pip install tqai"
    ) from exc

# ---------------------------------------------------------------------------
# Optional LMCache base classes — fall back to local ABCs when not installed
# so the codec can be tested and used standalone.
# ---------------------------------------------------------------------------

try:
    from lmcache.storage_backend.serde.serde import (
        Deserializer as _LMDeserializer,
    )
    from lmcache.storage_backend.serde.serde import (
        Serializer as _LMSerializer,
    )
except ImportError:
    import abc

    class _LMSerializer(abc.ABC):  # type: ignore[no-redef]
        @abc.abstractmethod
        def to_bytes(self, t: torch.Tensor) -> bytes: ...

    class _LMDeserializer(abc.ABC):  # type: ignore[no-redef]
        @abc.abstractmethod
        def from_bytes(self, bs: Union[bytes, bytearray]) -> torch.Tensor: ...


_SUPPORTED_BITS = frozenset({2, 3, 4, 6, 8})


# ---------------------------------------------------------------------------
# Shared quantizer cache — keyed by all construction params so serializer
# and deserializer independently reproduce identical rotation matrices.
# lru_cache is GIL-protected in CPython; safe for concurrent use.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _get_quantizer(
    head_dim: int,
    bits: int,
    seed: int,
    use_qjl: bool,
    qjl_sketch_size: int,
    device: str,
) -> PolarQuantizer:
    ops = get_backend("torch", device)
    return PolarQuantizer(
        head_dim=head_dim,
        bits=bits,
        seed=seed,
        ops=ops,
        use_qjl=use_qjl,
        qjl_sketch_size=qjl_sketch_size,
    )


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------


class TurboQuantSerializer(_LMSerializer):
    """Compress a KV tensor with TurboQuant before writing to LMCache storage.

    Args:
        bits: Bits per coordinate.  Lower = smaller storage, slightly lower
              quality.  Default 4 gives ~3.9× compression at Δppl ≈ 0.00.
              Supported: 2, 3, 4, 6, 8.
        seed: RNG seed for the Haar rotation matrix.  Must match the
              corresponding TurboQuantDeserializer (encoded in wire format).
        use_qjl: Enable Stage-2 QJL residual correction for marginal quality
                 gain at the cost of ~qjl_sketch_size / head_dim extra bytes.
        qjl_sketch_size: Number of JL projections (only relevant when
                         use_qjl=True).
        device: Torch device for quantization computation (default "cpu").
    """

    def __init__(
        self,
        bits: int = 4,
        seed: int = 42,
        use_qjl: bool = False,
        qjl_sketch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        if bits not in _SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {sorted(_SUPPORTED_BITS)}, got {bits}")
        self.bits = bits
        self.seed = seed
        self.use_qjl = use_qjl
        self.qjl_sketch_size = qjl_sketch_size
        self.device = device

    @classmethod
    def from_lmcache_config(cls, config: object) -> "TurboQuantSerializer":
        """Build from an LMCacheEngineConfig, reading turbo_quant_* attributes."""
        return cls(
            bits=getattr(config, "turbo_quant_bits", 4),
            seed=getattr(config, "turbo_quant_seed", 42),
            use_qjl=getattr(config, "turbo_quant_use_qjl", False),
            qjl_sketch_size=getattr(config, "turbo_quant_qjl_sketch_size", 64),
            device=getattr(config, "turbo_quant_device", "cpu"),
        )

    def to_bytes(self, t: torch.Tensor) -> bytes:
        original_shape: tuple[int, ...] = tuple(t.shape)
        original_dtype: str = str(t.dtype).replace("torch.", "")
        head_dim: int = t.shape[-1]

        q = _get_quantizer(head_dim, self.bits, self.seed, self.use_qjl, self.qjl_sketch_size, self.device)

        x = t.contiguous().to(device=self.device, dtype=torch.float32)
        result = q.quantize(x)

        if self.use_qjl:
            indices_t, norms_t, (sketch_t, rn_t) = result
            sketch_np: Optional[np.ndarray] = sketch_t.cpu().numpy()
            rn_np: Optional[np.ndarray] = rn_t.cpu().numpy()
        else:
            indices_t, norms_t = result
            sketch_np = rn_np = None

        indices_np = indices_t.cpu().numpy().astype(np.uint8)  # (..., head_dim)
        norms_np = norms_t.cpu().numpy().astype(np.float16)    # (..., 1)

        # Bit-pack all indices in one call
        packed = pack(indices_np.flatten(), self.bits)

        header = WireHeader(
            bits=self.bits,
            seed=self.seed,
            use_qjl=self.use_qjl,
            qjl_sketch_size=self.qjl_sketch_size,
            original_shape=original_shape,
            original_dtype=original_dtype,
        )
        return encode(header, packed, norms_np, sketch_np, rn_np)


# ---------------------------------------------------------------------------
# Deserializer
# ---------------------------------------------------------------------------


class TurboQuantDeserializer(_LMDeserializer):
    """Decompress a TurboQuant-compressed KV tensor read from LMCache storage.

    All compression parameters (bits, seed, use_qjl …) are read from the
    wire format — the deserializer needs no configuration beyond the device.

    Args:
        device: Torch device for dequantization computation (default "cpu").
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    @classmethod
    def from_lmcache_config(cls, config: object) -> "TurboQuantDeserializer":
        return cls(device=getattr(config, "turbo_quant_device", "cpu"))

    def from_bytes(self, bs: Union[bytes, bytearray]) -> torch.Tensor:
        if isinstance(bs, bytearray):
            bs = bytes(bs)

        header, packed_indices, norms_flat, qjl_raw = decode(bs)

        q = _get_quantizer(
            header.original_shape[-1],  # head_dim is always the last dim
            header.bits,
            header.seed,
            header.use_qjl,
            header.qjl_sketch_size,
            self.device,
        )

        # Unpack bit-packed indices → original tensor shape
        total = int(np.prod(header.original_shape))
        indices_np = unpack(packed_indices, header.bits, shape=(total,))
        indices_np = indices_np.reshape(header.original_shape)

        # Norms shape: same as original but with last dim = 1
        norms_shape = header.original_shape[:-1] + (1,)
        norms_np = norms_flat.reshape(norms_shape)

        indices_t = torch.from_numpy(indices_np.astype(np.uint8)).to(self.device)
        norms_t = torch.from_numpy(norms_np.astype(np.float16)).to(self.device)

        qjl_tuple: Optional[tuple] = None
        if header.use_qjl and qjl_raw is not None:
            sketch_np, rn_np = qjl_raw
            sketch_shape = header.original_shape[:-1] + (header.qjl_sketch_size,)
            rn_shape = norms_shape
            sketch_t = torch.from_numpy(sketch_np.reshape(sketch_shape).astype(np.int8)).to(self.device)
            rn_t = torch.from_numpy(rn_np.reshape(rn_shape).astype(np.float16)).to(self.device)
            qjl_tuple = (sketch_t, rn_t)

        reconstructed = q.dequantize(indices_t, norms_t, qjl_tuple)

        target_dtype = getattr(torch, header.original_dtype, torch.float32)
        return reconstructed.to(dtype=target_dtype).reshape(header.original_shape)
