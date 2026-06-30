"""LMCache v1 distributed-serde interface for tqai compression.

Implements lmcache.v1.distributed.serde.base.Serializer / Deserializer so
the tqai PolarQuantizer can be registered with LMCache's v1 storage layer
via register_serde_factory("tqai", ...).

Tensor contract (matches LMCache's internal KV layout):
    src.tensor / dst.tensor  shape: [2, num_layers, num_tokens, hidden_dim]
    hidden_dim = num_heads * head_dim

The compressed byte stream uses the same self-describing wire format as
the standalone codec (_wire.py) so files written by either codec are
interchangeable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ._codec import _get_quantizer  # reuse shared quantizer cache
from ._wire import WireHeader, decode, encode

try:
    from tqai.packing import pack, unpack
except ImportError as exc:
    raise ImportError("tqai is required: pip install tqai") from exc

try:
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.distributed.serde.base import (
        Deserializer as _LMDeserializer,
    )
    from lmcache.v1.distributed.serde.base import (
        Serializer as _LMSerializer,
    )
    from lmcache.v1.memory_management import MemoryObj  # noqa: F401 (used in docstrings/hints)
except ImportError:
    import abc as _abc

    # Local shims so the module can be imported without lmcache installed.
    # When lmcache IS installed the real ABCs are used, enabling isinstance checks.
    class _LMSerializer(_abc.ABC):  # type: ignore[no-redef]
        @_abc.abstractmethod
        def serialize(self, src: object, dst: object) -> int: ...

        @_abc.abstractmethod
        def estimate_serialized_size(self, layout: object) -> int: ...

    class _LMDeserializer(_abc.ABC):  # type: ignore[no-redef]
        @_abc.abstractmethod
        def deserialize(self, src: object, dst: object) -> None: ...


class TqaiSerializer(_LMSerializer):
    """Compress KV tensors with tqai PolarQuantizer before L2 storage.

    Implements the sync LMCache v1 Serializer interface.
    Wrap with AsyncSerdeProcessor (done automatically by register()) for
    non-blocking use inside the StorageManager.

    Args:
        head_dim: Per-head hidden dimension (e.g. 128 for LLaMA-3).
                  Required for norm allocation in estimate_serialized_size.
        bits: Bits per coordinate (2|3|4|6|8).  Default 4 gives ~3.9×
              compression with Δppl ≈ 0.00.
        seed: RNG seed for the Haar rotation matrix (must match TqaiDeserializer).
        use_qjl: Enable Stage-2 QJL residual correction.
        qjl_sketch_size: JL sketch size (ignored when use_qjl=False).
    """

    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 4,
        seed: int = 42,
        use_qjl: bool = False,
        qjl_sketch_size: int = 64,
    ) -> None:
        self.head_dim = head_dim
        self.bits = bits
        self.seed = seed
        self.use_qjl = use_qjl
        self.qjl_sketch_size = qjl_sketch_size

    def estimate_serialized_size(self, layout: MemoryLayoutDesc) -> int:
        """Upper bound on compressed bytes for a given KV layout.

        The value is deliberately conservative to ensure the pre-allocated
        temp buffer is always large enough.
        """
        shape = layout.shapes[0]
        total_elements = 1
        for d in shape:
            total_elements *= d

        # Worst-case index bytes: bits=8 → 1 byte per index (no packing gain)
        max_index_bytes = total_elements

        # One fp16 norm per vector of size head_dim
        total_vectors = total_elements // self.head_dim
        norm_bytes = total_vectors * 2

        # QJL overhead: int8 sketch + fp16 residual norm per vector
        qjl_bytes = total_vectors * (self.qjl_sketch_size + 2) if self.use_qjl else 0

        # Wire format header + safety margin
        header_overhead = 256

        return max_index_bytes + norm_bytes + qjl_bytes + header_overhead

    def serialize(self, src: MemoryObj, dst: MemoryObj) -> int:
        """Compress src.tensor into dst.tensor (uint8 byte buffer).

        Returns the number of bytes actually written, which is ≤
        estimate_serialized_size().  The caller (AsyncSerdeProcessor) will
        call dst.set_used_size(n) to narrow the logical size of dst.
        """
        kv = src.tensor  # LMCache layout: [2, num_layers, num_tokens, hidden_dim]
        original_dtype = str(kv.dtype).replace("torch.", "")

        # PolarQuantizer requires head_dim as the last axis.
        # Reshape [2, L, T, H*D] → [2, L, T, H, head_dim] before quantizing.
        *leading, hidden_dim = kv.shape
        if hidden_dim % self.head_dim != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} is not divisible by head_dim={self.head_dim}"
            )
        num_heads = hidden_dim // self.head_dim
        per_head_shape = tuple((*leading, num_heads, self.head_dim))

        q = _get_quantizer(
            self.head_dim, self.bits, self.seed,
            self.use_qjl, self.qjl_sketch_size, "cpu",
        )

        x = kv.reshape(per_head_shape).cpu().contiguous().float()
        result = q.quantize(x)

        if self.use_qjl:
            indices_t, norms_t, (sketch_t, rn_t) = result
            sketch_np: Optional[np.ndarray] = sketch_t.cpu().numpy()
            rn_np: Optional[np.ndarray] = rn_t.cpu().numpy()
        else:
            indices_t, norms_t = result
            sketch_np = rn_np = None

        indices_np = indices_t.cpu().numpy().astype(np.uint8)
        norms_np = norms_t.cpu().numpy().astype(np.float16)
        packed = pack(indices_np.flatten(), self.bits)

        header = WireHeader(
            bits=self.bits,
            seed=self.seed,
            use_qjl=self.use_qjl,
            qjl_sketch_size=self.qjl_sketch_size,
            original_shape=per_head_shape,  # store per-head shape; deserializer merges back
            original_dtype=original_dtype,
        )
        compressed = encode(header, packed, norms_np, sketch_np, rn_np)

        n = len(compressed)
        data = torch.from_numpy(np.frombuffer(compressed, dtype=np.uint8).copy())
        dst.tensor.ravel()[:n].copy_(data)
        return n


class TqaiDeserializer(_LMDeserializer):
    """Decompress tqai-encoded bytes from src.tensor into dst.tensor.

    All compression parameters (bits, seed …) are read from the wire
    format embedded in src.tensor — this class needs no quantizer config.
    """

    def deserialize(self, src: MemoryObj, dst: MemoryObj) -> None:
        """Decompress src.tensor into dst.tensor (KV-shaped tensor)."""
        raw = src.tensor.ravel().cpu().numpy().tobytes()
        header, packed, norms_flat, qjl_raw = decode(raw)

        head_dim = header.original_shape[-1]
        q = _get_quantizer(
            head_dim, header.bits, header.seed,
            header.use_qjl, header.qjl_sketch_size, "cpu",
        )

        total = int(np.prod(header.original_shape))
        indices_np = unpack(packed, header.bits, shape=(total,)).reshape(header.original_shape)
        norms_shape = header.original_shape[:-1] + (1,)
        norms_np = norms_flat.reshape(norms_shape)

        indices_t = torch.from_numpy(indices_np.astype(np.uint8))
        norms_t = torch.from_numpy(norms_np.astype(np.float16))

        qjl_tuple: Optional[tuple] = None
        if header.use_qjl and qjl_raw is not None:
            sketch_np, rn_np = qjl_raw
            sketch_shape = header.original_shape[:-1] + (header.qjl_sketch_size,)
            rn_shape = norms_shape
            qjl_tuple = (
                torch.from_numpy(sketch_np.reshape(sketch_shape).astype(np.int8)),
                torch.from_numpy(rn_np.reshape(rn_shape).astype(np.float16)),
            )

        reconstructed = q.dequantize(indices_t, norms_t, qjl_tuple)
        target_dtype = getattr(torch, header.original_dtype, torch.float32)
        # Wire stores per-head shape [..., num_heads, head_dim].
        # Merge last two dims to recover LMCache's [..., hidden_dim] layout.
        *leading, num_heads, hd = header.original_shape
        result = reconstructed.to(dtype=target_dtype).reshape(*leading, num_heads * hd)
        dst.tensor.copy_(result.to(dst.tensor.device))
