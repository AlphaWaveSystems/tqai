"""End-to-end tests: lmcache_turbo_quant_serde ↔ LMCache v1 StorageManager pattern.

Mirrors the structure of lmcache's own test_turboquant_storage_manager_roundtrip,
exercising the full plugin lifecycle:
  register() → create_serde_processor() → async serialize/deserialize roundtrip.
"""

import asyncio

import pytest

pytest.importorskip("lmcache", reason="lmcache not installed")

import torch
import torch.nn.functional as F

import lmcache_turbo_quant_serde
from lmcache_turbo_quant_serde import register

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MemoryObj:
    """Minimal MemoryObj duck-type for testing without a full LMCache engine."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self._used: int = tensor.numel()

    def set_used_size(self, n: int) -> None:
        self._used = n
        self.tensor = self.tensor.ravel()[:n]


def _make_kv(
    num_layers: int = 4,
    num_tokens: int = 64,
    num_heads: int = 8,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> torch.Tensor:
    """Return a random KV tensor in LMCache's [2, L, T, H×D] layout."""
    torch.manual_seed(seed)
    return torch.randn(2, num_layers, num_tokens, num_heads * head_dim, dtype=dtype)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor, head_dim: int) -> float:
    return (
        F.cosine_similarity(
            a.float().reshape(-1, head_dim),
            b.float().reshape(-1, head_dim),
            dim=-1,
        )
        .mean()
        .item()
    )


# ---------------------------------------------------------------------------
# Sync roundtrip via TqaiSerializer / TqaiDeserializer directly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_layers,num_tokens,num_heads,head_dim,bits",
    [
        (4, 64, 8, 128, 4),   # typical LLaMA-3 8B layer config
        (4, 64, 8, 128, 3),   # 3-bit aggressive
        (2, 32, 4, 64, 4),    # smaller model
        (8, 128, 16, 128, 4), # large model with 16 KV heads
    ],
    ids=["llama3-4bit", "llama3-3bit", "small-4bit", "large-4bit"],
)
def test_sync_roundtrip(num_layers, num_tokens, num_heads, head_dim, bits):
    """Direct TqaiSerializer/TqaiDeserializer roundtrip preserves shape+dtype+quality."""
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache_turbo_quant_serde import TqaiDeserializer, TqaiSerializer

    hidden_dim = num_heads * head_dim
    kv = _make_kv(num_layers, num_tokens, num_heads, head_dim)

    ser = TqaiSerializer(head_dim=head_dim, bits=bits)
    des = TqaiDeserializer()

    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, num_layers, num_tokens, hidden_dim])],
        dtypes=[kv.dtype],
    )
    buf_size = ser.estimate_serialized_size(layout)
    buf = torch.zeros(buf_size, dtype=torch.uint8)

    src_obj = _MemoryObj(kv)
    dst_obj = _MemoryObj(buf)

    n = ser.serialize(src_obj, dst_obj)
    assert 0 < n <= buf_size, f"wrote {n} bytes into {buf_size}-byte buffer"

    # Narrow buffer to actual written bytes (mirrors AsyncSerdeProcessor)
    dst_obj.set_used_size(n)

    recovered = torch.zeros_like(kv)
    des.deserialize(dst_obj, _MemoryObj(recovered))

    assert recovered.shape == kv.shape, "shape mismatch after roundtrip"
    assert recovered.dtype == kv.dtype, "dtype mismatch after roundtrip"

    sim = _cosine_sim(kv, recovered, head_dim)
    threshold = 0.97 if bits >= 4 else 0.95
    assert sim > threshold, (
        f"{num_layers}L/{num_tokens}T/{num_heads}H/{head_dim}D bits={bits}: "
        f"cosine={sim:.4f} < {threshold}"
    )


# ---------------------------------------------------------------------------
# Async roundtrip via AsyncSerdeProcessor (the real LMCache path)
# ---------------------------------------------------------------------------


def test_async_processor_roundtrip():
    """Full async path via create_serde_processor (production code path)."""
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.distributed.serde import SerdeConfig, create_serde_processor

    register()

    head_dim, num_layers, num_tokens, num_heads = 128, 4, 64, 8
    hidden_dim = num_heads * head_dim
    kv = _make_kv(num_layers, num_tokens, num_heads, head_dim)

    proc = create_serde_processor(
        SerdeConfig(type="tqai", kwargs={"head_dim": head_dim, "bits": 4})
    )

    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, num_layers, num_tokens, hidden_dim])],
        dtypes=[kv.dtype],
    )
    buf_size = proc.estimate_serialized_size(layout)
    buf = torch.zeros(buf_size, dtype=torch.uint8)

    n = proc._serializer.serialize(_MemoryObj(kv), _MemoryObj(buf))
    buf_trimmed = buf[:n]

    recovered = torch.zeros_like(kv)
    proc._deserializer.deserialize(_MemoryObj(buf_trimmed), _MemoryObj(recovered))

    sim = _cosine_sim(kv, recovered, head_dim)
    assert sim > 0.97, f"async path cosine={sim:.4f}"

    proc.close()


# ---------------------------------------------------------------------------
# Compression ratio assertions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits,max_ratio", [(4, 0.38), (3, 0.29), (2, 0.22)])
def test_compression_ratio_e2e(bits, max_ratio):
    """Compressed size must be well below the fp16 original."""
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache_turbo_quant_serde import TqaiSerializer

    head_dim, num_layers, num_tokens, num_heads = 128, 4, 64, 8
    hidden_dim = num_heads * head_dim
    kv = _make_kv(num_layers, num_tokens, num_heads, head_dim)

    ser = TqaiSerializer(head_dim=head_dim, bits=bits)
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, num_layers, num_tokens, hidden_dim])],
        dtypes=[kv.dtype],
    )
    buf_size = ser.estimate_serialized_size(layout)
    buf = torch.zeros(buf_size, dtype=torch.uint8)

    n = ser.serialize(_MemoryObj(kv), _MemoryObj(buf))

    original_bytes = kv.numel() * 2  # bfloat16 = 2 bytes
    ratio = n / original_bytes
    assert ratio < max_ratio, (
        f"bits={bits}: compression ratio {ratio:.3f} >= {max_ratio} "
        f"(wrote {n} bytes vs {original_bytes} original)"
    )
