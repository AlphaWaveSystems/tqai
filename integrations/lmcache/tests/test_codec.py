"""Integration tests for TurboQuantSerializer / TurboQuantDeserializer.

Requires tqai to be installed (from the repo src/).
LMCache is optional; tests that need it are marked lmcache.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lmcache_turbo_quant_serde import TurboQuantDeserializer, TurboQuantSerializer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _roundtrip(t: torch.Tensor, bits: int = 4, **kw) -> torch.Tensor:
    ser = TurboQuantSerializer(bits=bits, **kw)
    des = TurboQuantDeserializer()
    return des.from_bytes(ser.to_bytes(t))


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1, a.shape[-1])
    b = b.float().reshape(-1, b.shape[-1])
    return F.cosine_similarity(a, b, dim=-1).mean().item()


# ---------------------------------------------------------------------------
# Shape / dtype parametrization
# ---------------------------------------------------------------------------


SHAPES = [
    (32, 128),                   # (tokens, head_dim)
    (2, 32, 64),                 # (batch, tokens, head_dim)
    (4, 2, 10, 8, 64),           # (layers, kv, tokens, heads, head_dim)
    (1, 2, 100, 16, 128),        # typical vLLM shape, head_dim=128
]

DTYPES = [torch.float16, torch.bfloat16, torch.float32]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("bits", [2, 4])
def test_roundtrip_shape(shape, bits):
    x = torch.randn(*shape)
    r = _roundtrip(x, bits=bits)
    assert r.shape == x.shape, f"Shape mismatch: {r.shape} != {x.shape}"


@pytest.mark.parametrize("dtype", DTYPES)
def test_roundtrip_dtype_preserved(dtype):
    x = torch.randn(4, 8, 128).to(dtype)
    r = _roundtrip(x, bits=4)
    assert r.dtype == dtype, f"dtype mismatch: {r.dtype} != {dtype}"


@pytest.mark.parametrize("bits,min_cos", [
    (8, 0.9999),
    (4, 0.97),
    (3, 0.95),
    (2, 0.90),
])
def test_reconstruction_quality(bits, min_cos):
    """Cosine similarity should exceed min_cos for random unit vectors."""
    torch.manual_seed(0)
    x = torch.randn(64, 128)
    r = _roundtrip(x, bits=bits)
    sim = _cosine_sim(x, r)
    assert sim >= min_cos, f"bits={bits}: cosine={sim:.4f} < {min_cos}"


# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits,max_ratio", [
    (4, 0.35),   # expect < 35% of original size (i.e., >2.8× compression)
    (2, 0.20),   # < 20% of original
])
def test_compression_ratio(bits, max_ratio):
    x = torch.randn(4, 2, 50, 16, 128).to(torch.bfloat16)
    original_bytes = x.numel() * 2  # bfloat16 = 2 bytes/element
    compressed = TurboQuantSerializer(bits=bits).to_bytes(x)
    ratio = len(compressed) / original_bytes
    assert ratio < max_ratio, (
        f"bits={bits}: compressed/original={ratio:.3f} ≥ {max_ratio}"
    )


# ---------------------------------------------------------------------------
# Wire-format self-description
# ---------------------------------------------------------------------------


def test_deserializer_needs_no_config():
    """Deserializer must recover tensor using only wire-format metadata."""
    x = torch.randn(2, 4, 64).to(torch.float16)
    ser = TurboQuantSerializer(bits=3, seed=999)
    des = TurboQuantDeserializer()          # no params

    r = des.from_bytes(ser.to_bytes(x))
    assert r.shape == x.shape
    assert r.dtype == torch.float16


def test_bytearray_input():
    """from_bytes must accept bytearray as well as bytes (per LMCache contract)."""
    x = torch.randn(8, 128)
    ser = TurboQuantSerializer(bits=4)
    des = TurboQuantDeserializer()

    data = bytearray(ser.to_bytes(x))
    r = des.from_bytes(data)
    assert r.shape == x.shape


def test_deterministic_output():
    """Same tensor → same compressed bytes → same reconstructed tensor."""
    x = torch.randn(4, 64)
    ser = TurboQuantSerializer(bits=4, seed=7)
    b1 = ser.to_bytes(x)
    b2 = ser.to_bytes(x)
    assert b1 == b2


# ---------------------------------------------------------------------------
# QJL path
# ---------------------------------------------------------------------------


def test_qjl_roundtrip():
    x = torch.randn(8, 128)
    r = _roundtrip(x, bits=4, use_qjl=True, qjl_sketch_size=64)
    assert r.shape == x.shape
    sim = _cosine_sim(x, r)
    assert sim > 0.97, f"QJL cosine={sim:.4f}"


