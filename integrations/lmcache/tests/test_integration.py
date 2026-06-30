"""Integration tests against the real LMCache v1 serde interface.

Skipped automatically when lmcache is not installed.
"""

import pytest

lmcache = pytest.importorskip("lmcache", reason="lmcache not installed")

import torch
import torch.nn.functional as F

from lmcache_turbo_quant_serde import (
    TqaiDeserializer,
    TqaiSerializer,
    TurboQuantDeserializer,
    TurboQuantSerializer,
    register,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeMemoryObj:
    """Minimal MemoryObj duck-type (mirrors LMCache's test pattern)."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def set_used_size(self, n: int) -> None:
        self.tensor = self.tensor.ravel()[:n]


def _kv_tensor(
    num_layers: int = 4,
    num_tokens: int = 32,
    num_heads: int = 8,
    head_dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Make a random KV tensor in LMCache's [2, L, T, H*D] layout."""
    hidden_dim = num_heads * head_dim
    torch.manual_seed(42)
    return torch.randn(2, num_layers, num_tokens, hidden_dim, dtype=dtype)


# ---------------------------------------------------------------------------
# Standalone-class conformance
# ---------------------------------------------------------------------------


def test_standalone_serializer_conforms_to_old_lmcache_interface():
    """TurboQuantSerializer must extend lmcache's old Serializer ABC."""
    try:
        from lmcache.storage_backend.serde.serde import Serializer, Deserializer
        assert isinstance(TurboQuantSerializer(), Serializer)
        assert isinstance(TurboQuantDeserializer(), Deserializer)
    except (ImportError, AttributeError):
        pytest.skip("old lmcache serde not present in this version")


def test_v1_serializer_conforms_to_v1_interface():
    """TqaiSerializer / TqaiDeserializer must extend lmcache v1 ABCs."""
    from lmcache.v1.distributed.serde.base import Serializer, Deserializer

    assert isinstance(TqaiSerializer(), Serializer)
    assert isinstance(TqaiDeserializer(), Deserializer)


# ---------------------------------------------------------------------------
# v1 serialize / deserialize via MemoryObj
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [3, 4])
def test_v1_serialize_deserialize(bits):
    """serialize → deserialize via MemoryObj must preserve shape, dtype, quality."""
    from lmcache.v1.distributed.api import MemoryLayoutDesc

    head_dim = 64
    num_layers, num_tokens, num_heads = 4, 32, 8
    hidden_dim = num_heads * head_dim
    kv = _kv_tensor(num_layers, num_tokens, num_heads, head_dim)

    ser = TqaiSerializer(head_dim=head_dim, bits=bits)
    des = TqaiDeserializer()

    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, num_layers, num_tokens, hidden_dim])],
        dtypes=[torch.bfloat16],
    )

    buf_size = ser.estimate_serialized_size(layout)
    buf = torch.zeros(buf_size, dtype=torch.uint8)
    src_obj = _FakeMemoryObj(kv)
    dst_obj = _FakeMemoryObj(buf)

    n = ser.serialize(src_obj, dst_obj)
    assert isinstance(n, int) and 0 < n <= buf_size

    # Shrink to actual written size (mirrors what AsyncSerdeProcessor does)
    dst_obj.set_used_size(n)

    recovered_buf = torch.zeros_like(kv)
    des.deserialize(dst_obj, _FakeMemoryObj(recovered_buf))

    assert recovered_buf.shape == kv.shape
    assert recovered_buf.dtype == kv.dtype

    sim = F.cosine_similarity(
        kv.float().reshape(-1, head_dim),
        recovered_buf.float().reshape(-1, head_dim),
        dim=-1,
    ).mean().item()
    assert sim > 0.95, f"bits={bits}: cosine={sim:.4f}"


def test_v1_estimate_is_upper_bound():
    """estimate_serialized_size must always be >= actual bytes written."""
    from lmcache.v1.distributed.api import MemoryLayoutDesc

    head_dim = 128
    num_layers, num_tokens, num_heads = 8, 64, 16
    hidden_dim = num_heads * head_dim
    kv = _kv_tensor(num_layers, num_tokens, num_heads, head_dim)

    ser = TqaiSerializer(head_dim=head_dim, bits=4)
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, num_layers, num_tokens, hidden_dim])],
        dtypes=[torch.bfloat16],
    )

    estimated = ser.estimate_serialized_size(layout)
    buf = torch.zeros(estimated, dtype=torch.uint8)
    n = ser.serialize(_FakeMemoryObj(kv), _FakeMemoryObj(buf))

    assert n <= estimated, f"wrote {n} bytes but estimate was only {estimated}"


# ---------------------------------------------------------------------------
# register() → register_serde_factory
# ---------------------------------------------------------------------------


def test_register_adds_tqai_to_factory():
    """register() must make 'tqai' available via create_serde_processor."""
    from lmcache.v1.distributed.serde import create_serde_processor, get_registered_serde_types, SerdeConfig

    register()

    assert "tqai" in get_registered_serde_types()

    proc = create_serde_processor(
        SerdeConfig(type="tqai", kwargs={"head_dim": 128, "bits": 4})
    )
    proc.close()


def test_register_idempotent():
    """Calling register() multiple times must not raise."""
    register()
    register()


def test_register_does_not_break_existing_serdes():
    """After registration, existing serde types must still work."""
    from lmcache.v1.distributed.serde import create_serde_processor, SerdeConfig

    register()

    for serde_type in ("fp8",):  # "fp8" is always shipped with LMCache v1
        try:
            proc = create_serde_processor(
                SerdeConfig(type=serde_type, kwargs={})
            )
            proc.close()
        except Exception as exc:
            pytest.fail(f"'{serde_type}' broke after register(): {exc}")


def test_end_to_end_via_create_serde_processor():
    """Full path: create_serde_processor → serialize → deserialize."""
    from lmcache.v1.distributed.serde import create_serde_processor, SerdeConfig
    from lmcache.v1.distributed.api import MemoryLayoutDesc

    register()

    head_dim = 64
    num_layers, num_tokens, num_heads = 2, 16, 4
    hidden_dim = num_heads * head_dim

    proc = create_serde_processor(
        SerdeConfig(type="tqai", kwargs={"head_dim": head_dim, "bits": 4})
    )

    kv = _kv_tensor(num_layers, num_tokens, num_heads, head_dim)
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, num_layers, num_tokens, hidden_dim])],
        dtypes=[torch.bfloat16],
    )

    buf_size = proc.estimate_serialized_size(layout)
    buf = torch.zeros(buf_size, dtype=torch.uint8)
    n = proc._serializer.serialize(_FakeMemoryObj(kv), _FakeMemoryObj(buf))

    buf_trimmed = buf[:n]
    recovered_kv = torch.zeros_like(kv)
    proc._deserializer.deserialize(_FakeMemoryObj(buf_trimmed), _FakeMemoryObj(recovered_kv))

    assert recovered_kv.shape == kv.shape
    sim = F.cosine_similarity(
        kv.float().reshape(-1, head_dim),
        recovered_kv.float().reshape(-1, head_dim),
        dim=-1,
    ).mean().item()
    assert sim > 0.95, f"cosine={sim:.4f}"

    proc.close()
