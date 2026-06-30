"""Unit tests for the wire format (no tqai / lmcache dependency)."""

import struct

import numpy as np
import pytest

from lmcache_turbo_quant_serde._wire import MAGIC, VERSION, WireHeader, decode, encode


def _make_header(**kw) -> WireHeader:
    defaults = dict(
        bits=4,
        seed=42,
        use_qjl=False,
        qjl_sketch_size=64,
        original_shape=(2, 8, 128),
        original_dtype="bfloat16",
    )
    defaults.update(kw)
    return WireHeader(**defaults)


def _dummy_data(shape, bits):
    n = int(np.prod(shape))
    indices = np.random.randint(0, 2**bits, size=n, dtype=np.uint8)
    from tqai.packing import pack
    packed = pack(indices, bits)
    norms = np.random.randn(n // shape[-1]).astype(np.float16)
    return packed, norms


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
def test_roundtrip_no_qjl(bits):
    shape = (4, 16, 64)
    header = _make_header(bits=bits, original_shape=shape)
    packed, norms = _dummy_data(shape, bits)

    bs = encode(header, packed, norms, None, None)
    h2, pi2, nm2, qjl2 = decode(bs)

    assert h2.bits == bits
    assert h2.original_shape == shape
    assert h2.original_dtype == "bfloat16"
    assert h2.seed == 42
    assert not h2.use_qjl
    assert qjl2 is None
    np.testing.assert_array_equal(pi2, packed)
    np.testing.assert_array_equal(nm2, norms)


def test_roundtrip_with_qjl():
    shape = (2, 4, 128)
    n_vecs = 2 * 4
    qjl_sz = 32
    header = _make_header(bits=4, original_shape=shape, use_qjl=True, qjl_sketch_size=qjl_sz)
    packed, norms = _dummy_data(shape, 4)
    sketch = np.random.randint(-1, 2, size=(n_vecs * qjl_sz,), dtype=np.int8)
    rn = np.random.randn(n_vecs).astype(np.float16)

    bs = encode(header, packed, norms, sketch, rn)
    h2, pi2, nm2, qjl2 = decode(bs)

    assert h2.use_qjl
    assert qjl2 is not None
    np.testing.assert_array_equal(qjl2[0], sketch)
    np.testing.assert_array_equal(qjl2[1], rn)


def test_bad_magic_raises():
    shape = (1, 2, 64)
    header = _make_header(original_shape=shape)
    packed, norms = _dummy_data(shape, 4)
    bs = bytearray(encode(header, packed, norms, None, None))
    bs[:4] = b"XXXX"
    with pytest.raises(ValueError, match="magic"):
        decode(bytes(bs))


def test_dtype_strings_preserved():
    for dtype in ("float32", "float16", "bfloat16"):
        shape = (1, 4, 64)
        header = _make_header(original_shape=shape, original_dtype=dtype)
        packed, norms = _dummy_data(shape, 4)
        bs = encode(header, packed, norms, None, None)
        h2, *_ = decode(bs)
        assert h2.original_dtype == dtype


def test_large_shape():
    shape = (32, 2, 1024, 32, 128)
    header = _make_header(original_shape=shape)
    n = int(np.prod(shape))
    from tqai.packing import pack
    packed = pack(np.zeros(n, dtype=np.uint8), 4)
    norms = np.zeros(n // 128, dtype=np.float16)
    bs = encode(header, packed, norms, None, None)
    h2, *_ = decode(bs)
    assert h2.original_shape == shape
