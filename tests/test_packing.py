"""Tests for tqai.packing — bit-packing / unpacking of quantized indices."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from tqai.packing import (
    compression_ratio,
    pack,
    packed_size,
    unpack,
)

# ---------------------------------------------------------------------------
# Parametrize over all supported bit-widths
# ---------------------------------------------------------------------------

ALL_BITS = [2, 3, 4, 6, 8]
VECTORIZED_BITS = [2, 4]
BITSTREAM_BITS = [3, 6]


# ---------------------------------------------------------------------------
# Roundtrip correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", ALL_BITS)
@pytest.mark.parametrize("n", [1, 3, 4, 7, 8, 15, 16, 63, 64, 100, 128, 255, 256])
def test_roundtrip_1d(bits, n):
    """pack → unpack recovers the original indices for all shapes."""
    rng = np.random.default_rng(42)
    orig = rng.integers(0, 2**bits, size=n, dtype=np.uint8)
    packed = pack(orig, bits)
    recovered = unpack(packed, bits, orig.shape)
    npt.assert_array_equal(recovered, orig)


@pytest.mark.parametrize("bits", ALL_BITS)
def test_roundtrip_multidim(bits):
    """Multidimensional shapes are preserved through pack→unpack."""
    rng = np.random.default_rng(7)
    orig = rng.integers(0, 2**bits, size=(4, 32, 16), dtype=np.uint8)
    packed = pack(orig, bits)
    recovered = unpack(packed, bits, orig.shape)
    npt.assert_array_equal(recovered, orig)


@pytest.mark.parametrize("bits", ALL_BITS)
def test_roundtrip_all_values(bits):
    """Every valid index value (0..2^bits-1) survives pack→unpack."""
    n_levels = 2**bits
    orig = np.tile(np.arange(n_levels, dtype=np.uint8), 4)
    packed = pack(orig, bits)
    recovered = unpack(packed, bits, orig.shape)
    npt.assert_array_equal(recovered, orig)


# ---------------------------------------------------------------------------
# Packed size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits,n,expected", [
    (2, 4, 1),
    (2, 5, 2),
    (4, 2, 1),
    (4, 3, 2),
    (3, 8, 3),     # 3*8=24 bits = 3 bytes
    (3, 9, 4),     # 3*9=27 bits → 4 bytes
    (6, 4, 3),     # 6*4=24 bits = 3 bytes
    (8, 10, 10),
])
def test_packed_size(bits, n, expected):
    assert packed_size(n, bits) == expected


@pytest.mark.parametrize("bits", ALL_BITS)
@pytest.mark.parametrize("n", [7, 64, 128])
def test_packed_size_matches_output(bits, n):
    """packed_size() agrees with the actual byte length from pack()."""
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 2**bits, size=n, dtype=np.uint8)
    assert len(pack(orig, bits)) == packed_size(n, bits)


# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits,expected", [
    (2, 4.0),
    (4, 2.0),
    (8, 1.0),
])
def test_compression_ratio(bits, expected):
    assert compression_ratio(bits) == pytest.approx(expected)


@pytest.mark.parametrize("bits", ALL_BITS)
def test_compression_ratio_vs_uint8_storage(bits):
    """Packed array is (8/bits) times smaller than uint8 storage (for byte-aligned)."""
    n = 64
    rng = np.random.default_rng(99)
    orig = rng.integers(0, 2**bits, size=n, dtype=np.uint8)
    packed = pack(orig, bits)
    actual_ratio = n / len(packed)
    # Allow up to 1 byte overhead for padding
    assert actual_ratio >= 8 / bits - (1 / len(packed))


# ---------------------------------------------------------------------------
# Output dtype and shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", ALL_BITS)
def test_pack_output_is_uint8(bits):
    orig = np.array([1, 0, 1], dtype=np.uint8)
    packed = pack(orig, bits)
    assert packed.dtype == np.uint8


@pytest.mark.parametrize("bits", ALL_BITS)
def test_unpack_output_is_uint8(bits):
    orig = np.array([0, 1, 2, 3], dtype=np.uint8)
    packed = pack(orig, bits)
    recovered = unpack(packed, bits, orig.shape)
    assert recovered.dtype == np.uint8


@pytest.mark.parametrize("bits", ALL_BITS)
def test_unpack_output_shape(bits):
    orig = np.zeros((3, 7, 5), dtype=np.uint8)
    packed = pack(orig, bits)
    recovered = unpack(packed, bits, orig.shape)
    assert recovered.shape == orig.shape


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", ALL_BITS)
def test_single_element(bits):
    orig = np.array([2**bits - 1], dtype=np.uint8)  # max value
    packed = pack(orig, bits)
    recovered = unpack(packed, bits, orig.shape)
    npt.assert_array_equal(recovered, orig)


@pytest.mark.parametrize("bits", ALL_BITS)
def test_all_zeros(bits):
    orig = np.zeros(64, dtype=np.uint8)
    packed = pack(orig, bits)
    recovered = unpack(packed, bits, orig.shape)
    npt.assert_array_equal(recovered, orig)


@pytest.mark.parametrize("bits", ALL_BITS)
def test_all_max_values(bits):
    orig = np.full(64, 2**bits - 1, dtype=np.uint8)
    packed = pack(orig, bits)
    recovered = unpack(packed, bits, orig.shape)
    npt.assert_array_equal(recovered, orig)


def test_invalid_bits():
    with pytest.raises(ValueError, match="bits must be"):
        pack(np.zeros(4, dtype=np.uint8), bits=5)
    with pytest.raises(ValueError, match="bits must be"):
        unpack(np.zeros(4, dtype=np.uint8), bits=5, shape=(4,))


# ---------------------------------------------------------------------------
# 4-bit: realistic KV cache sizes
# ---------------------------------------------------------------------------


def test_4bit_kv_cache_roundtrip():
    """Simulate packing a real-sized KV cache slice: T_kv=1024, D=128."""
    T_kv, D, bits = 1024, 128, 4
    rng = np.random.default_rng(11)
    indices = rng.integers(0, 2**bits, size=(T_kv, D), dtype=np.uint8)
    packed = pack(indices, bits)
    recovered = unpack(packed, bits, indices.shape)
    npt.assert_array_equal(recovered, indices)
    # Verify 2x compression
    assert len(packed) == T_kv * D // 2


def test_2bit_kv_cache_roundtrip():
    """Simulate packing at 2-bit: should be 4x vs uint8."""
    T_kv, D, bits = 512, 64, 2
    rng = np.random.default_rng(22)
    indices = rng.integers(0, 2**bits, size=(T_kv, D), dtype=np.uint8)
    packed = pack(indices, bits)
    recovered = unpack(packed, bits, indices.shape)
    npt.assert_array_equal(recovered, indices)
    assert len(packed) == T_kv * D // 4
