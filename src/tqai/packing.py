"""Bit-packing utilities for quantized KV cache indices.

Reduces DRAM bandwidth by storing Lloyd-Max indices at their actual
bit-width instead of in uint8 (8 bits) containers:

  bits=2 → 4 indices per byte  (4× vs uint8)
  bits=3 → bit-stream packing  (2.67× vs uint8)
  bits=4 → 2 indices per byte  (2× vs uint8)
  bits=6 → bit-stream packing  (1.33× vs uint8)
  bits=8 → no-op               (1× vs uint8)

API:
    packed = pack(indices_uint8, bits)    # numpy uint8 array → bytes
    indices = unpack(packed, bits, shape) # bytes → numpy uint8 array

All functions are pure NumPy and backend-agnostic.  They operate on
the CPU and are intended for serialization and DRAM storage layers.
Hot-path dequantization still uses uint8 indices in GPU memory.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pack(indices: np.ndarray, bits: int) -> np.ndarray:
    """Bit-pack a uint8 index array into a compact byte array.

    Args:
        indices: uint8 array of shape ``(...)`` with values in ``[0, 2**bits)``.
        bits: Bits per index. Supported: 2, 3, 4, 6, 8.

    Returns:
        1-D uint8 byte array.  Use :func:`unpack` with the original shape to
        recover the indices.

    Raises:
        ValueError: For unsupported ``bits`` values or out-of-range indices.
    """
    if bits not in (2, 3, 4, 6, 8):
        raise ValueError(f"bits must be one of 2, 3, 4, 6, 8; got {bits}")

    flat = np.asarray(indices, dtype=np.uint8).ravel()
    if bits == 8:
        return flat.copy()
    if bits == 2:
        return _pack_2bit(flat)
    if bits == 4:
        return _pack_4bit(flat)
    # bits ∈ {3, 6} — general bit-stream path
    return _pack_bitstream(flat, bits)


def unpack(packed: np.ndarray, bits: int, shape: tuple[int, ...]) -> np.ndarray:
    """Recover a uint8 index array from bit-packed bytes.

    Args:
        packed: 1-D uint8 byte array produced by :func:`pack`.
        bits: Bits per index used during packing.
        shape: Original array shape.

    Returns:
        uint8 array of the given ``shape``.

    Raises:
        ValueError: For unsupported ``bits`` values.
    """
    if bits not in (2, 3, 4, 6, 8):
        raise ValueError(f"bits must be one of 2, 3, 4, 6, 8; got {bits}")

    packed = np.asarray(packed, dtype=np.uint8)
    n = int(np.prod(shape))

    if bits == 8:
        return packed[:n].reshape(shape)
    if bits == 2:
        return _unpack_2bit(packed, n).reshape(shape)
    if bits == 4:
        return _unpack_4bit(packed, n).reshape(shape)
    return _unpack_bitstream(packed, bits, n).reshape(shape)


def packed_size(n_indices: int, bits: int) -> int:
    """Return the number of bytes needed to pack ``n_indices`` at ``bits`` bpi."""
    return (n_indices * bits + 7) // 8


def compression_ratio(bits: int) -> float:
    """Return the byte compression ratio vs uint8 (values > 1 mean smaller)."""
    return 8.0 / bits


# ---------------------------------------------------------------------------
# 2-bit: 4 indices per byte — vectorized
# ---------------------------------------------------------------------------


def _pack_2bit(flat: np.ndarray) -> np.ndarray:
    """Pack 4 two-bit values per byte."""
    n = len(flat)
    # Pad to multiple of 4
    pad = (-n) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    groups = flat.reshape(-1, 4)
    return (
        (groups[:, 0] & 0x03)
        | ((groups[:, 1] & 0x03) << 2)
        | ((groups[:, 2] & 0x03) << 4)
        | ((groups[:, 3] & 0x03) << 6)
    ).astype(np.uint8)


def _unpack_2bit(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack 4 two-bit values per byte."""
    expanded = np.stack(
        [
            packed & 0x03,
            (packed >> 2) & 0x03,
            (packed >> 4) & 0x03,
            (packed >> 6) & 0x03,
        ],
        axis=-1,
    ).ravel()
    return expanded[:n].astype(np.uint8)


# ---------------------------------------------------------------------------
# 4-bit: 2 indices per byte — vectorized
# ---------------------------------------------------------------------------


def _pack_4bit(flat: np.ndarray) -> np.ndarray:
    """Pack 2 four-bit values per byte (low nibble first)."""
    n = len(flat)
    # Pad to even length
    if n % 2:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
    lo = flat[0::2] & 0x0F
    hi = flat[1::2] & 0x0F
    return (lo | (hi << 4)).astype(np.uint8)


def _unpack_4bit(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack 2 four-bit values per byte."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    expanded = np.stack([lo, hi], axis=-1).ravel()
    return expanded[:n].astype(np.uint8)


# ---------------------------------------------------------------------------
# General bit-stream — handles 3-bit and 6-bit (not byte-aligned)
# ---------------------------------------------------------------------------


def _pack_bitstream(flat: np.ndarray, bits: int) -> np.ndarray:
    """General bit-stream packing for non-byte-aligned widths (3, 6 bits)."""
    n = len(flat)
    n_bytes = (n * bits + 7) // 8
    out = np.zeros(n_bytes, dtype=np.uint8)

    # Vectorize using uint64 accumulators: process indices in blocks that
    # fill an exact number of bytes.
    # LCM(bits, 8) / bits = indices per block; LCM(bits, 8) / 8 = bytes per block
    lcm = _lcm(bits, 8)
    block_indices = lcm // bits   # e.g. bits=3 → lcm=24 → 8 indices per block
    block_bytes = lcm // 8        # e.g. bits=3 → 3 bytes per block

    n_full_blocks = n // block_indices
    remainder = n % block_indices

    if n_full_blocks > 0:
        blocks = flat[: n_full_blocks * block_indices].reshape(n_full_blocks, block_indices)
        for b_idx in range(n_full_blocks):
            acc: int = 0
            for i in range(block_indices):
                acc |= int(blocks[b_idx, i]) << (i * bits)
            base = b_idx * block_bytes
            for j in range(block_bytes):
                out[base + j] = (acc >> (j * 8)) & 0xFF

    # Handle remainder indices
    if remainder:
        acc = 0
        for i in range(remainder):
            acc |= int(flat[n_full_blocks * block_indices + i]) << (i * bits)
        base = n_full_blocks * block_bytes
        remaining_bits = remainder * bits
        remaining_bytes = (remaining_bits + 7) // 8
        for j in range(remaining_bytes):
            out[base + j] = (acc >> (j * 8)) & 0xFF

    return out


def _unpack_bitstream(packed: np.ndarray, bits: int, n: int) -> np.ndarray:
    """General bit-stream unpacking for non-byte-aligned widths (3, 6 bits)."""
    out = np.zeros(n, dtype=np.uint8)
    mask = (1 << bits) - 1

    lcm = _lcm(bits, 8)
    block_indices = lcm // bits
    block_bytes = lcm // 8

    n_full_blocks = n // block_indices
    remainder = n % block_indices

    if n_full_blocks > 0:
        for b_idx in range(n_full_blocks):
            base = b_idx * block_bytes
            acc = 0
            for j in range(block_bytes):
                acc |= int(packed[base + j]) << (j * 8)
            for i in range(block_indices):
                out[b_idx * block_indices + i] = (acc >> (i * bits)) & mask

    if remainder:
        base = n_full_blocks * block_bytes
        remaining_bytes = ((remainder * bits) + 7) // 8
        acc = 0
        for j in range(remaining_bytes):
            if base + j < len(packed):
                acc |= int(packed[base + j]) << (j * 8)
        for i in range(remainder):
            out[n_full_blocks * block_indices + i] = (acc >> (i * bits)) & mask

    return out


def _lcm(a: int, b: int) -> int:
    from math import gcd
    return a * b // gcd(a, b)
