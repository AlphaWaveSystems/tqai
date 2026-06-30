"""Self-describing binary wire format for TurboQuant-compressed KV tensors.

Layout (all integers little-endian):
  [4]           magic      b"TQ01"
  [1]           version    uint8 = 1
  [1]           bits       uint8  — quantizer bits (2|3|4|6|8)
  [1]           use_qjl    uint8  — 1 if QJL residual is present
  [2]           qjl_sz     uint16 — QJL sketch size (ignored when use_qjl=0)
  [4]           seed       int32  — quantizer RNG seed
  [1]           ndim       uint8
  [ndim*8]      shape      int64 × ndim — original tensor shape
  [1]           dlen       uint8  — len of dtype string
  [dlen]        dtype      ASCII  — e.g. "bfloat16"
  [8]           pi_nbytes  int64  — byte count of packed indices block
  [pi_nbytes]   indices    uint8  — bit-packed Lloyd-Max indices (tqai.packing)
  [8]           nm_nbytes  int64  — byte count of norms block
  [nm_nbytes]   norms      raw float16 C-order bytes
  -- if use_qjl --
  [8]           sk_nbytes  int64
  [sk_nbytes]   sketch     raw int8 C-order bytes
  [8]           rn_nbytes  int64
  [rn_nbytes]   res_norms  raw float16 C-order bytes
"""

from __future__ import annotations

import struct
from typing import NamedTuple, Optional

import numpy as np

MAGIC = b"TQ01"
VERSION = 1

# Fixed-size prefix before the variable-length shape field
_PREFIX_FMT = "<4sBBBHi"  # magic(4) version(1) bits(1) use_qjl(1) qjl_sz(2) seed(4)
_PREFIX_SIZE = struct.calcsize(_PREFIX_FMT)  # 13


class WireHeader(NamedTuple):
    bits: int
    seed: int
    use_qjl: bool
    qjl_sketch_size: int
    original_shape: tuple[int, ...]
    original_dtype: str   # e.g. "bfloat16"


def encode(
    header: WireHeader,
    packed_indices: np.ndarray,        # 1-D uint8, bit-packed
    norms: np.ndarray,                 # float16, C-order
    sketch: Optional[np.ndarray],      # int8, C-order  — None when use_qjl=False
    residual_norms: Optional[np.ndarray],  # float16       — None when use_qjl=False
) -> bytes:
    dtype_b = header.original_dtype.encode("ascii")
    shape_b = struct.pack(f"<{len(header.original_shape)}q", *header.original_shape)

    buf = bytearray()

    # Fixed prefix
    buf += struct.pack(
        _PREFIX_FMT,
        MAGIC,
        VERSION,
        header.bits,
        int(header.use_qjl),
        header.qjl_sketch_size,
        header.seed,
    )

    # Shape
    buf += struct.pack("<B", len(header.original_shape))
    buf += shape_b

    # dtype string
    buf += struct.pack("<B", len(dtype_b))
    buf += dtype_b

    # Packed indices
    pi_b = bytes(packed_indices)
    buf += struct.pack("<q", len(pi_b))
    buf += pi_b

    # Norms (store as float16)
    nm_b = norms.astype(np.float16).tobytes()
    buf += struct.pack("<q", len(nm_b))
    buf += nm_b

    # Optional QJL data
    if header.use_qjl:
        assert sketch is not None and residual_norms is not None
        sk_b = sketch.astype(np.int8).tobytes()
        buf += struct.pack("<q", len(sk_b))
        buf += sk_b
        rn_b = residual_norms.astype(np.float16).tobytes()
        buf += struct.pack("<q", len(rn_b))
        buf += rn_b

    return bytes(buf)


def decode(
    bs: bytes,
) -> tuple[WireHeader, np.ndarray, np.ndarray, Optional[tuple[np.ndarray, np.ndarray]]]:
    """Returns (header, packed_indices, norms, qjl_data_or_None)."""
    off = 0

    magic, version, bits, use_qjl_int, qjl_sz, seed = struct.unpack_from(_PREFIX_FMT, bs, off)
    off += _PREFIX_SIZE

    if magic != MAGIC:
        raise ValueError(f"Bad magic bytes: {magic!r} — expected {MAGIC!r}")
    if version != VERSION:
        raise ValueError(f"Unsupported wire version {version}")

    use_qjl = bool(use_qjl_int)

    ndim = struct.unpack_from("<B", bs, off)[0]; off += 1
    shape: tuple[int, ...] = struct.unpack_from(f"<{ndim}q", bs, off)
    off += ndim * 8

    dlen = struct.unpack_from("<B", bs, off)[0]; off += 1
    dtype_str = bs[off: off + dlen].decode("ascii"); off += dlen

    pi_nbytes = struct.unpack_from("<q", bs, off)[0]; off += 8
    packed_indices = np.frombuffer(bs[off: off + pi_nbytes], dtype=np.uint8).copy()
    off += pi_nbytes

    nm_nbytes = struct.unpack_from("<q", bs, off)[0]; off += 8
    norms = np.frombuffer(bs[off: off + nm_nbytes], dtype=np.float16).copy()
    off += nm_nbytes

    qjl_data: Optional[tuple[np.ndarray, np.ndarray]] = None
    if use_qjl:
        sk_nbytes = struct.unpack_from("<q", bs, off)[0]; off += 8
        sketch = np.frombuffer(bs[off: off + sk_nbytes], dtype=np.int8).copy()
        off += sk_nbytes
        rn_nbytes = struct.unpack_from("<q", bs, off)[0]; off += 8
        residual_norms = np.frombuffer(bs[off: off + rn_nbytes], dtype=np.float16).copy()
        qjl_data = (sketch, residual_norms)

    header = WireHeader(
        bits=bits,
        seed=seed,
        use_qjl=use_qjl,
        qjl_sketch_size=qjl_sz,
        original_shape=shape,
        original_dtype=dtype_str,
    )
    return header, packed_indices, norms, qjl_data
