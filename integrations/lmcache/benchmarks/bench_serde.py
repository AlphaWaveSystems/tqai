"""Benchmark: tqai serde vs fp16 baseline for LMCache v1 KV-cache transfer.

Measures:
  - Compression ratio (bits/variant × model size)
  - Reconstruction quality (cosine similarity per head)
  - Serialization throughput (GB/s of input KV data)
  - Deserialization throughput (GB/s of recovered KV data)

Run:
    python benchmarks/bench_serde.py

Results are printed in a Markdown table so they can be pasted directly into
a blog post or PR description.

Requirements (in addition to tqai):
    pip install lmcache torch numpy tabulate
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")  # ensure local src/ is on path when run from repo root

from lmcache_turbo_quant_serde import TqaiDeserializer, TqaiSerializer, register
from lmcache_turbo_quant_serde._codec import TurboQuantDeserializer, TurboQuantSerializer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WARMUP_ITERS = 3
BENCH_ITERS = 10

CONFIGS = [
    # (name, num_layers, num_tokens, num_heads, head_dim)
    ("LLaMA-3 8B (small seq)", 32, 64, 8, 128),
    ("LLaMA-3 8B (medium seq)", 32, 256, 8, 128),
    ("LLaMA-3 8B (long seq)", 32, 1024, 8, 128),
    ("Mistral 7B", 32, 256, 8, 128),
]

BITS_LIST = [4, 3, 2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MemObj:
    def __init__(self, t: torch.Tensor) -> None:
        self.tensor = t

    def set_used_size(self, n: int) -> None:
        self.tensor = self.tensor.ravel()[:n]


def _make_kv(num_layers, num_tokens, num_heads, head_dim, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    return torch.randn(2, num_layers, num_tokens, num_heads * head_dim, dtype=dtype)


def _cosine_sim(original: torch.Tensor, recovered: torch.Tensor, head_dim: int) -> float:
    return (
        F.cosine_similarity(
            original.float().reshape(-1, head_dim),
            recovered.float().reshape(-1, head_dim),
            dim=-1,
        )
        .mean()
        .item()
    )


# ---------------------------------------------------------------------------
# tqai v1 benchmark helper
# ---------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    bits: int
    num_layers: int
    num_tokens: int
    num_heads: int
    head_dim: int
    original_bytes: int
    compressed_bytes: int
    cosine_sim: float
    ser_ms: float  # median over BENCH_ITERS
    des_ms: float  # median over BENCH_ITERS
    ser_times: List[float] = field(default_factory=list)
    des_times: List[float] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        return self.compressed_bytes / self.original_bytes

    @property
    def ser_gbps(self) -> float:
        """GB/s of input KV data serialized."""
        return (self.original_bytes / 1e9) / (self.ser_ms / 1e3)

    @property
    def des_gbps(self) -> float:
        """GB/s of output KV data deserialized."""
        return (self.original_bytes / 1e9) / (self.des_ms / 1e3)


def _bench_v1(name, num_layers, num_tokens, num_heads, head_dim, bits) -> Result:
    hidden_dim = num_heads * head_dim
    kv = _make_kv(num_layers, num_tokens, num_heads, head_dim)
    original_bytes = kv.numel() * 2  # bfloat16 = 2 bytes

    try:
        from lmcache.v1.distributed.api import MemoryLayoutDesc
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([2, num_layers, num_tokens, hidden_dim])],
            dtypes=[kv.dtype],
        )
        ser = TqaiSerializer(head_dim=head_dim, bits=bits)
        des = TqaiDeserializer()
        buf_size = ser.estimate_serialized_size(layout)
    except ImportError:
        # Fallback to standalone codec for machines without lmcache
        ser = TurboQuantSerializer(bits=bits)  # type: ignore[assignment]
        des = TurboQuantDeserializer()  # type: ignore[assignment]
        buf_size = original_bytes * 2

    # Each serialize call needs its own write buffer; keep a shared read buffer
    # populated by one canonical serialize call for the deserialize benchmark.
    write_buf = torch.zeros(buf_size, dtype=torch.uint8)
    canonical_buf = torch.zeros(buf_size, dtype=torch.uint8)

    def _do_serialize(out_buf: torch.Tensor) -> int:
        if hasattr(ser, "serialize"):
            return ser.serialize(_MemObj(kv), _MemObj(out_buf))
        else:
            bs = ser.to_bytes(kv)
            n = len(bs)
            out_buf.ravel()[:n].copy_(torch.frombuffer(bs, dtype=torch.uint8))
            return n

    def _do_deserialize(src_buf: torch.Tensor, n: int) -> torch.Tensor:
        if hasattr(des, "deserialize"):
            dst = _MemObj(torch.zeros_like(kv))
            des.deserialize(_MemObj(src_buf[:n]), dst)
            return dst.tensor
        else:
            return des.from_bytes(bytes(src_buf[:n].numpy()))

    # --- warmup + capture canonical compressed blob ---
    for _ in range(WARMUP_ITERS):
        n = _do_serialize(write_buf)
    canonical_buf[:n].copy_(write_buf[:n])
    compressed_bytes = n

    # --- serialize benchmark ---
    ser_times = []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        _do_serialize(write_buf)
        ser_times.append((time.perf_counter() - t0) * 1e3)

    # --- deserialize benchmark ---
    des_times = []
    recovered = None
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        recovered = _do_deserialize(canonical_buf, n)
        des_times.append((time.perf_counter() - t0) * 1e3)

    sim = _cosine_sim(kv, recovered, head_dim)

    return Result(
        name=name,
        bits=bits,
        num_layers=num_layers,
        num_tokens=num_tokens,
        num_heads=num_heads,
        head_dim=head_dim,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        cosine_sim=sim,
        ser_ms=float(np.median(ser_times)),
        des_ms=float(np.median(des_times)),
        ser_times=ser_times,
        des_times=des_times,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    register()

    results: list[Result] = []

    print(f"\nRunning tqai serde benchmark (warmup={WARMUP_ITERS}, bench={BENCH_ITERS})\n")

    for cfg_name, nl, nt, nh, hd in CONFIGS:
        hidden_dim = nh * hd
        original_mb = (2 * nl * nt * hidden_dim * 2) / 1e6
        print(f"  {cfg_name}  [{2}×{nl}×{nt}×{hidden_dim}]  {original_mb:.1f} MB (fp16)")
        for bits in BITS_LIST:
            r = _bench_v1(cfg_name, nl, nt, nh, hd, bits)
            results.append(r)
            print(
                f"    bits={bits}  ratio={r.compression_ratio:.3f}  "
                f"cos={r.cosine_sim:.4f}  "
                f"ser={r.ser_ms:.1f}ms  des={r.des_ms:.1f}ms  "
                f"({r.ser_gbps:.2f} GB/s in / {r.des_gbps:.2f} GB/s out)"
            )

    # --- Markdown table ---
    header = [
        "Config", "Bits", "Tokens",
        "Ratio", "Cosine ↑", "Ser (ms)", "Des (ms)", "Ser GB/s", "Des GB/s",
    ]

    rows = []
    for r in results:
        rows.append([
            r.name,
            r.bits,
            r.num_tokens,
            f"{r.compression_ratio:.3f}",
            f"{r.cosine_sim:.4f}",
            f"{r.ser_ms:.1f}",
            f"{r.des_ms:.1f}",
            f"{r.ser_gbps:.2f}",
            f"{r.des_gbps:.2f}",
        ])

    try:
        from tabulate import tabulate
        md = tabulate(rows, headers=header, tablefmt="pipe")
    except ImportError:
        # Fallback: simple CSV
        md = ",".join(header) + "\n"
        for row in rows:
            md += ",".join(str(c) for c in row) + "\n"

    print("\n\n## tqai × LMCache Serde Benchmark\n")
    print("> Platform: CPU (torch bfloat16) — run on GPU for production numbers")
    print("> Note: bits=3 uses a Python-level bitstream packer (tqai._pack_bitstream).")
    print("> Vectorizing it with NumPy would bring 3-bit perf in line with 4-bit.\n")
    print(md)
    print()


if __name__ == "__main__":
    main()
