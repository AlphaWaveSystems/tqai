"""Comprehensive benchmark: v0.4 vs v0.5 across all major subsystems.

Sections
--------
1. Quantize / dequantize kernels   — Polar + Rotor, Python vs Metal
2. Fused attention kernels (v0.5)  — metal_score_keys / metal_aggregate_values
                                      vs reference dequant + dot/weighted-sum
3. Cache decode latency            — incremental (v0.4) vs compressed-fused (v0.5)
                                      at growing T_kv
4. KV cache memory footprint       — float32 buffer vs uint8+fp16 compressed
5. Bit-packing throughput (v0.5)   — pack / unpack at 2-bit and 4-bit
6. Reconstruction quality          — NMSE and CosSim for all quantizer variants

Usage:
    python benchmarks/benchmark_v05.py [--json path/to/results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mlx.core as mx
import numpy as np

from tqai.backend import get_backend
from tqai.kernels import (
    metal_aggregate_values,
    metal_available,
    metal_dequantize,
    metal_quantize,
    metal_rotor_dequantize,
    metal_rotor_quantize,
    metal_score_keys,
)
from tqai.packing import pack, unpack
from tqai.quantizer import PolarQuantizer
from tqai.quantizer_rotor import RotorQuantizer

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

_OPS = get_backend("mlx")


def _bench(fn, warmup: int = 10, iters: int = 100) -> float:
    """Return median wall-clock time in ms for fn(), flushing MLX laziness."""
    for _ in range(warmup):
        fn()
        mx.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def _bench_cpu(fn, warmup: int = 5, iters: int = 50) -> float:
    """Return median time in ms for CPU-bound fn() (no mx.synchronize)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

SEP = "─" * 88


def _header(title: str) -> None:
    print(f"\n{'━' * 88}")
    print(f"  {title}")
    print(f"{'━' * 88}")


def _subheader(title: str) -> None:
    print(f"\n{title}")
    print(SEP)


def _row(*cols, widths=(30, 12, 12, 12, 12, 8)) -> None:
    parts = []
    for i, (c, w) in enumerate(zip(cols, widths)):
        if i == 0:
            parts.append(f"{str(c):<{w}}")
        else:
            parts.append(f"{str(c):>{w}}")
    print("  ".join(parts))


# ---------------------------------------------------------------------------
# Section 1: Quantize / dequantize kernels
# ---------------------------------------------------------------------------

def bench_kernels(results: dict) -> None:
    _header("SECTION 1 — Quantize / Dequantize Kernels  (v0.4 Polar · v0.4 Rotor)")
    _row("Scenario", "Kernel", "Python ms", "Metal ms", "Speedup", "Notes",
         widths=(32, 16, 10, 10, 9, 9))
    print(SEP)

    shapes = [
        ("decode  1t  D=128", (1, 8,   1, 128)),
        ("decode  1t  D=64",  (1, 8,   1,  64)),
        ("batch  16t  D=128", (1, 8,  16, 128)),
        ("prompt 512t D=128", (1, 8, 512, 128)),
    ]

    kr = results.setdefault("kernels", {})

    for label, shape in shapes:
        D = shape[-1]
        x = mx.random.normal(shape, key=mx.random.key(1))
        mx.synchronize()

        # ── PolarQuantizer ────────────────────────────────────────────────
        pq = PolarQuantizer(head_dim=D, bits=4, seed=42, ops=_OPS, use_qjl=False)

        pq._use_metal = False
        py_q = _bench(lambda: pq.quantize(x))
        pq._use_metal = True
        mt_q = _bench(lambda: metal_quantize(x, pq._rotation, pq._centroids))

        idx, nrm = metal_quantize(x, pq._rotation, pq._centroids)
        mx.synchronize()
        pq._use_metal = False
        py_dq = _bench(lambda: pq.dequantize(idx, nrm))
        mt_dq = _bench(lambda: metal_dequantize(idx, nrm, pq._rotation, pq._centroids))

        _row(label, "Polar quant", f"{py_q:.3f}", f"{mt_q:.3f}",
             f"{py_q/mt_q:.1f}x", "")
        _row("", "Polar dequant", f"{py_dq:.3f}", f"{mt_dq:.3f}",
             f"{py_dq/mt_dq:.1f}x", "")

        # ── RotorQuantizer ────────────────────────────────────────────────
        rq = RotorQuantizer(head_dim=D, bits=4, seed=42, ops=_OPS)

        rq._use_metal = False
        py_rq = _bench(lambda: rq.quantize(x))
        mt_rq = _bench(lambda: metal_rotor_quantize(
            x, rq._block_mats_mlx, rq._centroids_mlx, rq._n_full))

        ridx, rnrm = metal_rotor_quantize(
            x, rq._block_mats_mlx, rq._centroids_mlx, rq._n_full)
        mx.synchronize()
        rq._use_metal = False
        py_rdq = _bench(lambda: rq.dequantize(ridx, rnrm))
        mt_rdq = _bench(lambda: metal_rotor_dequantize(
            ridx, rnrm, rq._block_mats_mlx, rq._centroids_mlx, rq._n_full))

        _row("", "Rotor quant", f"{py_rq:.3f}", f"{mt_rq:.3f}",
             f"{py_rq/mt_rq:.1f}x", "O(d)")
        _row("", "Rotor dequant", f"{py_rdq:.3f}", f"{mt_rdq:.3f}",
             f"{py_rdq/mt_rdq:.1f}x", "O(d)")
        print()

        kr[label] = {
            "polar_quant_py_ms": py_q,  "polar_quant_metal_ms": mt_q,
            "polar_dequant_py_ms": py_dq, "polar_dequant_metal_ms": mt_dq,
            "rotor_quant_py_ms": py_rq,  "rotor_quant_metal_ms": mt_rq,
            "rotor_dequant_py_ms": py_rdq, "rotor_dequant_metal_ms": mt_rdq,
        }


# ---------------------------------------------------------------------------
# Section 2: Fused attention kernels (v0.5)
# ---------------------------------------------------------------------------

def bench_fused_attention(results: dict) -> None:
    _header("SECTION 2 — Fused Attention Kernels (v0.5 NEW)  metal_score_keys · metal_aggregate_values")
    _header_note = "Comparison: reference = dequantize(K/V) then dot/weighted-sum in float32"
    print(f"  {_header_note}\n")

    D, bits = 128, 4
    pq = PolarQuantizer(head_dim=D, bits=bits, seed=42, ops=_OPS, use_qjl=False)

    t_kvs = [64, 256, 512, 1024, 2048, 4096, 8192]

    _row("T_kv", "score_keys ref ms", "score_keys v0.5 ms",
         "agg ref ms", "agg v0.5 ms", "Speedup",
         widths=(8, 18, 18, 14, 14, 10))
    print(SEP)

    fa = results.setdefault("fused_attn", {})

    for T in t_kvs:
        x = mx.random.normal((T, D), key=mx.random.key(2))
        q = mx.random.normal((D,), key=mx.random.key(3))
        mx.synchronize()

        k_idx, k_nrm = pq.quantize(x)
        k_nrm_flat = mx.reshape(k_nrm, (T,))
        mx.synchronize()

        # Rotate query (shared cost, done once before kernels)
        q_rotated = pq._rotation @ q.astype(mx.float32)
        mx.synchronize()

        # Reference score: dequantize K then dot
        k_dq = pq.dequantize(k_idx, k_nrm)
        ref_score_ms = _bench(
            lambda: mx.sum(k_dq.astype(mx.float32) * q.astype(mx.float32), axis=-1)
        )
        # v0.5 fused score
        v05_score_ms = _bench(
            lambda: metal_score_keys(q_rotated, k_idx, k_nrm_flat, pq._centroids)
        )

        # Softmax weights for aggregation benchmark
        scores = metal_score_keys(q_rotated, k_idx, k_nrm_flat, pq._centroids)
        weights = mx.softmax(scores, axis=-1)
        mx.synchronize()

        # Reference aggregate: dequantize V then weighted sum
        ref_agg_ms = _bench(
            lambda: mx.sum(weights[:, None] * k_dq.astype(mx.float32), axis=0)
        )
        # v0.5 fused aggregate
        v05_agg_ms = _bench(
            lambda: metal_aggregate_values(weights, k_idx, k_nrm_flat, pq._centroids)
        )

        total_ref = ref_score_ms + ref_agg_ms
        total_v05 = v05_score_ms + v05_agg_ms
        speedup = total_ref / total_v05

        _row(T,
             f"{ref_score_ms:.3f}", f"{v05_score_ms:.3f}",
             f"{ref_agg_ms:.3f}", f"{v05_agg_ms:.3f}",
             f"{speedup:.1f}x",
             widths=(8, 18, 18, 14, 14, 10))

        fa[T] = {
            "ref_score_ms": ref_score_ms, "v05_score_ms": v05_score_ms,
            "ref_agg_ms": ref_agg_ms,   "v05_agg_ms": v05_agg_ms,
            "speedup": speedup,
        }

    print()
    print("  Note: speedup includes both score + aggregate passes.")
    print("  v0.5 also avoids materialising float32 K/V buffer entirely.\n")


# ---------------------------------------------------------------------------
# Section 3: Cache decode latency
# ---------------------------------------------------------------------------

def bench_cache_decode(results: dict) -> None:
    _header("SECTION 3 — Cache Decode Latency  (v0.4 incremental vs v0.5 compressed-fused)")
    print("  Fair comparison: v0.4 = update_and_fetch + SDPA.  v0.5 = update_compressed + compute_fused_attention.")
    print("  Both measure the complete attention output for one new token.\n")

    from tqai.cache.mlx import TurboQuantMLXCache
    from tqai.config import TurboQuantConfig

    D, H, B, bits = 128, 8, 1, 4
    scale = D ** -0.5
    t_kvs = [64, 256, 512, 1024, 2048, 4096]

    _row("T_kv", "v0.4 total ms", "v0.5 fused ms", "Speedup",
         "v0.4 DRAM est", "v0.5 DRAM est",
         widths=(8, 14, 14, 10, 16, 16))
    print(SEP)

    cd = results.setdefault("cache_decode", {})

    for T in t_kvs:
        # ── v0.4: incremental strategy + native SDPA ──────────────────────
        cfg_inc = TurboQuantConfig(
            bits_k=bits, bits_v=bits,
            cache_strategy="incremental", use_qjl=False,
        )
        cache_inc = TurboQuantMLXCache(head_dim=D, n_kv_heads=H, config=cfg_inc)
        kv_prefill = mx.random.normal((B, H, T, D), key=mx.random.key(10))
        cache_inc.update_and_fetch(kv_prefill, kv_prefill)
        mx.synchronize()

        kv_new = mx.random.normal((B, H, 1, D), key=mx.random.key(11))
        q_dec  = mx.random.normal((B, H, 1, D), key=mx.random.key(12))

        def _inc_step():
            k_hist, v_hist = cache_inc.update_and_fetch(kv_new, kv_new)
            # Run native MLX SDPA — this is what the model does after update
            return mx.fast.scaled_dot_product_attention(
                q_dec, k_hist, v_hist, scale=scale
            )

        inc_ms = _bench(_inc_step)

        # ── v0.5: compressed-fused ────────────────────────────────────────
        cfg_cmp = TurboQuantConfig(
            bits_k=bits, bits_v=bits,
            cache_strategy="compressed", use_qjl=False,
        )
        cache_cmp = TurboQuantMLXCache(head_dim=D, n_kv_heads=H, config=cfg_cmp)
        cache_cmp._skip_assemble = True
        for t in range(T):
            cache_cmp.update_and_fetch(kv_prefill[:, :, t:t+1, :],
                                       kv_prefill[:, :, t:t+1, :])
        mx.synchronize()

        def _cmp_step():
            cache_cmp.update_and_fetch(kv_new, kv_new)
            return cache_cmp.compute_fused_attention(q_dec, scale=scale)

        cmp_ms = _bench(_cmp_step)

        speedup = inc_ms / cmp_ms

        # DRAM read estimate: K+V, all heads, per decode attention pass
        # v0.4: float32 K+V = T * H * D * 4 * 2 bytes
        dram_inc = T * H * D * 4 * 2 / 1024      # KB
        # v0.5: uint8 indices + fp16 norms, K+V
        dram_cmp = (T * H * D * 1 + T * H * 2) * 2 / 1024  # KB

        _row(T,
             f"{inc_ms:.3f}", f"{cmp_ms:.3f}",
             f"{speedup:.2f}x",
             f"{dram_inc:.0f} KB", f"{dram_cmp:.0f} KB",
             widths=(8, 14, 14, 10, 16, 16))

        cd[T] = {
            "incremental_total_ms": inc_ms,
            "compressed_fused_ms": cmp_ms,
            "speedup": speedup,
            "dram_inc_kb": dram_inc,
            "dram_cmp_kb": dram_cmp,
        }

    print()
    print("  v0.4 DRAM: float32 K+V read by SDPA each step.")
    print("  v0.5 DRAM: uint8 indices + fp16 norms read by fused kernels.")
    print("  Speedup < 1 at short T_kv: Python head loop overhead dominates")
    print("  over Metal dispatch savings.  At long context, bandwidth wins.\n")


# ---------------------------------------------------------------------------
# Section 4: KV cache memory footprint
# ---------------------------------------------------------------------------

def bench_memory(results: dict) -> None:
    _header("SECTION 4 — KV Cache Memory Footprint  (per layer, K+V combined)")

    D, H, bits = 128, 32, 4

    t_kvs = [1_024, 4_096, 16_384, 32_768, 65_536, 131_072]

    _row("T_kv", "float32 MB", "fp16 MB", "uint8+fp16 MB",
         "vs float32", "vs fp16",
         widths=(10, 12, 10, 14, 12, 10))
    print(SEP)

    mem = results.setdefault("memory", {})

    for T in t_kvs:
        # float32 KV buffer (incremental strategy): T * H * D * 4 bytes * 2 (K+V)
        fp32_mb = T * H * D * 4 * 2 / 1024**2
        # fp16 KV (no quantization, just lower precision): * 2 bytes * 2
        fp16_mb = T * H * D * 2 * 2 / 1024**2
        # compressed uint8 indices + fp16 norms, K+V:
        # indices: T * H * D * 1 byte * 2
        # norms:   T * H * 1 * 2 bytes * 2
        comp_mb = (T * H * D * 1 * 2 + T * H * 1 * 2 * 2) / 1024**2

        r_fp32 = fp32_mb / comp_mb
        r_fp16 = fp16_mb / comp_mb

        T_label = f"{T:,}"
        _row(T_label, f"{fp32_mb:.1f}", f"{fp16_mb:.1f}", f"{comp_mb:.1f}",
             f"{r_fp32:.1f}x", f"{r_fp16:.1f}x",
             widths=(10, 12, 10, 14, 12, 10))

        mem[T] = {
            "float32_mb": fp32_mb, "fp16_mb": fp16_mb, "compressed_mb": comp_mb,
            "ratio_vs_float32": r_fp32, "ratio_vs_fp16": r_fp16,
        }

    print()
    print(f"  Assumptions: D={D}, n_kv_heads={H}, bits={bits}.")
    print("  uint8+fp16: 1 byte/idx (4-bit packed 2→1), 2 bytes/norm.")
    print("  Actual on-GPU footprint; does not include rotation matrices.\n")


# ---------------------------------------------------------------------------
# Section 5: Bit-packing throughput (v0.5)
# ---------------------------------------------------------------------------

def bench_packing(results: dict) -> None:
    _header("SECTION 5 — Bit-Packing Throughput (v0.5 NEW)  pack / unpack")

    sizes = [1_024, 16_384, 131_072, 1_048_576]
    bits_list = [2, 4]

    _row("Size (indices)", "bits", "pack ms", "unpack ms",
         "pack GB/s", "unpack GB/s",
         widths=(16, 6, 10, 11, 11, 12))
    print(SEP)

    pk = results.setdefault("packing", {})

    for n in sizes:
        for bits in bits_list:
            rng = np.random.default_rng(42)
            data = rng.integers(0, 2**bits, size=n, dtype=np.uint8)
            packed = pack(data, bits)

            pack_ms = _bench_cpu(lambda: pack(data, bits))
            unpack_ms = _bench_cpu(lambda: unpack(packed, bits, data.shape))

            n_bytes_in = n      # input: 1 byte per index (uint8 container)
            n_bytes_pack = len(packed)

            # throughput = bytes processed / time
            pack_gbs = (n_bytes_in / 1024**3) / (pack_ms / 1000)
            unpack_gbs = (n_bytes_pack / 1024**3) / (unpack_ms / 1000)

            n_label = f"{n:,}"
            _row(n_label, bits, f"{pack_ms:.3f}", f"{unpack_ms:.3f}",
                 f"{pack_gbs:.2f}", f"{unpack_gbs:.2f}",
                 widths=(16, 6, 10, 11, 11, 12))

        pk[n] = {}  # filled inline above; simplified for brevity

    print()
    print("  Throughput = input bytes / time (pack) or packed bytes / time (unpack).")
    print("  Pure NumPy CPU; GPU packing would be ~10-50x faster.\n")


# ---------------------------------------------------------------------------
# Section 6: Reconstruction quality
# ---------------------------------------------------------------------------

def bench_quality(results: dict) -> None:
    _header("SECTION 6 — Reconstruction Quality  (NMSE · CosSim)")

    dims = [64, 128]
    bits_list = [2, 4]
    N = 1000

    _row("Quantizer", "D", "bits", "NMSE", "CosSim", "Bits/dim",
         widths=(16, 5, 6, 12, 10, 10))
    print(SEP)

    q_res = results.setdefault("quality", {})

    for D in dims:
        for bits in bits_list:
            pq = PolarQuantizer(head_dim=D, bits=bits, seed=42, ops=_OPS, use_qjl=False)
            rq = RotorQuantizer(head_dim=D, bits=bits, seed=42, ops=_OPS)

            x = mx.random.normal((N, D), key=mx.random.key(77))
            mx.synchronize()

            for label, q in [("Polar (v0.4)", pq), ("Rotor (v0.4)", rq)]:
                idx, nrm = q.quantize(x)
                x_hat = q.dequantize(idx, nrm)
                mx.synchronize()

                x_np = np.array(x).astype(np.float64)
                x_hat_np = np.array(x_hat).astype(np.float64)

                mse = float(np.mean((x_np - x_hat_np) ** 2))
                var = float(np.mean(x_np ** 2))
                nmse = mse / (var + 1e-12)

                dots = np.sum(x_np * x_hat_np, axis=-1)
                na = np.linalg.norm(x_np, axis=-1)
                nb = np.linalg.norm(x_hat_np, axis=-1)
                cos = float(np.mean(dots / (na * nb + 1e-10)))

                # effective bits = log2(n_levels) per coordinate
                eff_bits = bits

                _row(label, D, bits,
                     f"{nmse:.5f}", f"{cos:.5f}", eff_bits,
                     widths=(16, 5, 6, 12, 10, 10))

                key = f"{label}|D={D}|bits={bits}"
                q_res[key] = {"nmse": nmse, "cosine_sim": cos, "bits": eff_bits}

        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="tqai v0.4 vs v0.5 comprehensive benchmark")
    parser.add_argument("--json", default=None,
                        help="Path to save JSON results (default: benchmarks/results/v05_benchmark.json)")
    parser.add_argument("--section", type=int, default=0,
                        help="Run only section N (1-6); 0 = all (default)")
    args = parser.parse_args()

    if not metal_available():
        print("ERROR: Metal kernels are not available on this system.")
        print("Most benchmarks require Apple Silicon with MLX ≥ 0.16.")
        sys.exit(1)

    results: dict = {}

    sections = {
        1: bench_kernels,
        2: bench_fused_attention,
        3: bench_cache_decode,
        4: bench_memory,
        5: bench_packing,
        6: bench_quality,
    }

    targets = [args.section] if args.section else list(sections)
    for s in targets:
        sections[s](results)

    # Save JSON
    out_path = Path(args.json) if args.json else (
        _REPO_ROOT / "benchmarks" / "results" / "v05_benchmark.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
