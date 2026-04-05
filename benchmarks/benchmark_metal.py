"""Microbenchmark: Metal kernel vs Python path for quantize/dequantize.

Usage:
    python benchmarks/benchmark_metal.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mlx.core as mx
import numpy as np

from tqai.backend import get_backend
from tqai.kernels import metal_available, metal_quantize, metal_dequantize
from tqai.quantizer import PolarQuantizer


def _bench(fn, warmup=10, iters=100):
    """Return median time in ms for fn()."""
    for _ in range(warmup):
        result = fn()
        if isinstance(result, tuple):
            mx.synchronize()
        else:
            mx.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = fn()
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times)


def run_benchmark():
    if not metal_available():
        print("Metal kernels not available on this system.")
        return

    ops = get_backend("mlx")

    shapes = [
        ("single-token decode", (1, 8, 1, 128)),
        ("16-token batch", (1, 8, 16, 128)),
        ("512-token prompt", (1, 8, 512, 128)),
    ]

    bits_list = [4]
    head_dim = 128

    print(f"{'Scenario':<25} {'Op':<12} {'Python (ms)':>12} {'Metal (ms)':>12} {'Speedup':>8}")
    print("-" * 72)

    for label, shape in shapes:
        for bits in bits_list:
            pq = PolarQuantizer(head_dim=head_dim, bits=bits, seed=42, ops=ops)
            x = mx.random.normal(shape, key=mx.random.key(0))
            mx.synchronize()

            # Python quantize
            pq._use_metal = False
            py_q_ms = _bench(lambda: pq.quantize(x))

            # Metal quantize
            m_q_ms = _bench(lambda: metal_quantize(x, pq._rotation, pq._centroids))

            speedup_q = py_q_ms / m_q_ms
            print(f"{label:<25} {'quantize':<12} {py_q_ms:>11.3f}  {m_q_ms:>11.3f}  {speedup_q:>7.1f}x")

            # Get indices/norms for dequantize bench
            indices, norms = metal_quantize(x, pq._rotation, pq._centroids)
            mx.synchronize()

            # Python dequantize
            py_d_ms = _bench(lambda: pq.dequantize(indices, norms))

            # Metal dequantize
            m_d_ms = _bench(lambda: metal_dequantize(indices, norms, pq._rotation, pq._centroids))

            speedup_d = py_d_ms / m_d_ms
            print(f"{'':<25} {'dequantize':<12} {py_d_ms:>11.3f}  {m_d_ms:>11.3f}  {speedup_d:>7.1f}x")

            # Roundtrip
            pq._use_metal = True
            m_rt_ms = _bench(lambda: pq.dequantize(*pq.quantize(x)))
            pq._use_metal = False
            py_rt_ms = _bench(lambda: pq.dequantize(*pq.quantize(x)))
            speedup_rt = py_rt_ms / m_rt_ms
            print(f"{'':<25} {'roundtrip':<12} {py_rt_ms:>11.3f}  {m_rt_ms:>11.3f}  {speedup_rt:>7.1f}x")
            print()


if __name__ == "__main__":
    run_benchmark()
