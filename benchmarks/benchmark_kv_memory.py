"""KV cache memory benchmark — proves the v0.3.1 K4/V2 win at scale.

Measures peak MLX active memory and prefill throughput on a real model
(Qwen 2.5-7B Q8 by default) at increasing context lengths, comparing
baseline mlx-lm with `tqai.patch(model, bits_k=4, bits_v=2)`.

The K4/V2 quantization compresses the KV cache by ~5x. At small contexts
this is invisible because the model weights (~8.7 GB for Qwen 7B Q8)
dominate the memory footprint. At long contexts (128K+) the KV cache
becomes the dominant memory consumer and quantization unlocks contexts
that would otherwise OOM.

Per-token KV cache for Qwen 2.5-7B (28 layers, 4 KV heads, head_dim 128):
    uncompressed: 56 KB / token
    K4/V2:        11 KB / token  (5.12x reduction)

Context length comparison:
    Context | Uncompressed KV | K4/V2 KV | Savings
       32K  |    1.88 GB      |  0.37 GB |  1.51 GB
      128K  |    7.52 GB      |  1.47 GB |  6.05 GB
      256K  |   15.03 GB      |  2.94 GB | 12.10 GB
      512K  |   30.06 GB      |  5.87 GB | 24.19 GB

Architecture
============
Each (context, config) pair runs in its own Python subprocess so that
MLX memory measurements are isolated. Within each subprocess we use
mx.get_active_memory() / mx.get_peak_memory() / mx.reset_peak_memory()
to measure exactly the MLX tensor memory (not Python heap or model
weights), and we reset the peak watermark between the model-load and
prefill phases so we can attribute KV cache growth specifically.

Usage:
    python benchmarks/benchmark_kv_memory.py
    python benchmarks/benchmark_kv_memory.py --contexts 32768,131072,262144
    python benchmarks/benchmark_kv_memory.py --model mlx-community/Qwen2.5-7B-Instruct-8bit
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent


# A long passage repeated to hit target token counts. Generic Wikipedia-style
# content compresses well and is representative of RAG / long-document use cases.
PASSAGE = """The history of computing is a tale of relentless innovation, beginning
with mechanical calculators in the 17th century and progressing through
electromechanical relays, vacuum tubes, transistors, and integrated circuits.
Each generation brought orders of magnitude improvements in speed, size,
power consumption, and cost. The transistor, invented at Bell Labs in 1947,
replaced fragile vacuum tubes and enabled the miniaturization that defined
the modern era. Jack Kilby's integrated circuit in 1958 packed multiple
transistors onto a single piece of silicon, and Moore's Law, formulated by
Intel co-founder Gordon Moore in 1965, predicted the doubling of transistor
counts every two years. This prediction held remarkably true for over five
decades, driving the personal computer revolution, the internet, mobile
computing, and now the AI era. Today's processors contain tens of billions
of transistors on a single chip, performing trillions of operations per
second while consuming only a few watts of power. The journey from the
ENIAC, which weighed 30 tons and consumed 150 kilowatts, to the smartphone
in your pocket represents perhaps the most dramatic technological progression
in human history. Yet the most profound consequences may still lie ahead,
as machine learning models trained on this hardware begin to exhibit
capabilities that were once the exclusive province of biological intelligence.
"""


@dataclass
class KVMemoryResult:
    config: str  # "baseline" or "tqai_k4v2"
    target_context: int
    actual_context: int
    bits_k: int
    bits_v: int
    model_load_gb: float       # MLX active memory after model load
    prefill_peak_gb: float     # MLX peak memory during prefill (KV cache + intermediates)
    kv_growth_gb: float        # prefill_peak_gb - model_load_gb
    decode_peak_gb: float      # MLX peak memory during decode (post-reset)
    prefill_time_s: float
    decode_time_s: float
    decode_tokens: int
    decode_tokens_per_sec: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Subprocess worker — runs ONE configuration in isolation
# ---------------------------------------------------------------------------

WORKER_SOURCE = '''
"""Subprocess worker. Runs a single (model, context, config) measurement."""
import gc
import json
import sys
import time

# Inputs come via stdin as a JSON dict
spec = json.loads(sys.stdin.read())
model_id = spec["model_id"]
target_context = spec["target_context"]
use_tqai = spec["use_tqai"]
bits_k = spec["bits_k"]
bits_v = spec["bits_v"]
max_new_tokens = spec["max_new_tokens"]
passage = spec["passage"]

import mlx.core as mx
import mlx_lm

import tqai


def _build_prompt(tokenizer, target_tokens):
    chunk_tokens = len(tokenizer.encode(passage))
    repeats = max(1, (target_tokens - 50) // chunk_tokens)
    body = passage * repeats
    return body + "\\n\\nSummarize the passage above in one sentence:"


result = {
    "config": "tqai_k4v2" if use_tqai else "baseline",
    "target_context": target_context,
    "bits_k": bits_k if use_tqai else 0,
    "bits_v": bits_v if use_tqai else 0,
    "actual_context": 0,
    "model_load_gb": 0.0,
    "prefill_peak_gb": 0.0,
    "kv_growth_gb": 0.0,
    "decode_peak_gb": 0.0,
    "prefill_time_s": 0.0,
    "decode_time_s": 0.0,
    "decode_tokens": 0,
    "decode_tokens_per_sec": 0.0,
    "error": None,
}

try:
    mx.reset_peak_memory()

    # Phase 1: load model
    model, tokenizer = mlx_lm.load(model_id)
    mx.synchronize()
    result["model_load_gb"] = round(mx.get_active_memory() / 1e9, 3)

    if use_tqai:
        tqai.patch(model, bits_k=bits_k, bits_v=bits_v, backend="mlx")

    # Reset peak so the next measurement attributes only KV cache + prefill activations
    mx.reset_peak_memory()

    # Phase 2: prefill (1 new token)
    prompt = _build_prompt(tokenizer, target_context)
    result["actual_context"] = len(tokenizer.encode(prompt))

    t0 = time.perf_counter()
    _ = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=1, verbose=False)
    mx.synchronize()
    result["prefill_time_s"] = round(time.perf_counter() - t0, 2)
    result["prefill_peak_gb"] = round(mx.get_peak_memory() / 1e9, 3)
    result["kv_growth_gb"] = round(result["prefill_peak_gb"] - result["model_load_gb"], 3)

    # Phase 3: decode N more tokens (separate measurement)
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    _ = mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=max_new_tokens, verbose=False,
    )
    mx.synchronize()
    decode_total = time.perf_counter() - t0
    # The decode call also redoes prefill, so subtract prefill_time and clamp
    decode_only = max(decode_total - result["prefill_time_s"], 0.001)
    result["decode_time_s"] = round(decode_only, 3)
    result["decode_tokens"] = max_new_tokens
    result["decode_tokens_per_sec"] = round(max_new_tokens / decode_only, 2)
    result["decode_peak_gb"] = round(mx.get_peak_memory() / 1e9, 3)

except Exception as e:
    result["error"] = f"{type(e).__name__}: {e}"[:300]

# Capture peak even on error
try:
    if result["prefill_peak_gb"] == 0.0:
        result["prefill_peak_gb"] = round(mx.get_peak_memory() / 1e9, 3)
except Exception:
    pass

print(json.dumps(result))
'''


def _run_one(
    model_id: str,
    target_context: int,
    use_tqai: bool,
    bits_k: int,
    bits_v: int,
    max_new_tokens: int,
) -> KVMemoryResult:
    """Run a single (context, config) measurement in an isolated subprocess."""
    label = "tqai_k4v2" if use_tqai else "baseline"
    print(f"  [{label}] launching subprocess...")

    spec = {
        "model_id": model_id,
        "target_context": target_context,
        "use_tqai": use_tqai,
        "bits_k": bits_k,
        "bits_v": bits_v,
        "max_new_tokens": max_new_tokens,
        "passage": PASSAGE,
    }

    env_args = [
        sys.executable,
        "-c",
        WORKER_SOURCE,
    ]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            env_args,
            input=json.dumps(spec),
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            check=False,
            timeout=1800,  # 30 min per config
        )
    except subprocess.TimeoutExpired:
        return KVMemoryResult(
            config=label, target_context=target_context, actual_context=0,
            bits_k=bits_k if use_tqai else 0, bits_v=bits_v if use_tqai else 0,
            model_load_gb=0, prefill_peak_gb=0, kv_growth_gb=0, decode_peak_gb=0,
            prefill_time_s=0, decode_time_s=0, decode_tokens=0, decode_tokens_per_sec=0,
            error="TIMEOUT after 30 minutes",
        )
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] subprocess done in {elapsed:.0f}s (exit {proc.returncode})")

    # The worker prints exactly one JSON line; find it
    last_json = None
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                last_json = json.loads(line)
            except json.JSONDecodeError:
                continue

    if last_json is None:
        # Worker died before printing JSON — record stderr
        err_tail = proc.stderr.strip().splitlines()[-3:] if proc.stderr else ["(no stderr)"]
        return KVMemoryResult(
            config=label, target_context=target_context, actual_context=0,
            bits_k=bits_k if use_tqai else 0, bits_v=bits_v if use_tqai else 0,
            model_load_gb=0, prefill_peak_gb=0, kv_growth_gb=0, decode_peak_gb=0,
            prefill_time_s=0, decode_time_s=0, decode_tokens=0, decode_tokens_per_sec=0,
            error=f"worker died (exit {proc.returncode}): " + " | ".join(err_tail),
        )

    return KVMemoryResult(**last_json)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_benchmark(
    model_id: str,
    contexts: list[int],
    bits_k: int = 4,
    bits_v: int = 2,
    max_new_tokens: int = 5,
    output_path: str = "benchmarks/results/kv_memory.json",
) -> list[KVMemoryResult]:
    print(f"\n{'=' * 80}")
    print("tqai KV Cache Memory Benchmark")
    print(f"{'=' * 80}")
    print(f"Model:        {model_id}")
    print(f"Contexts:     {contexts}")
    print(f"Compression:  K{bits_k}/V{bits_v}")
    print("Memory metric: mx.get_peak_memory() (MLX-native, isolated per subprocess)")
    print(f"{'=' * 80}\n")

    results: list[KVMemoryResult] = []

    for ctx in contexts:
        print(f"\n--- Context: {ctx} tokens ---")
        for use_tqai in (False, True):
            result = _run_one(
                model_id, ctx, use_tqai, bits_k, bits_v, max_new_tokens,
            )
            results.append(result)
            if result.error:
                print(f"    ERROR: {result.error}")
            else:
                print(
                    f"    OK    actual_ctx={result.actual_context}  "
                    f"model={result.model_load_gb}G  prefill_peak={result.prefill_peak_gb}G  "
                    f"kv={result.kv_growth_gb}G  prefill={result.prefill_time_s}s  "
                    f"decode={result.decode_tokens_per_sec}tok/s"
                )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model_id,
        "bits_k": bits_k,
        "bits_v": bits_v,
        "contexts": contexts,
        "results": [asdict(r) for r in results],
    }, indent=2))
    print(f"\nResults saved: {out_path}")

    _print_summary(results)
    return results


def _print_summary(results: list[KVMemoryResult]):
    print(f"\n{'=' * 80}")
    print("KV CACHE MEMORY BENCHMARK SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"{'Context':>9} {'Config':<12} {'Model':>9} {'Prefill peak':>13} "
        f"{'KV growth':>11} {'Status':<10}"
    )
    print(
        f"{'-' * 9} {'-' * 12} {'-' * 9} {'-' * 13} "
        f"{'-' * 11} {'-' * 10}"
    )

    contexts = sorted(set(r.target_context for r in results))
    for ctx in contexts:
        ctx_results = [r for r in results if r.target_context == ctx]
        for r in ctx_results:
            if r.error:
                print(
                    f"{r.target_context:>9} {r.config:<12} "
                    f"{'---':>9} {'---':>13} {'---':>11} OOM/ERROR"
                )
            else:
                print(
                    f"{r.target_context:>9} {r.config:<12} "
                    f"{r.model_load_gb:>8.2f}G {r.prefill_peak_gb:>12.2f}G "
                    f"{r.kv_growth_gb:>10.2f}G OK"
                )

        baseline = next((r for r in ctx_results if r.config == "baseline" and r.error is None), None)
        tqai = next((r for r in ctx_results if r.config == "tqai_k4v2" and r.error is None), None)
        if baseline and tqai:
            saved = baseline.kv_growth_gb - tqai.kv_growth_gb
            ratio = (baseline.kv_growth_gb / tqai.kv_growth_gb) if tqai.kv_growth_gb > 0 else 0
            print(
                f"{'':>9} {'Δ savings':<12} "
                f"{'':>9} {'':>13} {saved:>+10.2f}G  ({ratio:.1f}x reduction)"
            )
        print()


def main():
    p = argparse.ArgumentParser(description="tqai KV cache memory benchmark")
    p.add_argument(
        "--model", "-m",
        default="mlx-community/Qwen2.5-7B-Instruct-8bit",
        help="MLX model ID",
    )
    p.add_argument(
        "--contexts",
        default="32768,65536,131072",
        help="Comma-separated context token counts",
    )
    p.add_argument("--bits-k", type=int, default=4)
    p.add_argument("--bits-v", type=int, default=2)
    p.add_argument(
        "--max-new-tokens", type=int, default=5,
        help="Just enough to validate decode works (default 5)",
    )
    p.add_argument("--output", default="benchmarks/results/kv_memory.json")
    args = p.parse_args()

    contexts = [int(c) for c in args.contexts.split(",")]
    run_benchmark(
        model_id=args.model,
        contexts=contexts,
        bits_k=args.bits_k,
        bits_v=args.bits_v,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
