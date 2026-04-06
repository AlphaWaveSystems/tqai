"""KV cache memory benchmark — proves the v0.3.1 K4/V2 win at scale.

Measures peak resident set size and prefill throughput on a real model
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

Usage:
    python benchmarks/benchmark_kv_memory.py
    python benchmarks/benchmark_kv_memory.py --contexts 32768,131072,262144
    python benchmarks/benchmark_kv_memory.py --model mlx-community/Qwen2.5-7B-Instruct-8bit
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _rss_gb() -> float:
    """Process resident set size in GB.

    On macOS, ru_maxrss is reported in bytes (not KB as on Linux).
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


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


def _build_prompt(tokenizer, target_tokens: int) -> str:
    """Build a prompt of approximately ``target_tokens`` tokens."""
    chunk_tokens = len(tokenizer.encode(PASSAGE))
    repeats = max(1, (target_tokens - 50) // chunk_tokens)
    body = PASSAGE * repeats
    return body + "\n\nSummarize the passage above in one sentence:"


@dataclass
class KVMemoryResult:
    config: str  # "baseline" or "tqai_k4v2"
    target_context: int
    actual_context: int
    bits_k: int
    bits_v: int
    rss_before_load_gb: float
    rss_after_load_gb: float
    rss_after_prefill_gb: float
    rss_peak_gb: float
    prefill_time_s: float
    generation_time_s: float
    tokens_per_sec: float
    error: str | None = None


def _run_one(
    model_id: str,
    target_context: int,
    use_tqai: bool,
    bits_k: int,
    bits_v: int,
    max_new_tokens: int,
) -> KVMemoryResult:
    """Run a single configuration and return memory + timing results."""
    import mlx_lm

    import tqai

    rss_before = _rss_gb()
    print(f"    [load] RSS before: {rss_before:.2f} GB")

    try:
        model, tokenizer = mlx_lm.load(model_id)
    except Exception as e:
        return KVMemoryResult(
            config="tqai_k4v2" if use_tqai else "baseline",
            target_context=target_context,
            actual_context=0,
            bits_k=bits_k if use_tqai else 0,
            bits_v=bits_v if use_tqai else 0,
            rss_before_load_gb=rss_before,
            rss_after_load_gb=0, rss_after_prefill_gb=0, rss_peak_gb=0,
            prefill_time_s=0, generation_time_s=0, tokens_per_sec=0,
            error=f"load failed: {e}",
        )

    rss_after_load = _rss_gb()
    print(f"    [load] RSS after: {rss_after_load:.2f} GB (model: {rss_after_load - rss_before:.2f} GB)")

    if use_tqai:
        tqai.patch(model, bits_k=bits_k, bits_v=bits_v, backend="mlx")
        print(f"    [tqai] patched K{bits_k}/V{bits_v}")

    try:
        prompt = _build_prompt(tokenizer, target_context)
        actual_tokens = len(tokenizer.encode(prompt))
        print(f"    [prompt] {actual_tokens} tokens (target {target_context})")

        # Prefill: time + memory after prefill
        t0 = time.perf_counter()
        _ = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=1, verbose=False)
        prefill_time = time.perf_counter() - t0
        rss_after_prefill = _rss_gb()
        print(f"    [prefill] {prefill_time:.1f}s  RSS: {rss_after_prefill:.2f} GB (+{rss_after_prefill - rss_after_load:.2f} GB for KV)")

        # Generate a small number of new tokens to measure decode + peak memory
        t0 = time.perf_counter()
        _ = mlx_lm.generate(
            model, tokenizer, prompt=prompt, max_tokens=max_new_tokens, verbose=False,
        )
        gen_time = time.perf_counter() - t0
        rss_peak = _rss_gb()
        # Subtract prefill from total to estimate decode-only time
        decode_time = max(gen_time - prefill_time, 0.001)
        tps = max_new_tokens / decode_time if decode_time > 0 else 0

    except Exception as e:
        if use_tqai:
            tqai.unpatch(model)
        del model, tokenizer
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass
        return KVMemoryResult(
            config="tqai_k4v2" if use_tqai else "baseline",
            target_context=target_context,
            actual_context=0,
            bits_k=bits_k if use_tqai else 0,
            bits_v=bits_v if use_tqai else 0,
            rss_before_load_gb=round(rss_before, 2),
            rss_after_load_gb=round(rss_after_load, 2),
            rss_after_prefill_gb=0, rss_peak_gb=round(_rss_gb(), 2),
            prefill_time_s=0, generation_time_s=0, tokens_per_sec=0,
            error=f"{type(e).__name__}: {e}"[:200],
        )

    if use_tqai:
        tqai.unpatch(model)

    del model, tokenizer
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass

    return KVMemoryResult(
        config="tqai_k4v2" if use_tqai else "baseline",
        target_context=target_context,
        actual_context=actual_tokens,
        bits_k=bits_k if use_tqai else 0,
        bits_v=bits_v if use_tqai else 0,
        rss_before_load_gb=round(rss_before, 2),
        rss_after_load_gb=round(rss_after_load, 2),
        rss_after_prefill_gb=round(rss_after_prefill, 2),
        rss_peak_gb=round(rss_peak, 2),
        prefill_time_s=round(prefill_time, 1),
        generation_time_s=round(gen_time, 1),
        tokens_per_sec=round(tps, 1),
    )


def run_benchmark(
    model_id: str,
    contexts: list[int],
    bits_k: int = 4,
    bits_v: int = 2,
    max_new_tokens: int = 5,
    output_path: str = "benchmarks/results/kv_memory.json",
) -> list[KVMemoryResult]:
    print(f"\n{'=' * 80}")
    print("tqai KV Cache Memory Benchmark — proves the v0.3.1 K4/V2 win at scale")
    print(f"{'=' * 80}")
    print(f"Model:        {model_id}")
    print(f"Contexts:     {contexts}")
    print(f"Compression:  K{bits_k}/V{bits_v}")
    print(f"Max new tok:  {max_new_tokens} (we measure prefill memory, not generation)")
    print(f"{'=' * 80}\n")

    results: list[KVMemoryResult] = []

    for ctx in contexts:
        print(f"\n--- Context: {ctx} tokens ---")
        for use_tqai in (False, True):
            label = "tqai_k4v2" if use_tqai else "baseline"
            print(f"  [{label}]")
            result = _run_one(
                model_id, ctx, use_tqai, bits_k, bits_v, max_new_tokens,
            )
            results.append(result)
            if result.error:
                print(f"    ERROR: {result.error}")
            else:
                print(
                    f"    DONE  prefill={result.prefill_time_s}s "
                    f"peak={result.rss_peak_gb}GB  decode={result.tokens_per_sec}tok/s"
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
        f"{'Context':>9} {'Config':<12} {'After load':>11} {'After prefill':>14} "
        f"{'Peak':>8} {'KV size':>9} {'Status':<10}"
    )
    print(
        f"{'-' * 9} {'-' * 12} {'-' * 11} {'-' * 14} "
        f"{'-' * 8} {'-' * 9} {'-' * 10}"
    )

    contexts = sorted(set(r.target_context for r in results))
    for ctx in contexts:
        ctx_results = [r for r in results if r.target_context == ctx]
        for r in ctx_results:
            if r.error:
                print(
                    f"{r.target_context:>9} {r.config:<12} "
                    f"{r.rss_after_load_gb:>10.2f}G {'---':>14} "
                    f"{r.rss_peak_gb:>7.2f}G {'---':>9} OOM/ERROR"
                )
            else:
                kv_size = r.rss_after_prefill_gb - r.rss_after_load_gb
                print(
                    f"{r.target_context:>9} {r.config:<12} "
                    f"{r.rss_after_load_gb:>10.2f}G {r.rss_after_prefill_gb:>13.2f}G "
                    f"{r.rss_peak_gb:>7.2f}G {kv_size:>8.2f}G OK"
                )

        # Compute the savings if both succeeded
        baseline = next((r for r in ctx_results if r.config == "baseline" and r.error is None), None)
        tqai = next((r for r in ctx_results if r.config == "tqai_k4v2" and r.error is None), None)
        if baseline and tqai:
            kv_baseline = baseline.rss_after_prefill_gb - baseline.rss_after_load_gb
            kv_tqai = tqai.rss_after_prefill_gb - tqai.rss_after_load_gb
            saved = kv_baseline - kv_tqai
            ratio = (kv_baseline / kv_tqai) if kv_tqai > 0 else 0
            print(
                f"{'':>9} {'Δ savings':<12} "
                f"{'':>11} {'':>14} "
                f"{'':>8} {saved:>+7.2f}G ({ratio:.1f}x reduction)"
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
