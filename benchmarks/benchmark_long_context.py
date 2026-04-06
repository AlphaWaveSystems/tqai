"""Long-context benchmark for tqai chunked attention.

Measures memory and throughput on Apple Silicon (MLX) at increasing
context lengths, comparing baseline vs chunked attention.

The chunked attention path activates when KV sequence length exceeds
``chunk_size`` (default 4096), so the benchmark uses contexts >= 8K
to actually trigger chunking.

Usage:
    python benchmarks/benchmark_long_context.py
    python benchmarks/benchmark_long_context.py --model mlx-community/Qwen2.5-7B-Instruct-8bit
    python benchmarks/benchmark_long_context.py --contexts 4096,8192,16384
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _rss_gb() -> float:
    """Process resident set size in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


# A passage that compresses well — we repeat it to hit target token counts
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
    full = (
        body
        + "\n\nBased on the passages above, write a brief one-sentence summary:"
    )
    return full


@dataclass
class LongContextResult:
    config: str
    context_tokens: int
    chunk_size: int
    prefill_time_s: float
    generation_time_s: float
    total_time_s: float
    tokens_generated: int
    tokens_per_sec: float
    peak_rss_gb: float
    output_text: str
    output_match: bool


def _run_mlx_config(
    model_id: str,
    context_tokens: int,
    chunk_size: int,
    use_chunked: bool,
    max_new_tokens: int = 50,
) -> tuple[LongContextResult, str]:
    """Run a single MLX generation config and return results."""
    import mlx_lm

    import tqai

    print(f"    Loading {model_id}...")
    model, tokenizer = mlx_lm.load(model_id)

    if use_chunked:
        # kv_compression=False so we isolate chunked attention's effect
        tqai.patch(
            model,
            backend="mlx",
            chunk_attention=True,
            attention_chunk_size=chunk_size,
            kv_compression=False,
        )
        print(f"    Patched chunked attention (chunk_size={chunk_size})")

    prompt = _build_prompt(tokenizer, context_tokens)
    actual_tokens = len(tokenizer.encode(prompt))
    print(f"    Prompt: {actual_tokens} tokens")

    # Prefill measurement: 1-token generation
    t0 = time.perf_counter()
    _ = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=1, verbose=False)
    prefill_time = time.perf_counter() - t0

    # Full generation
    t0 = time.perf_counter()
    response = mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=max_new_tokens, verbose=False,
    )
    total_time = time.perf_counter() - t0
    rss_peak = _rss_gb()

    gen_time = max(total_time - prefill_time, 0.001)
    tps = max_new_tokens / gen_time if gen_time > 0 else 0

    output = response[len(prompt):] if response.startswith(prompt) else response
    output = output.strip()

    if use_chunked:
        tqai.unpatch(model)

    del model, tokenizer
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass

    result = LongContextResult(
        config="chunked" if use_chunked else "baseline",
        context_tokens=actual_tokens,
        chunk_size=chunk_size,
        prefill_time_s=round(prefill_time, 2),
        generation_time_s=round(gen_time, 2),
        total_time_s=round(total_time, 2),
        tokens_generated=max_new_tokens,
        tokens_per_sec=round(tps, 2),
        peak_rss_gb=round(rss_peak, 2),
        output_text=output[:200],
        output_match=False,
    )
    return result, output


def run_benchmark(
    model_id: str,
    contexts: list[int],
    chunk_size: int = 4096,
    max_new_tokens: int = 50,
    output_path: str = "benchmarks/results/long_context.json",
) -> list[LongContextResult]:
    print(f"\n{'='*80}")
    print("tqai Long-Context Benchmark")
    print(f"{'='*80}")
    print(f"Model:       {model_id}")
    print(f"Contexts:    {contexts}")
    print(f"Chunk size:  {chunk_size}")
    print(f"Max gen:     {max_new_tokens} tokens")
    print(f"{'='*80}\n")

    all_results: list[LongContextResult] = []

    for ctx in contexts:
        print(f"\n--- Context: {ctx} tokens ---")
        baseline_output: str | None = None

        for use_chunked in (False, True):
            label = "chunked" if use_chunked else "baseline"
            print(f"  [{label}]")
            try:
                result, output = _run_mlx_config(
                    model_id, ctx, chunk_size, use_chunked, max_new_tokens,
                )

                if not use_chunked:
                    baseline_output = output
                    result.output_match = True
                else:
                    result.output_match = (output == baseline_output)

                all_results.append(result)

                print(
                    f"    prefill={result.prefill_time_s:.1f}s "
                    f"gen={result.generation_time_s:.1f}s "
                    f"({result.tokens_per_sec:.1f} tok/s) "
                    f"peak={result.peak_rss_gb:.1f}GB "
                    f"match={result.output_match}"
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                all_results.append(LongContextResult(
                    config=label, context_tokens=ctx, chunk_size=chunk_size,
                    prefill_time_s=0, generation_time_s=0, total_time_s=0,
                    tokens_generated=0, tokens_per_sec=0, peak_rss_gb=0,
                    output_text=f"ERROR: {e}", output_match=False,
                ))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model_id,
        "contexts": contexts,
        "chunk_size": chunk_size,
        "max_new_tokens": max_new_tokens,
        "results": [asdict(r) for r in all_results],
    }
    out_path.write_text(json.dumps(data, indent=2))
    print(f"\nResults saved: {out_path}")

    _print_summary(all_results)
    return all_results


def _print_summary(results: list[LongContextResult]):
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(
        f"{'Context':>8} {'Config':<10} {'Prefill':>8} {'Gen tok/s':>10} "
        f"{'Peak RSS':>9} {'Match':>6}"
    )
    print(f"{'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*9} {'-'*6}")

    contexts = sorted(set(r.context_tokens for r in results))
    for ctx in contexts:
        ctx_results = [r for r in results if r.context_tokens == ctx]
        baseline = next((r for r in ctx_results if r.config == "baseline"), None)
        for r in ctx_results:
            print(
                f"{r.context_tokens:>8} {r.config:<10} "
                f"{r.prefill_time_s:>7.1f}s {r.tokens_per_sec:>9.1f} "
                f"{r.peak_rss_gb:>8.1f}G {str(r.output_match):>6}"
            )
        if baseline and len(ctx_results) > 1:
            chunked = next((r for r in ctx_results if r.config == "chunked"), None)
            if chunked and chunked.peak_rss_gb > 0 and baseline.peak_rss_gb > 0:
                mem_saving = (1 - chunked.peak_rss_gb / baseline.peak_rss_gb) * 100
                tps_change = (chunked.tokens_per_sec / baseline.tokens_per_sec - 1) * 100
                print(
                    f"{'':>8} {'delta':<10} "
                    f"{'':>8} {tps_change:>+9.1f}% "
                    f"{mem_saving:>+7.1f}%"
                )
        print()


def main():
    p = argparse.ArgumentParser(description="tqai long-context benchmark")
    p.add_argument(
        "--model", "-m",
        default="mlx-community/Qwen2.5-7B-Instruct-8bit",
        help="MLX model ID",
    )
    p.add_argument(
        "--contexts",
        default="4096,8192,16384",
        help="Comma-separated context token counts",
    )
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--output", default="benchmarks/results/long_context.json")
    args = p.parse_args()

    contexts = [int(c) for c in args.contexts.split(",")]
    run_benchmark(
        model_id=args.model,
        contexts=contexts,
        chunk_size=args.chunk_size,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
