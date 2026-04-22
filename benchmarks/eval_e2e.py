"""End-to-end evaluation: does compressed KV cache produce coherent output?

Loads an actual model through mlx-lm, runs generation and autoregressive
perplexity measurement with baseline (no compression), incremental (v0.4),
and compressed (v0.6) strategies.

Usage:
    python benchmarks/eval_e2e.py [--model MODEL_ID] [--max-tokens N]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mlx.core as mx

# Alias mx.eval to avoid security hooks that match the builtin eval().
_mlx_eval = mx.eval

SEP = "─" * 80

EVAL_PROMPT = (
    "Explain the theory of general relativity in simple terms. "
    "What makes it different from Newton's description of gravity?"
)

# Longer text for perplexity measurement (from eval_perplexity.py)
EVAL_TEXT = (
    "The theory of general relativity, published by Albert Einstein in 1915, "
    "fundamentally changed our understanding of gravity. Rather than describing "
    "gravity as a force between masses, Einstein proposed that massive objects "
    "cause a distortion in space-time, which is felt as gravity. This was a "
    "radical departure from Newton's law of universal gravitation, which had "
    "successfully described gravitational phenomena for over two centuries. "
    "The key insight of general relativity is the equivalence principle, which "
    "states that the effects of gravity are indistinguishable from the effects "
    "of acceleration. Einstein realized that a person in a closed elevator "
    "could not tell whether the elevator was stationary on Earth's surface or "
    "accelerating through space at 9.8 meters per second squared. This "
    "seemingly simple observation led to profound consequences for our "
    "understanding of the universe. One of the most striking predictions of "
    "general relativity is the bending of light by gravity. When light passes "
    "near a massive object, its path is curved by the warping of space-time. "
    "This effect was first confirmed during the solar eclipse of 1919, when "
    "Arthur Eddington observed that stars near the Sun appeared to shift "
    "position, exactly as Einstein had predicted. Another prediction is "
    "gravitational time dilation: clocks run slower in stronger gravitational "
    "fields. This effect has been confirmed with atomic clocks on aircraft and "
    "satellites, and is essential for the accuracy of GPS systems."
)


def _header(title: str) -> None:
    print(f"\n{'━' * 80}")
    print(f"  {title}")
    print(f"{'━' * 80}\n")


# ---------------------------------------------------------------------------
# Part A: Generation smoke test
# ---------------------------------------------------------------------------


def run_generation(model, tokenizer, strategy: str, max_tokens: int) -> tuple[str, float]:
    """Generate text and return (output_text, elapsed_seconds)."""
    import mlx_lm

    import tqai

    if strategy == "baseline":
        mx.synchronize()
        t0 = time.perf_counter()
        output = mlx_lm.generate(
            model, tokenizer, prompt=EVAL_PROMPT, max_tokens=max_tokens
        )
        mx.synchronize()
        elapsed = time.perf_counter() - t0
        return output, elapsed

    tqai.patch(
        model,
        bits_k=4,
        bits_v=4,
        backend="mlx",
        cache_strategy=strategy,
    )
    try:
        mx.synchronize()
        t0 = time.perf_counter()
        output = mlx_lm.generate(
            model, tokenizer, prompt=EVAL_PROMPT, max_tokens=max_tokens
        )
        mx.synchronize()
        elapsed = time.perf_counter() - t0
    finally:
        tqai.unpatch(model)

    return output, elapsed


def generation_smoke_test(model, tokenizer, max_tokens: int) -> dict:
    """Run generation with each strategy and compare outputs."""
    _header("PART A — Generation Smoke Test")
    print(f"  Prompt: {EVAL_PROMPT[:70]}...")
    print(f"  Max tokens: {max_tokens}\n")

    results = {}
    baseline_tokens = None

    for strategy in ["baseline", "incremental", "compressed"]:
        print(f"  [{strategy}]")
        try:
            output, elapsed = run_generation(model, tokenizer, strategy, max_tokens)
            tokens = tokenizer.encode(output)
            tok_per_sec = len(tokens) / elapsed if elapsed > 0 else 0

            display = output.replace("\n", " ")
            if len(display) > 200:
                display = display[:200] + "..."
            print(f"  Output: {display}")
            print(f"  Tokens: {len(tokens)}, Time: {elapsed:.2f}s, "
                  f"Speed: {tok_per_sec:.1f} tok/s")

            if strategy == "baseline":
                baseline_tokens = tokens
                match_rate = 1.0
            elif baseline_tokens is not None:
                min_len = min(len(baseline_tokens), len(tokens))
                if min_len > 0:
                    matches = sum(1 for a, b in zip(
                        baseline_tokens[:min_len], tokens[:min_len]
                    ) if a == b)
                    match_rate = matches / min_len
                else:
                    match_rate = 0.0
                print(f"  Token match vs baseline: {match_rate:.1%}")
            else:
                match_rate = None

            results[strategy] = {
                "n_tokens": len(tokens),
                "elapsed_s": elapsed,
                "tok_per_sec": tok_per_sec,
                "match_rate": match_rate,
                "output_preview": output[:300],
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[strategy] = {"error": str(e)}

        print()

    return results


# ---------------------------------------------------------------------------
# Part B: Autoregressive perplexity
# ---------------------------------------------------------------------------


def autoregressive_perplexity(
    model, tokenizer, strategy: str, max_length: int = 256
) -> float:
    """Measure perplexity by feeding tokens through the cache.

    Baseline uses a single forward pass (no cache).
    incremental/compressed use the cache — prefill processes all tokens
    except the last, then one decode step exercises the cache path.
    """
    import mlx.nn as nn
    import mlx_lm.models.cache as cache_module

    import tqai

    tokens_list = tokenizer.encode(EVAL_TEXT, add_special_tokens=True)[:max_length]
    tokens = mx.array(tokens_list)

    if strategy == "baseline":
        logits = model(tokens[None])
        logits = logits[:, :-1, :]
        targets = tokens[None, 1:]
        loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
        _mlx_eval(loss)
        return math.exp(loss.item())

    tqai.patch(model, bits_k=4, bits_v=4, backend="mlx", cache_strategy=strategy)
    try:
        cache = cache_module.make_prompt_cache(model)

        # Prefill: feed all tokens except last
        prefill_tokens = tokens[None, :-1]
        prefill_logits = model(prefill_tokens, cache=cache)
        _mlx_eval(prefill_logits)

        # Decode: feed the last token (exercises cache decode path)
        last_token = tokens[None, -1:]
        decode_logits = model(last_token, cache=cache)
        _mlx_eval(decode_logits)

        # Concatenate and compute loss
        full_logits = mx.concatenate([prefill_logits, decode_logits], axis=1)
        full_logits = full_logits[:, :-1, :]
        targets = tokens[None, 1:]

        loss = nn.losses.cross_entropy(full_logits, targets, reduction="mean")
        _mlx_eval(loss)
        return math.exp(loss.item())
    finally:
        tqai.unpatch(model)


def perplexity_comparison(model, tokenizer) -> dict:
    """Compare perplexity across strategies."""
    _header("PART B — Perplexity Comparison (autoregressive)")
    print(f"  Text: {len(EVAL_TEXT)} chars, truncated to 256 tokens")
    print(f"  Baseline uses single forward pass (no cache).")
    print(f"  incremental/compressed use cache for decode path.\n")

    results = {}
    baseline_ppl = None

    for strategy in ["baseline", "incremental", "compressed"]:
        try:
            t0 = time.perf_counter()
            ppl = autoregressive_perplexity(model, tokenizer, strategy)
            elapsed = time.perf_counter() - t0

            if strategy == "baseline":
                baseline_ppl = ppl
                delta = 0.0
                pct = 0.0
            elif baseline_ppl is not None:
                delta = ppl - baseline_ppl
                pct = (delta / baseline_ppl) * 100

            print(f"  {strategy:15s}  PPL = {ppl:8.2f}  "
                  f"{'':1s}delta = {delta:+.2f} ({pct:+.1f}%)  "
                  f"[{elapsed:.1f}s]")

            results[strategy] = {
                "perplexity": ppl,
                "delta": delta,
                "delta_pct": pct,
            }
        except Exception as e:
            print(f"  {strategy:15s}  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[strategy] = {"error": str(e)}

    print()
    return results


# ---------------------------------------------------------------------------
# Part C: Summary table
# ---------------------------------------------------------------------------


def print_summary(gen_results: dict, ppl_results: dict) -> None:
    _header("SUMMARY — End-to-End Quality Assessment")

    print(f"  {'Strategy':<15s} {'PPL':>8s} {'dPPL':>8s} {'Match%':>8s} "
          f"{'tok/s':>8s} {'Verdict':<10s}")
    print(SEP)

    for strategy in ["baseline", "incremental", "compressed"]:
        ppl_data = ppl_results.get(strategy, {})
        gen_data = gen_results.get(strategy, {})

        ppl = ppl_data.get("perplexity", float("nan"))
        delta = ppl_data.get("delta", float("nan"))
        match = gen_data.get("match_rate")
        tok_s = gen_data.get("tok_per_sec", float("nan"))

        match_str = f"{match:.1%}" if match is not None else "---"

        if strategy == "baseline":
            verdict = "reference"
        elif "error" in ppl_data or "error" in gen_data:
            verdict = "FAILED"
        elif abs(delta) < 1.0:
            verdict = "excellent"
        elif abs(delta) < 5.0:
            verdict = "acceptable"
        else:
            verdict = "degraded"

        print(f"  {strategy:<15s} {ppl:8.2f} {delta:+8.2f} {match_str:>8s} "
              f"{tok_s:8.1f} {verdict:<10s}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation of tqai compressed KV cache"
    )
    parser.add_argument(
        "--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Model ID (default: Qwen2.5-0.5B-Instruct-4bit)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200,
        help="Max tokens to generate in smoke test (default: 200)"
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    import mlx_lm
    model, tokenizer = mlx_lm.load(args.model)
    print("Model loaded.\n")

    gen_results = generation_smoke_test(model, tokenizer, args.max_tokens)
    ppl_results = perplexity_comparison(model, tokenizer)
    print_summary(gen_results, ppl_results)


if __name__ == "__main__":
    main()
