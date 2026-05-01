"""A/B benchmark on MLX with the fused-Metal compressed cache strategy.

Loads the same model under three configs:
    * baseline (no patch)
    * tqai 8/8, cache_strategy="compressed" (fused Metal decode)
    * tqai 4/4, cache_strategy="compressed"

For each config, measures on a fixed prompt + greedy decode:
    * Generated text (qualitative agreement with baseline)
    * Top-1 token agreement vs baseline over the decode horizon
    * Decode throughput (tokens/sec)
    * MLX peak GPU memory via mx.metal.get_peak_memory()
    * Process RSS delta as a sanity check

Run:
    PYTHONPATH=src python3 scripts/bench_mlx_compressed.py
"""

from __future__ import annotations

import argparse
import gc
import os
import time
import warnings
from dataclasses import dataclass

import mlx.core as mx
import psutil

warnings.filterwarnings("ignore")


def _force_eval(*arrays):
    """Trigger MLX evaluation via the standard streaming-graph commit call."""
    fn = getattr(mx, "eval")
    fn(*arrays)


@dataclass
class RunResult:
    label: str
    text: str
    tokens: list[int]
    decode_s: float
    decode_tps: float
    mlx_peak_mb: float
    rss_delta_mb: float


def greedy_decode(model, tok, prompt: str, n_new: int) -> tuple[str, list[int], float]:
    """Greedy decode using mlx_lm's stream_generate (cache-aware, deterministic
    via temp=0). Applies the model's chat template so instruct-tuned models
    actually produce a meaningful answer."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    if hasattr(tok, "apply_chat_template"):
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted = prompt

    out_tokens: list[int] = []
    pieces: list[str] = []
    sampler = make_sampler(temp=0.0)
    t0 = time.perf_counter()
    for resp in stream_generate(
        model, tok, formatted, max_tokens=n_new, sampler=sampler,
    ):
        out_tokens.append(resp.token)
        pieces.append(resp.text)
    elapsed = time.perf_counter() - t0
    return "".join(pieces), out_tokens, elapsed


def measure(label: str, model, tok, prompt: str, n_new: int) -> RunResult:
    proc = psutil.Process(os.getpid())
    mx.metal.reset_peak_memory()
    rss_before = proc.memory_info().rss / (1024 * 1024)

    text, tokens, elapsed = greedy_decode(model, tok, prompt, n_new)

    mlx_peak = mx.metal.get_peak_memory() / (1024 * 1024)
    rss_after = proc.memory_info().rss / (1024 * 1024)
    return RunResult(
        label=label,
        text=text.strip().replace("\n", " "),
        tokens=tokens,
        decode_s=elapsed,
        decode_tps=len(tokens) / elapsed if elapsed > 0 else 0.0,
        mlx_peak_mb=mlx_peak,
        rss_delta_mb=rss_after - rss_before,
    )


def top1_agreement(a: list[int], b: list[int]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if a[i] == b[i]) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    ap.add_argument(
        "--prompt",
        default=(
            "Write a short paragraph about the Eiffel Tower including its height, "
            "construction year, and architect."
        ),
    )
    ap.add_argument(
        "--prompt-repeat",
        type=int,
        default=1,
        help="Repeat the prompt N times to grow the context.",
    )
    ap.add_argument("--gen-tokens", type=int, default=80)
    args = ap.parse_args()

    from mlx_lm import load
    import tqai

    prompt = args.prompt * args.prompt_repeat

    print(f"Loading {args.model} ...")
    print(f"Prompt: ~{len(prompt.split())} words, "
          f"generating {args.gen_tokens} tokens via greedy decode\n")

    model, tok = load(args.model)
    baseline = measure("baseline", model, tok, prompt, args.gen_tokens)
    del model; gc.collect()

    model, tok = load(args.model)
    tqai.patch(model, bits_k=8, bits_v=8, backend="mlx", cache_strategy="compressed")
    tqai_8 = measure("tqai 8/8 (compressed)", model, tok, prompt, args.gen_tokens)
    del model; gc.collect()

    model, tok = load(args.model)
    tqai.patch(model, bits_k=4, bits_v=4, backend="mlx", cache_strategy="compressed")
    tqai_4 = measure("tqai 4/4 (compressed)", model, tok, prompt, args.gen_tokens)
    del model; gc.collect()

    rows = [baseline, tqai_8, tqai_4]
    print(f"{'config':<24} {'top1_v_base':>11} {'dec_s':>7} {'dec_tps':>8} "
          f"{'mlx_peak_MB':>12} {'rss_dMB':>9}")
    print("-" * 80)
    for r in rows:
        agree = top1_agreement(r.tokens, baseline.tokens) if r is not baseline else 1.0
        print(f"{r.label:<24} {agree:>11.4f} {r.decode_s:>7.3f} "
              f"{r.decode_tps:>8.2f} {r.mlx_peak_mb:>12.1f} {r.rss_delta_mb:>9.1f}")

    print()
    for r in rows:
        print(f"  {r.label}: {r.text[:140]!r}")


if __name__ == "__main__":
    main()
