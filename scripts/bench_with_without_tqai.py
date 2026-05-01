"""A/B benchmark: model with vs. without tqai KV cache compression.

Loads Qwen2.5-0.5B-Instruct once, then for each config:
    * Baseline (no patch)
    * tqai 8-bit (default, near-lossless)
    * tqai 4-bit (aggressive, expected quality drop)

Measures, on a held-out passage and a generation prompt:
    * Logit cosine similarity vs baseline (per-position, averaged)
    * Greedy next-token agreement vs baseline (top-1 match rate)
    * Perplexity over the passage
    * Prefill + decode wall-clock and tokens/sec
    * Peak process RSS

Run:
    PYTHONPATH=src python3 scripts/bench_with_without_tqai.py
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import time
import warnings
from dataclasses import dataclass

import psutil

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import tqai

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


PASSAGE = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in "
    "Paris, France. It is named after the engineer Gustave Eiffel, whose company "
    "designed and built the tower for the 1889 World's Fair. The tower stands "
    "330 metres tall and was the world's tallest man-made structure for 41 "
    "years until the Chrysler Building in New York City was completed in 1930. "
    "It is the most-visited paid monument in the world: 6.91 million people "
    "ascended it in 2015. The tower is 324 metres tall, about the same height "
    "as an 81-storey building, and the tallest structure in Paris. Its base is "
    "square, measuring 125 metres on each side."
)

GEN_PROMPT = "List three famous landmarks in Paris and one fact about each:"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PROC = psutil.Process(os.getpid())


def rss_mb() -> float:
    return _PROC.memory_info().rss / (1024 * 1024)


def fresh_model(name: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.train(False)
    return model


@dataclass
class RunResult:
    label: str
    cos_logits: float
    top1_agree: float
    ppl: float
    prefill_s: float
    decode_s: float
    decode_tps: float
    peak_mb: float
    generated: str


def measure(
    label: str,
    model,
    tok,
    cache_factory,
    baseline_logits: torch.Tensor | None,
    baseline_top1: torch.Tensor | None,
    new_tokens: int,
) -> RunResult:
    enc = tok(PASSAGE, return_tensors="pt")
    input_ids = enc.input_ids
    n_tokens = input_ids.shape[1]

    rss_before = rss_mb()
    rss_peak = rss_before

    # ---- Quality / perplexity pass ----
    cache = cache_factory()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    prefill_s = time.perf_counter() - t0
    rss_peak = max(rss_peak, rss_mb())

    logits = out.logits  # (1, S, V)
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits[:, :-1, :].to(torch.float32), dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()
    ppl = float(math.exp(nll.item()))

    if baseline_logits is None:
        cos = 1.0
        top1 = 1.0
    else:
        l_flat = logits.reshape(-1, logits.shape[-1]).to(torch.float32)
        b_flat = baseline_logits.reshape(-1, baseline_logits.shape[-1]).to(torch.float32)
        cos = float(F.cosine_similarity(l_flat, b_flat, dim=-1).mean())
        my_top1 = logits.argmax(dim=-1)
        top1 = float((my_top1 == baseline_top1).float().mean())

    # ---- Generation pass ----
    enc2 = tok(GEN_PROMPT, return_tensors="pt")
    cache2 = cache_factory()
    t0 = time.perf_counter()
    with torch.no_grad():
        gen_ids = model.generate(
            **enc2,
            max_new_tokens=new_tokens,
            do_sample=False,
            past_key_values=cache2,
            pad_token_id=tok.eos_token_id,
        )
    decode_s = time.perf_counter() - t0
    rss_peak = max(rss_peak, rss_mb())
    decode_tps = new_tokens / decode_s if decode_s > 0 else 0.0
    generated = tok.decode(gen_ids[0, enc2.input_ids.shape[1] :], skip_special_tokens=True)

    peak_mb = rss_peak - rss_before

    return RunResult(
        label=label,
        cos_logits=cos,
        top1_agree=top1,
        ppl=ppl,
        prefill_s=prefill_s,
        decode_s=decode_s,
        decode_tps=decode_tps,
        peak_mb=peak_mb,
        generated=generated.strip().replace("\n", " "),
    ), logits.detach(), logits.argmax(dim=-1).detach()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--gen-tokens", type=int, default=40)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Loading {args.model} ({args.dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"Passage tokens : {len(tok(PASSAGE).input_ids)}")
    print(f"Gen prompt tok : {len(tok(GEN_PROMPT).input_ids)}  →  +{args.gen_tokens} new")
    print()

    # ---- Baseline ----
    print("Run 1/3  baseline (no patch) ...")
    model = fresh_model(args.model, dtype)
    baseline, baseline_logits, baseline_top1 = measure(
        "baseline", model, tok,
        cache_factory=lambda: None,  # HF will allocate DynamicCache
        baseline_logits=None,
        baseline_top1=None,
        new_tokens=args.gen_tokens,
    )
    del model; gc.collect()

    # ---- tqai 8-bit ----
    print("Run 2/3  tqai 8/8 ...")
    model = fresh_model(args.model, dtype)
    cache_proto = tqai.patch(model, bits_k=8, bits_v=8, backend="torch")
    # The patch returns a single cache instance; for a clean per-call cache we
    # rebuild via the same config used to make cache_proto.
    config_8 = cache_proto.tq_config  # internal access for benching
    from tqai.cache.hf import TurboQuantDynamicCache
    r_tqai8, _, _ = measure(
        "tqai 8/8", model, tok,
        cache_factory=lambda: TurboQuantDynamicCache(config_8),
        baseline_logits=baseline_logits,
        baseline_top1=baseline_top1,
        new_tokens=args.gen_tokens,
    )
    del model; gc.collect()

    # ---- tqai 4-bit ----
    print("Run 3/3  tqai 4/4 ...")
    model = fresh_model(args.model, dtype)
    cache_proto = tqai.patch(model, bits_k=4, bits_v=4, backend="torch")
    config_4 = cache_proto.tq_config
    r_tqai4, _, _ = measure(
        "tqai 4/4", model, tok,
        cache_factory=lambda: TurboQuantDynamicCache(config_4),
        baseline_logits=baseline_logits,
        baseline_top1=baseline_top1,
        new_tokens=args.gen_tokens,
    )
    del model; gc.collect()

    # ---- Report ----
    rows = [baseline, r_tqai8, r_tqai4]
    print()
    print(f"{'config':<12} {'cos':>7} {'top1':>7} {'ppl':>8} "
          f"{'prefill_s':>10} {'dec_s':>7} {'dec_tps':>8} {'peak_dMB':>9}")
    print("-" * 77)
    for r in rows:
        print(f"{r.label:<12} {r.cos_logits:>7.4f} {r.top1_agree:>7.4f} "
              f"{r.ppl:>8.3f} {r.prefill_s:>10.3f} {r.decode_s:>7.3f} "
              f"{r.decode_tps:>8.2f} {r.peak_mb:>9.1f}")

    print()
    for r in rows:
        print(f"  {r.label}: {r.generated[:120]!r}")


if __name__ == "__main__":
    main()
