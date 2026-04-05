"""tqai v0.3 benchmark: rotation baking vs runtime rotation vs QJL variants.

Compares five modes side-by-side on each model:
  baseline        — no compression
  tqai-runtime    — current behaviour (rotation at every token)
  tqai-baked      — rotation pre-baked into weights (target: ~baseline throughput)
  tqai-qjl-only   — runtime rotation + QJL Stage 2 (to isolate QJL effect)
  tqai-baked+qjl  — baked rotation + QJL (best of both)

Usage:
    python benchmarks/benchmark_bake.py --model Qwen/Qwen2.5-0.5B-Instruct
    python benchmarks/benchmark_bake.py --model Qwen/Qwen2.5-3B-Instruct --bits-k 4 --bits-v 2
    python benchmarks/benchmark_bake.py --modes baseline tqai-runtime tqai-baked
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

ALL_MODES = ["baseline", "tqai-runtime", "tqai-baked", "tqai-qjl-only", "tqai-baked+qjl"]


def _load_perp_mod():
    spec = importlib.util.spec_from_file_location(
        "perp_mod", Path(__file__).parent / "eval_perplexity.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class BenchResult:
    mode: str
    perplexity: float
    delta_ppl: float
    tokens_per_sec: float
    match_rate: float


def _run_mode(
    model,
    tokenizer,
    mode: str,
    baseline_tokens: Optional[list],
    bits_k: int,
    bits_v: int,
    baked_dir: Optional[Path],
) -> tuple[float, float, list]:
    """Run one mode, return (ppl, tps, tokens)."""

    import tqai

    perp_mod = _load_perp_mod()
    cache = None

    patch_kwargs = dict(bits_k=bits_k, bits_v=bits_v, backend="torch")

    if mode == "baseline":
        pass
    elif mode == "tqai-runtime":
        cache = tqai.patch(model, **patch_kwargs)
    elif mode == "tqai-baked":
        cache = tqai.patch(model, pre_rotated=True, **patch_kwargs)
    elif mode == "tqai-qjl-only":
        cache = tqai.patch(model, use_qjl=True, **patch_kwargs)
    elif mode == "tqai-baked+qjl":
        cache = tqai.patch(model, pre_rotated=True, use_qjl=True, **patch_kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ppl = perp_mod.perplexity_hf(model, tokenizer)

    t0 = time.perf_counter()
    tokens = perp_mod.generate_tokens(model, tokenizer, backend="torch", max_new_tokens=100)
    elapsed = time.perf_counter() - t0
    tps = 100 / elapsed if elapsed > 0 else 0.0

    if cache is not None:
        tqai.unpatch(model)

    gc.collect()
    return ppl, tps, tokens


def _load_baked_model(model_id: str, bits_k: int, bits_v: int, seed: int, baked_dir: Path):
    """Bake a model and return (baked_model, tokenizer, baked_dir)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tqai.bake import save_baked_model

    print(f"  Baking model to {baked_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_fp = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model_fp.train(False)

    save_baked_model(
        model=model_fp,
        tokenizer=tokenizer,
        output_dir=baked_dir,
        base_model_id=model_id,
        bits_k=bits_k,
        bits_v=bits_v,
        seed=seed,
    )
    del model_fp
    gc.collect()

    print(f"  Loading baked model from {baked_dir} ...")
    baked_model = AutoModelForCausalLM.from_pretrained(
        str(baked_dir), torch_dtype=torch.bfloat16
    )
    baked_model.train(False)
    return baked_model, tokenizer


def run_benchmark(
    model_id: str,
    modes: list[str],
    bits_k: int = 4,
    bits_v: int = 2,
    seed: int = 42,
    tmp_dir: Optional[Path] = None,
) -> list[BenchResult]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*64}")
    print("tqai Rotation Baking Benchmark")
    print(f"Model:   {model_id}")
    print(f"K{bits_k}/V{bits_v}  seed={seed}")
    print(f"Modes:   {', '.join(modes)}")
    print(f"{'='*64}\n")

    # Determine if we need a baked model
    needs_baked = any(m in modes for m in ("tqai-baked", "tqai-baked+qjl"))
    baked_dir = None
    baked_model = None
    baked_tokenizer = None

    if needs_baked:
        if tmp_dir is None:
            import tempfile
            tmp_dir = Path(tempfile.mkdtemp(prefix="tqai_baked_"))
        baked_dir = tmp_dir / "baked_model"
        baked_model, baked_tokenizer = _load_baked_model(
            model_id, bits_k, bits_v, seed, baked_dir
        )

    # Load runtime model (for non-baked modes)
    runtime_modes = [m for m in modes if m not in ("tqai-baked", "tqai-baked+qjl")]
    results: list[BenchResult] = []
    baseline_ppl: Optional[float] = None
    baseline_tokens: Optional[list] = None

    if runtime_modes:
        print(f"Loading {model_id} (runtime)...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        model.train(False)

        for mode in modes:
            if mode in ("tqai-baked", "tqai-baked+qjl"):
                continue
            print(f"  {mode:<20} ...", end="", flush=True)
            try:
                ppl, tps, tokens = _run_mode(
                    model, tokenizer, mode, baseline_tokens, bits_k, bits_v, baked_dir
                )
                if mode == "baseline":
                    baseline_ppl = ppl
                    baseline_tokens = tokens
                delta = ppl - baseline_ppl if baseline_ppl is not None else 0.0
                perp_mod = _load_perp_mod()
                is_baseline = mode == "baseline"
                match = (
                    perp_mod.compute_match_rate(baseline_tokens, tokens)
                    if baseline_tokens and not is_baseline
                    else 1.0
                )
                results.append(BenchResult(
                    mode=mode, perplexity=ppl, delta_ppl=delta,
                    tokens_per_sec=tps, match_rate=match,
                ))
                print(f" ppl={ppl:.3f} Δ={delta:+.3f} {tps:.1f}tok/s match={match:.0%}")
            except Exception as exc:
                print(f" ERROR: {exc}")

        del model
        gc.collect()

    # Run baked modes
    if needs_baked and baked_model is not None:
        for mode in ("tqai-baked", "tqai-baked+qjl"):
            if mode not in modes:
                continue
            print(f"  {mode:<20} ...", end="", flush=True)
            try:
                ppl, tps, tokens = _run_mode(
                    baked_model, baked_tokenizer, mode,
                    baseline_tokens, bits_k, bits_v, baked_dir
                )
                delta = ppl - baseline_ppl if baseline_ppl is not None else 0.0
                perp_mod = _load_perp_mod()
                match = (
                    perp_mod.compute_match_rate(baseline_tokens, tokens)
                    if baseline_tokens else 1.0
                )
                results.append(BenchResult(
                    mode=mode, perplexity=ppl, delta_ppl=delta,
                    tokens_per_sec=tps, match_rate=match,
                ))
                print(f" ppl={ppl:.3f} Δ={delta:+.3f} {tps:.1f}tok/s match={match:.0%}")
            except Exception as exc:
                print(f" ERROR: {exc}")

        del baked_model
        gc.collect()

    _print_table(results, model_id, bits_k, bits_v)
    _save_results(results, model_id, bits_k, bits_v)
    return results


def _print_table(results: list[BenchResult], model_id: str, bits_k: int, bits_v: int) -> None:
    if not results:
        return
    baseline_tps = next((r.tokens_per_sec for r in results if r.mode == "baseline"), None)

    print(f"\n{'='*72}")
    print(f"Results: {model_id}  K{bits_k}/V{bits_v}")
    print(f"{'='*72}")
    print(f"{'Mode':<22} {'PPL':>7} {'ΔPPL':>7} {'tok/s':>7} {'vs base':>8} {'Match':>7}")
    print(f"{'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")

    for r in results:
        vs_base = ""
        if baseline_tps and r.tokens_per_sec > 0:
            vs_base = f"{r.tokens_per_sec / baseline_tps:.0%}"
        print(
            f"{r.mode:<22} {r.perplexity:>7.3f} {r.delta_ppl:>+7.3f} "
            f"{r.tokens_per_sec:>7.1f} {vs_base:>8} {r.match_rate:>7.0%}"
        )
    print()


def _save_results(results: list[BenchResult], model_id: str, bits_k: int, bits_v: int) -> None:
    import json

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    safe_name = model_id.replace("/", "_")
    out_path = out_dir / f"{safe_name}_bake_k{bits_k}v{bits_v}.json"

    data = {
        "model": model_id,
        "bits_k": bits_k,
        "bits_v": bits_v,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [
            {
                "mode": r.mode,
                "perplexity": round(r.perplexity, 4),
                "delta_ppl": round(r.delta_ppl, 4),
                "tokens_per_sec": round(r.tokens_per_sec, 2),
                "match_rate": round(r.match_rate, 4),
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Results saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description="tqai rotation baking + QJL benchmark")
    p.add_argument("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace model ID")
    p.add_argument("--bits-k", type=int, default=4, help="Key bits (default: 4)")
    p.add_argument("--bits-v", type=int, default=2, help="Value bits (default: 2)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    p.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES,
                   help="Modes to run (default: all)")
    p.add_argument("--tmp-dir", default=None, help="Directory for temporary baked model")
    args = p.parse_args()

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else None
    run_benchmark(
        model_id=args.model,
        modes=args.modes,
        bits_k=args.bits_k,
        bits_v=args.bits_v,
        seed=args.seed,
        tmp_dir=tmp_dir,
    )


if __name__ == "__main__":
    main()
