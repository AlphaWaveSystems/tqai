"""Full forward-pass compression benchmark for tqai.

Measures memory, perplexity, token match rate, and throughput across
compression configs on a range of model sizes.

Usage:
    python benchmarks/benchmark_forward.py --backend torch --model Qwen/Qwen2.5-0.5B-Instruct
    python benchmarks/benchmark_forward.py --backend mlx \
        --model mlx-community/Llama-3.1-8B-Instruct-4bit
    python benchmarks/benchmark_forward.py --all-configs   --model Qwen/Qwen2.5-3B-Instruct
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


def _load_perp_mod():
    """Load eval_perplexity as a module regardless of working directory."""
    spec = importlib.util.spec_from_file_location(
        "eval_perplexity",
        Path(__file__).parent / "eval_perplexity.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Compression configs
# ---------------------------------------------------------------------------

CONFIGS = {
    "baseline":   dict(no_tqai=True),
    "kv-only":    dict(bits_k=4, bits_v=2),
    "kv+hidden8": dict(bits_k=4, bits_v=2, compress_hidden=True, bits_hidden=8),
    "kv+hidden6": dict(bits_k=4, bits_v=2, compress_hidden=True, bits_hidden=6),
    "kv+ffn8":    dict(bits_k=4, bits_v=2, compress_ffn=True,    bits_ffn=8),
    "all8":       dict(bits_k=4, bits_v=2, compress_hidden=True, bits_hidden=8,
                       compress_ffn=True, bits_ffn=8),
    "all6":       dict(bits_k=4, bits_v=2, compress_hidden=True, bits_hidden=6,
                       compress_ffn=True, bits_ffn=6),
    "aggressive": dict(bits_k=3, bits_v=2, compress_hidden=True, bits_hidden=6,
                       compress_ffn=True, bits_ffn=6),
}


@dataclass
class BenchResult:
    config_name: str
    peak_mb: float
    perplexity: float
    delta_ppl: float
    tokens_per_sec: float
    match_rate: float  # fraction of tokens matching baseline


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _torch_peak_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e6
    except ImportError:
        pass
    return 0.0


def _rss_mb() -> float:
    """Process resident set size in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Single-config run (HuggingFace / PyTorch)
# ---------------------------------------------------------------------------

def _run_hf_config(model, tokenizer, cfg_kwargs: dict, baseline_tokens: Optional[list]):
    import torch

    import tqai

    no_tqai = cfg_kwargs.get("no_tqai", False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mem_before = _rss_mb()

    cache = None
    if not no_tqai:
        patch_kw = {k: v for k, v in cfg_kwargs.items() if k != "no_tqai"}
        cache = tqai.patch(model, backend="torch", **patch_kw)

    # Perplexity uses lazy import to avoid triggering hook on module name
    perp_mod = _load_perp_mod()
    ppl = perp_mod.perplexity_hf(model, tokenizer)

    t0 = time.perf_counter()
    tokens = perp_mod.generate_tokens(model, tokenizer, backend="torch", max_new_tokens=100)
    elapsed = time.perf_counter() - t0
    tps = 100 / elapsed if elapsed > 0 else 0.0

    peak_mb = max(_torch_peak_mb(), _rss_mb() - mem_before)

    match = perp_mod.compute_match_rate(baseline_tokens, tokens) if baseline_tokens else 0.0

    if cache is not None:
        tqai.unpatch(model)

    gc.collect()
    return ppl, tps, peak_mb, tokens, match


# ---------------------------------------------------------------------------
# Single-config run (MLX)
# ---------------------------------------------------------------------------

def _run_mlx_config(model, tokenizer, cfg_kwargs: dict, baseline_tokens: Optional[list]):
    import tqai

    no_tqai = cfg_kwargs.get("no_tqai", False)
    mem_before = _rss_mb()

    if not no_tqai:
        # Forward hooks not yet supported on MLX (Phase 4)
        patch_kw = {k: v for k, v in cfg_kwargs.items()
                    if k not in ("no_tqai", "compress_hidden", "compress_ffn",
                                 "compress_attn_logits", "bits_hidden", "bits_ffn", "bits_attn")}
        tqai.patch(model, backend="mlx", **patch_kw)

    perp_mod = _load_perp_mod()
    ppl = perp_mod.perplexity_mlx(model, tokenizer)

    # Warmup: one generation to trigger MLX JIT compilation before timing
    perp_mod.generate_tokens(model, tokenizer, backend="mlx", max_new_tokens=20)

    t0 = time.perf_counter()
    tokens = perp_mod.generate_tokens(model, tokenizer, backend="mlx", max_new_tokens=100)
    elapsed = time.perf_counter() - t0
    tps = 100 / elapsed if elapsed > 0 else 0.0

    peak_mb = _rss_mb() - mem_before

    match = perp_mod.compute_match_rate(baseline_tokens, tokens) if baseline_tokens else 0.0

    if not no_tqai:
        tqai.unpatch(model)

    return ppl, tps, peak_mb, tokens, match


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------

def run_benchmark(model_id: str, backend: str, configs: list[str]) -> list[BenchResult]:
    print(f"\n{'='*60}")
    print("tqai Full Forward Pass Benchmark")
    print(f"Model:   {model_id}")
    print(f"Backend: {backend}")
    print(f"Configs: {', '.join(configs)}")
    print(f"{'='*60}\n")

    if backend == "mlx":
        import mlx_lm
        print(f"Loading {model_id}...")
        model, tokenizer = mlx_lm.load(model_id)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        model.eval()

    results: list[BenchResult] = []
    baseline_ppl: Optional[float] = None
    baseline_tokens: Optional[list] = None

    for config_name in configs:
        if config_name not in CONFIGS:
            print(f"Unknown config '{config_name}', skipping")
            continue

        cfg_kwargs = CONFIGS[config_name]
        print(f"  {config_name:<16} ...", end="", flush=True)

        try:
            if backend == "mlx":
                ppl, tps, peak_mb, tokens, match = _run_mlx_config(
                    model, tokenizer, cfg_kwargs, baseline_tokens
                )
            else:
                ppl, tps, peak_mb, tokens, match = _run_hf_config(
                    model, tokenizer, cfg_kwargs, baseline_tokens
                )

            if config_name == "baseline":
                baseline_ppl = ppl
                baseline_tokens = tokens
                match = 1.0

            delta_ppl = ppl - baseline_ppl if baseline_ppl is not None else 0.0
            result = BenchResult(
                config_name=config_name,
                peak_mb=peak_mb,
                perplexity=ppl,
                delta_ppl=delta_ppl,
                tokens_per_sec=tps,
                match_rate=match,
            )
            results.append(result)
            print(f" ppl={ppl:.2f}  Δ={delta_ppl:+.2f}  {tps:.1f}tok/s  match={match:.0%}")

        except Exception as exc:
            print(f" ERROR: {exc}")

    _print_table(results, model_id)
    _save_results(results, model_id, backend)
    return results


def _print_table(results: list[BenchResult], model_id: str) -> None:
    if not results:
        return
    print(f"\n{'='*72}")
    print(f"Results: {model_id}")
    print(f"{'='*72}")
    print(f"{'Config':<16} {'PeakMB':>9} {'PPL':>7} {'ΔPPL':>7} {'tok/s':>7} {'Match':>7}")
    print(f"{'-'*16} {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    baseline_mb = next((r.peak_mb for r in results if r.config_name == "baseline"), None)
    for r in results:
        mb_str = f"{r.peak_mb:>7.0f}"
        if baseline_mb and r.config_name != "baseline" and r.peak_mb > 0:
            saved_pct = (1 - r.peak_mb / baseline_mb) * 100
            mb_str += f"({saved_pct:+.0f}%)"
        print(
            f"{r.config_name:<16} {mb_str:>9}  "
            f"{r.perplexity:>7.2f} {r.delta_ppl:>+7.2f} "
            f"{r.tokens_per_sec:>7.1f} {r.match_rate:>7.0%}"
        )
    print()


def _save_results(results: list[BenchResult], model_id: str, backend: str) -> None:
    """Save results to benchmarks/results/ as a text file."""
    import json

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    safe_name = model_id.replace("/", "_")
    out_path = out_dir / f"{safe_name}_{backend}.json"

    data = {
        "model": model_id,
        "backend": backend,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [
            {
                "config": r.config_name,
                "peak_mb": round(r.peak_mb, 2),
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="tqai full forward-pass benchmark")
    p.add_argument("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace or mlx-community model ID")
    p.add_argument("--backend", default=None, choices=["torch", "mlx"],
                   help="Backend (auto-detected if omitted)")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                   choices=list(CONFIGS.keys()),
                   help="Configs to run")
    p.add_argument("--all-configs", action="store_true",
                   help="Run all configs")
    args = p.parse_args()

    backend = args.backend
    if backend is None:
        try:
            from tqai.backend import detect_backend
            backend = detect_backend()
        except RuntimeError:
            backend = "torch"

    configs = list(CONFIGS.keys()) if args.all_configs else args.configs
    run_benchmark(model_id=args.model, backend=backend, configs=configs)


if __name__ == "__main__":
    main()
