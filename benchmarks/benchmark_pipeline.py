"""Comprehensive pipeline benchmark suite for tqai v0.4.

Tests every scorer × strategy combination on synthetic data matching
real model dimensions (Qwen, Gemma, Llama, WAN 2.2).

Usage:
    python benchmarks/benchmark_pipeline.py
    python benchmarks/benchmark_pipeline.py --json results/pipeline_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Ensure all plugins registered
import tqai.monitors  # noqa: F401
import tqai.scorers  # noqa: F401
import tqai.strategies  # noqa: F401
from tqai.backend import get_backend
from tqai.config import TurboQuantConfig
from tqai.pipeline import build_pipeline
from tqai.pipeline.registry import list_available
from tqai.quantizer import PolarQuantizer
from tqai.quantizer_rotor import RotorQuantizer

# ---------------------------------------------------------------------------
# Model dimension profiles
# ---------------------------------------------------------------------------

MODEL_PROFILES = {
    "Qwen2.5-0.5B":   {"head_dim": 64,  "n_heads": 14, "n_kv_heads": 2,  "n_layers": 24,  "family": "llm"},
    "Qwen2.5-3B":     {"head_dim": 128, "n_heads": 16, "n_kv_heads": 2,  "n_layers": 36,  "family": "llm"},
    "Qwen2.5-7B":     {"head_dim": 128, "n_heads": 28, "n_kv_heads": 4,  "n_layers": 28,  "family": "llm"},
    "Gemma-2B":        {"head_dim": 256, "n_heads": 8,  "n_kv_heads": 1,  "n_layers": 18,  "family": "llm"},
    "Gemma-7B":        {"head_dim": 256, "n_heads": 16, "n_kv_heads": 16, "n_layers": 28,  "family": "llm"},
    "Llama-3.1-8B":    {"head_dim": 128, "n_heads": 32, "n_kv_heads": 8,  "n_layers": 32,  "family": "llm"},
    "WAN2.2-5B":       {"head_dim": 128, "n_heads": 12, "n_kv_heads": 12, "n_layers": 30,  "family": "dit"},
}

# Pipeline configs to benchmark
PIPELINE_CONFIGS = {
    "baseline":           None,
    "palm+tiered":        {"scorer": "palm",   "strategy": "tiered"},
    "fisher+tiered":      {"scorer": "fisher", "strategy": "tiered"},
    "sheaf+tiered":       {"scorer": "sheaf",  "strategy": "tiered"},
    "bsa+tiered":         {"scorer": "bsa",    "strategy": "tiered"},
    "palm+delta":         {"scorer": "palm",   "strategy": "delta"},
    "snr+delta2":         {"scorer": "snr",    "strategy": "delta2"},
    "palm+window":        {"scorer": "palm",   "strategy": "window"},
    "fisher+delta":       {"scorer": "fisher", "strategy": "delta"},
    "sheaf+delta2":       {"scorer": "sheaf",  "strategy": "delta2"},
    "bsa+window":         {"scorer": "bsa",    "strategy": "window"},
    "palm+tiered+stab":   {"scorer": "palm",   "strategy": "tiered", "monitor": "stability"},
    "snr+delta2+lyap":    {"scorer": "snr",    "strategy": "delta2", "monitor": "lyapunov"},
    "skip_layers":        {"scorer": "palm",   "strategy": "tiered", "skip_layers": [0, 1, 2, 3]},
    # RotorQuant configs (block-diagonal Clifford rotor rotation, Pope 2026)
    "rotorquant+bare":    {"_quantizer": "rotor"},
    "rotorquant+tiered":  {"scorer": "palm",   "strategy": "tiered", "_quantizer": "rotor"},
    "rotorquant+delta":   {"scorer": "palm",   "strategy": "delta",  "_quantizer": "rotor"},
    "rotorquant+delta2":  {"scorer": "snr",    "strategy": "delta2", "_quantizer": "rotor"},
    "rotorquant+window":  {"scorer": "palm",   "strategy": "window", "_quantizer": "rotor"},
}


@dataclass
class PipelineBenchResult:
    model: str
    family: str
    config_name: str
    head_dim: int
    bits_k: int
    bits_v: int
    nmse: float
    cosine_sim: float
    compress_ms: float
    decompress_ms: float
    compression_ratio: float
    scorer_used: str
    strategy_used: str
    monitor_used: str
    n_tokens: int
    n_layers_tested: int


def _generate_synthetic_kv(head_dim: int, n_kv_heads: int, seq_len: int, n_steps: int = 5):
    """Generate synthetic KV data mimicking real model activations."""
    import torch
    # Generate a base tensor + small per-step perturbations (mimics denoising)
    base = torch.randn(1, n_kv_heads, seq_len, head_dim)
    steps = []
    for i in range(n_steps):
        noise_scale = 0.1 * (1 - i / max(n_steps - 1, 1))  # decreasing noise
        steps.append(base + torch.randn_like(base) * noise_scale)
    return steps


def benchmark_config(
    model_name: str,
    profile: dict,
    config_name: str,
    pipeline_cfg: dict | None,
    bits_k: int = 4,
    bits_v: int = 2,
    seq_len: int = 64,
    n_steps: int = 5,
) -> PipelineBenchResult:
    """Benchmark one pipeline config on one model profile."""

    head_dim = profile["head_dim"]
    n_kv_heads = profile["n_kv_heads"]
    n_layers = min(profile["n_layers"], 4)  # test first 4 layers

    ops = get_backend("torch")

    # Generate synthetic data
    steps_data = _generate_synthetic_kv(head_dim, n_kv_heads, seq_len, n_steps)

    total_mse = 0.0
    total_cos = 0.0
    total_compress_time = 0.0
    total_decompress_time = 0.0
    count = 0

    # Select quantizer class based on config
    use_rotor = pipeline_cfg is not None and pipeline_cfg.get("_quantizer") == "rotor"
    quantizer_cls = RotorQuantizer if use_rotor else PolarQuantizer

    for layer_idx in range(n_layers):
        quantizer = quantizer_cls(head_dim=head_dim, bits=bits_k, seed=42 + layer_idx, ops=ops)
        quantizer_low = quantizer_cls(head_dim=head_dim, bits=bits_v, seed=42 + layer_idx + 10000, ops=ops)

        # Strip internal _quantizer key before passing to pipeline registry
        clean_cfg = {k: v for k, v in pipeline_cfg.items() if not k.startswith("_")} if pipeline_cfg else pipeline_cfg
        config = TurboQuantConfig(bits_k=bits_k, bits_v=bits_v, backend="torch", pipeline=clean_cfg)
        pipe = build_pipeline(config, quantizer=quantizer, quantizer_low=quantizer_low)

        for step_idx, x in enumerate(steps_data):
            t0 = time.perf_counter()
            compressed = pipe.compress(x, layer_idx=layer_idx, step=step_idx,
                                       context={"total_steps": n_steps})
            total_compress_time += time.perf_counter() - t0

            t0 = time.perf_counter()
            recon = pipe.decompress(compressed, layer_idx=layer_idx)
            total_decompress_time += time.perf_counter() - t0

            # Metrics
            x_np = x.numpy().astype(np.float64)
            r_np = recon.detach().numpy().astype(np.float64) if hasattr(recon, 'numpy') else np.array(recon, dtype=np.float64)

            mse = np.mean((x_np - r_np) ** 2)
            signal = np.mean(x_np ** 2)
            nmse = mse / (signal + 1e-10)
            total_mse += nmse

            # Cosine similarity (flatten)
            x_flat = x_np.ravel()
            r_flat = r_np.ravel()
            cos = np.dot(x_flat, r_flat) / (np.linalg.norm(x_flat) * np.linalg.norm(r_flat) + 1e-10)
            total_cos += cos

            count += 1

        pipe.reset()

    avg_nmse = total_mse / max(count, 1)
    avg_cos = total_cos / max(count, 1)
    avg_compress = (total_compress_time / max(count, 1)) * 1000
    avg_decompress = (total_decompress_time / max(count, 1)) * 1000

    orig_bits = 16 * head_dim
    comp_bits = bits_k * head_dim + 16
    ratio = orig_bits / comp_bits

    scorer_name = pipeline_cfg.get("scorer", "none") if pipeline_cfg else "none"
    strategy_name = pipeline_cfg.get("strategy", "none") if pipeline_cfg else "none"
    monitor_name = pipeline_cfg.get("monitor", "none") if pipeline_cfg else "none"

    return PipelineBenchResult(
        model=model_name,
        family=profile["family"],
        config_name=config_name,
        head_dim=head_dim,
        bits_k=bits_k,
        bits_v=bits_v,
        nmse=round(avg_nmse, 8),
        cosine_sim=round(avg_cos, 6),
        compress_ms=round(avg_compress, 3),
        decompress_ms=round(avg_decompress, 3),
        compression_ratio=round(ratio, 2),
        scorer_used=scorer_name,
        strategy_used=strategy_name,
        monitor_used=monitor_name,
        n_tokens=seq_len * n_steps,
        n_layers_tested=n_layers,
    )


def run_all(json_path: str | None = None) -> list[PipelineBenchResult]:
    """Run all pipeline configs on all model profiles."""
    print(f"\n{'='*80}")
    print("tqai v0.4 Pipeline Benchmark Suite")
    print(f"{'='*80}")
    print(f"Models:   {len(MODEL_PROFILES)}")
    print(f"Configs:  {len(PIPELINE_CONFIGS)}")

    available = list_available()
    print(f"Scorers:  {', '.join(available['scorers'])}")
    print(f"Strategies: {', '.join(available['strategies'])}")
    print(f"Monitors: {', '.join(available['monitors'])}")
    print(f"{'='*80}\n")

    results: list[PipelineBenchResult] = []

    for model_name, profile in MODEL_PROFILES.items():
        print(f"\n--- {model_name} (d={profile['head_dim']}, {profile['family']}) ---")
        print(f"{'Config':<22} {'NMSE':>10} {'CosSim':>8} {'Comp ms':>8} {'Decomp ms':>10} {'Ratio':>6}")
        print(f"{'-'*22} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")

        for config_name, pipeline_cfg in PIPELINE_CONFIGS.items():
            try:
                result = benchmark_config(model_name, profile, config_name, pipeline_cfg)
                results.append(result)
                print(
                    f"{config_name:<22} {result.nmse:>10.6f} {result.cosine_sim:>8.4f} "
                    f"{result.compress_ms:>8.3f} {result.decompress_ms:>10.3f} {result.compression_ratio:>6.1f}x"
                )
            except Exception as e:
                print(f"{config_name:<22} ERROR: {e}")

    # Save results
    if json_path:
        out_path = Path(json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "models": list(MODEL_PROFILES.keys()),
            "configs": list(PIPELINE_CONFIGS.keys()),
            "available_plugins": available,
            "results": [asdict(r) for r in results],
        }
        out_path.write_text(json.dumps(data, indent=2))
        print(f"\nResults saved: {out_path}")

    _print_summary(results)
    return results


def _print_summary(results: list[PipelineBenchResult]):
    """Print aggregate summary across all models."""
    print(f"\n{'='*80}")
    print("AGGREGATE SUMMARY (mean across all models)")
    print(f"{'='*80}")
    print(f"{'Config':<22} {'Mean NMSE':>10} {'Mean CosSim':>12} {'Comp ms':>8} {'Decomp ms':>10}")
    print(f"{'-'*22} {'-'*10} {'-'*12} {'-'*8} {'-'*10}")

    from collections import defaultdict
    agg = defaultdict(lambda: {"nmse": [], "cos": [], "comp": [], "decomp": []})
    for r in results:
        agg[r.config_name]["nmse"].append(r.nmse)
        agg[r.config_name]["cos"].append(r.cosine_sim)
        agg[r.config_name]["comp"].append(r.compress_ms)
        agg[r.config_name]["decomp"].append(r.decompress_ms)

    for name, vals in agg.items():
        mn = np.mean(vals["nmse"])
        mc = np.mean(vals["cos"])
        mcomp = np.mean(vals["comp"])
        mdecomp = np.mean(vals["decomp"])
        print(f"{name:<22} {mn:>10.6f} {mc:>12.4f} {mcomp:>8.3f} {mdecomp:>10.3f}")
    print()


def main():
    p = argparse.ArgumentParser(description="tqai pipeline benchmark suite")
    p.add_argument("--json", default="benchmarks/results/pipeline_benchmark.json",
                   help="Output JSON path")
    args = p.parse_args()
    run_all(json_path=args.json)


if __name__ == "__main__":
    main()
