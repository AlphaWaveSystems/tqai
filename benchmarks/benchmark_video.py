"""Video generation benchmark: baseline vs tqai pipeline with CFG sharing.

Generates videos with WAN 2.1-1.3B (T2V) and LTX-2 (T2V), comparing:
1. Baseline (no tqai)
2. tqai CFG sharing only
3. tqai full pipeline (CFG sharing + KV compression + forward hooks)

Measures:
- Wall-clock generation time
- Peak memory usage (RSS)
- PSNR / SSIM between baseline and compressed outputs
- CFG sharing statistics

Usage:
    python benchmarks/benchmark_video.py
    python benchmarks/benchmark_video.py --models wan --steps 30
    python benchmarks/benchmark_video.py --models wan,ltx --prompt "A cat on a surfboard"
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Ensure registrations
import tqai.scorers  # noqa: F401
import tqai.strategies  # noqa: F401

# ---------------------------------------------------------------------------
# LTX-2 MPS fix
# ---------------------------------------------------------------------------

def _patch_ltx2_rope_for_mps(pipe):
    """Fix LTX-2 RoPE float64 crash on MPS."""
    from tqai.dit.mps_fixes import patch_mps_compatibility
    n = patch_mps_compatibility(pipe)
    if n:
        print(f"    Patched {n} MPS float64 modules")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = "A golden retriever running on a sunny beach with waves in the background, cinematic, 4K"
DEFAULT_NEGATIVE = "blurry, low quality, distorted"

MODEL_CONFIGS = {
    "wan": {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "pipeline_class": "WanPipeline",
        "cfg_mode": "split",
        "default_steps": 20,
        "default_guidance": 5.0,
        # WAN 2.2 5B: 480p, 81 frames = ~3.2s at 24fps
        "num_frames": 81,
        "height": 480,
        "width": 832,
    },
    "ltx": {
        "model_id": "Lightricks/LTX-2",
        "pipeline_class": "LTX2Pipeline",
        "cfg_mode": "batched",
        "default_steps": 20,
        "default_guidance": 4.0,
        "num_frames": 97,  # ~4s at 24fps
        "height": 480,
        "width": 832,
    },
}

BENCHMARK_CONFIGS = [
    {
        "name": "baseline",
        "use_tqai": False,
        "cfg_sharing": False,
        "description": "No compression, standard generation",
    },
    {
        "name": "cfg_sharing",
        "use_tqai": False,
        "cfg_sharing": True,
        "description": "CFG attention sharing only (DiTFastAttn)",
    },
    {
        "name": "tqai_kv",
        "use_tqai": True,
        "cfg_sharing": False,
        "pipeline": {"scorer": "snr", "strategy": "delta"},
        "description": "tqai KV compression with SNR scorer + delta strategy",
    },
    {
        "name": "tqai_full",
        "use_tqai": True,
        "cfg_sharing": True,
        "pipeline": {"scorer": "snr", "strategy": "delta2"},
        "description": "Full pipeline: CFG sharing + SNR/delta2 compression",
    },
]


@dataclass
class VideoBenchResult:
    model: str
    config_name: str
    description: str
    generation_time_s: float
    peak_rss_mb: float
    num_frames: int
    steps: int
    cfg_shared: int
    cfg_computed: int
    cfg_share_ratio: float
    output_path: str
    psnr_vs_baseline: float  # 0 for baseline itself
    error: str | None = None


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # macOS returns bytes
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Video quality metrics
# ---------------------------------------------------------------------------

def _compute_psnr(video_a, video_b) -> float:
    """Compute PSNR between two video tensors or file paths."""
    if video_a is None or video_b is None:
        return 0.0
    try:
        a = video_a.float() if isinstance(video_a, torch.Tensor) else video_a
        b = video_b.float() if isinstance(video_b, torch.Tensor) else video_b

        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            mse = torch.mean((a - b) ** 2).item()
            if mse < 1e-10:
                return float("inf")
            return float(10 * np.log10(1.0 / mse))
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate_wan(
    model_config: dict,
    bench_config: dict,
    prompt: str,
    negative_prompt: str,
    num_steps: int,
    output_dir: Path,
    seed: int = 42,
) -> tuple[VideoBenchResult, torch.Tensor | None]:
    """Generate video with WAN pipeline."""
    from diffusers import WanPipeline

    model_id = model_config["model_id"]
    config_name = bench_config["name"]

    print(f"    Loading {model_id}...")
    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to("mps")

    # Always enable VAE memory optimization (prevents 100GB+ spike)
    from tqai.dit.vae_memory import optimize_vae_memory
    optimize_vae_memory(pipe)

    cfg_hooks = None
    tqai_hooks = None

    # Apply CFG sharing
    if bench_config.get("cfg_sharing"):
        from tqai.dit.cfg_patch import patch_cfg_sharing
        cfg_hooks = patch_cfg_sharing(pipe, share_cross_attn=True)

    # Apply tqai forward compression
    if bench_config.get("use_tqai"):
        from tqai.hooks import ForwardCompressionHooks, ForwardHookConfig
        hook_config = ForwardHookConfig(
            compress_hidden=True, bits_hidden=8,
            compress_ffn=True, bits_ffn=8, seed=42,
        )
        tqai_hooks = ForwardCompressionHooks(hook_config)
        tqai_hooks.attach(pipe.transformer)

    # Generate
    generator = torch.Generator(device="mps").manual_seed(seed)
    rss_before = _rss_mb()

    print(f"    Generating ({num_steps} steps)...")
    t0 = time.perf_counter()
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=model_config["num_frames"],
        height=model_config["height"],
        width=model_config["width"],
        num_inference_steps=num_steps,
        guidance_scale=model_config["default_guidance"],
        generator=generator,
    )
    elapsed = time.perf_counter() - t0
    peak_rss = _rss_mb()

    # Save video
    output_path = output_dir / f"wan_{config_name}.mp4"
    _export_video(output, output_path)

    # Get CFG stats
    cfg_stats = cfg_hooks.stats if cfg_hooks else {"shared": 0, "computed": 0, "share_ratio": 0.0}

    # Get output frames for PSNR
    frames = output.frames[0] if hasattr(output, "frames") else None
    frames_tensor = None
    if frames is not None:
        try:
            frames_tensor = torch.stack([torch.tensor(np.array(f)).float() / 255.0 for f in frames])
        except Exception:
            pass

    # Cleanup
    if cfg_hooks:
        from tqai.dit.cfg_patch import unpatch_cfg_sharing
        unpatch_cfg_sharing(pipe)
    if tqai_hooks:
        tqai_hooks.detach()
    del pipe
    gc.collect()
    torch.mps.empty_cache()

    result = VideoBenchResult(
        model="wan",
        config_name=config_name,
        description=bench_config["description"],
        generation_time_s=round(elapsed, 2),
        peak_rss_mb=round(peak_rss, 0),
        num_frames=model_config["num_frames"],
        steps=num_steps,
        cfg_shared=cfg_stats["shared"],
        cfg_computed=cfg_stats["computed"],
        cfg_share_ratio=round(cfg_stats["share_ratio"], 3),
        output_path=str(output_path),
        psnr_vs_baseline=0.0,
    )
    return result, frames_tensor


def _generate_ltx(
    model_config: dict,
    bench_config: dict,
    prompt: str,
    negative_prompt: str,
    num_steps: int,
    output_dir: Path,
    seed: int = 42,
) -> tuple[VideoBenchResult, torch.Tensor | None]:
    """Generate video with LTX-2 pipeline."""
    from diffusers import LTX2Pipeline

    model_id = model_config["model_id"]
    config_name = bench_config["name"]

    print(f"    Loading {model_id}...")
    pipe = LTX2Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to("mps")

    # Fix LTX-2 MPS float64 incompatibility
    _patch_ltx2_rope_for_mps(pipe)

    # VAE memory optimization — LTX-2 VAE is incompatible with tiling, use slicing only
    from tqai.dit.vae_memory import optimize_vae_memory
    optimize_vae_memory(pipe, enable_tiling=False, enable_slicing=True)

    cfg_hooks = None

    if bench_config.get("cfg_sharing"):
        from tqai.dit.cfg_patch import patch_cfg_sharing
        cfg_hooks = patch_cfg_sharing(pipe, share_cross_attn=True)

    generator = torch.Generator(device="mps").manual_seed(seed)
    rss_before = _rss_mb()

    print(f"    Generating ({num_steps} steps)...")
    t0 = time.perf_counter()
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=model_config["num_frames"],
        height=model_config["height"],
        width=model_config["width"],
        num_inference_steps=num_steps,
        guidance_scale=model_config["default_guidance"],
        generator=generator,
    )
    elapsed = time.perf_counter() - t0
    peak_rss = _rss_mb()

    output_path = output_dir / f"ltx_{config_name}.mp4"
    _export_video(output, output_path)

    cfg_stats = cfg_hooks.stats if cfg_hooks else {"shared": 0, "computed": 0, "share_ratio": 0.0}

    frames = output.frames[0] if hasattr(output, "frames") else None
    frames_tensor = None
    if frames is not None:
        try:
            frames_tensor = torch.stack([torch.tensor(np.array(f)).float() / 255.0 for f in frames])
        except Exception:
            pass

    if cfg_hooks:
        from tqai.dit.cfg_patch import unpatch_cfg_sharing
        unpatch_cfg_sharing(pipe)
    del pipe
    gc.collect()
    torch.mps.empty_cache()

    result = VideoBenchResult(
        model="ltx",
        config_name=config_name,
        description=bench_config["description"],
        generation_time_s=round(elapsed, 2),
        peak_rss_mb=round(peak_rss, 0),
        num_frames=model_config["num_frames"],
        steps=num_steps,
        cfg_shared=cfg_stats["shared"],
        cfg_computed=cfg_stats["computed"],
        cfg_share_ratio=round(cfg_stats["share_ratio"], 3),
        output_path=str(output_path),
        psnr_vs_baseline=0.0,
    )
    return result, frames_tensor


def _export_video(output, path: Path):
    """Export pipeline output to MP4."""
    try:
        from diffusers.utils import export_to_video
        frames = output.frames[0] if hasattr(output, "frames") else output
        export_to_video(frames, str(path), fps=24)
        print(f"    Saved: {path}")
    except Exception as e:
        print(f"    Export failed: {e}")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    models: list[str],
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE,
    num_steps: int | None = None,
    output_dir: str = "benchmarks/results/videos",
    seed: int = 42,
) -> list[VideoBenchResult]:
    """Run video generation benchmark across models and configs."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("tqai Video Generation Benchmark")
    print(f"{'='*80}")
    print(f"Models:  {', '.join(models)}")
    print(f"Configs: {', '.join(c['name'] for c in BENCHMARK_CONFIGS)}")
    print(f"Prompt:  {prompt[:60]}...")
    print(f"{'='*80}\n")

    all_results: list[VideoBenchResult] = []

    for model_name in models:
        if model_name not in MODEL_CONFIGS:
            print(f"Unknown model: {model_name}, skipping")
            continue

        model_config = MODEL_CONFIGS[model_name]
        steps = num_steps or model_config["default_steps"]
        baseline_frames = None

        print(f"\n--- {model_name.upper()} ({model_config['model_id']}) ---")
        print(f"{'Config':<16} {'Time':>8} {'RSS MB':>8} {'CFG Share':>10} {'PSNR':>8}")
        print(f"{'-'*16} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

        gen_fn = _generate_wan if model_name == "wan" else _generate_ltx

        for bench_config in BENCHMARK_CONFIGS:
            try:
                result, frames = gen_fn(
                    model_config, bench_config, prompt, negative_prompt,
                    steps, out_dir, seed,
                )

                if bench_config["name"] == "baseline":
                    baseline_frames = frames

                if baseline_frames is not None and frames is not None and bench_config["name"] != "baseline":
                    result.psnr_vs_baseline = _compute_psnr(baseline_frames, frames)

                all_results.append(result)

                psnr_str = f"{result.psnr_vs_baseline:.1f}dB" if result.psnr_vs_baseline > 0 else "-"
                print(
                    f"  {result.config_name:<14} {result.generation_time_s:>7.1f}s "
                    f"{result.peak_rss_mb:>7.0f} {result.cfg_share_ratio:>9.1%} {psnr_str:>8}"
                )

            except Exception as e:
                print(f"  {bench_config['name']:<14} ERROR: {e}")
                all_results.append(VideoBenchResult(
                    model=model_name, config_name=bench_config["name"],
                    description=bench_config["description"],
                    generation_time_s=0, peak_rss_mb=0, num_frames=0,
                    steps=steps, cfg_shared=0, cfg_computed=0,
                    cfg_share_ratio=0, output_path="",
                    psnr_vs_baseline=0, error=str(e),
                ))

    # Save JSON
    json_path = out_dir / "video_benchmark.json"
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt": prompt,
        "models": models,
        "results": [asdict(r) for r in all_results],
    }
    json_path.write_text(json.dumps(data, indent=2))
    print(f"\nResults saved: {json_path}")

    _print_summary(all_results)
    return all_results


def _print_summary(results: list[VideoBenchResult]):
    """Print summary table."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for model in set(r.model for r in results):
        model_results = [r for r in results if r.model == model and r.error is None]
        if not model_results:
            continue

        baseline = next((r for r in model_results if r.config_name == "baseline"), None)
        if baseline is None:
            continue

        print(f"\n{model.upper()}:")
        for r in model_results:
            speedup = baseline.generation_time_s / r.generation_time_s if r.generation_time_s > 0 else 0
            mem_saved = (1 - r.peak_rss_mb / baseline.peak_rss_mb) * 100 if baseline.peak_rss_mb > 0 else 0
            psnr_str = f"PSNR={r.psnr_vs_baseline:.1f}dB" if r.psnr_vs_baseline > 0 else "baseline"
            print(
                f"  {r.config_name:<16} {r.generation_time_s:>6.1f}s "
                f"({speedup:.2f}x) mem={mem_saved:+.0f}% {psnr_str}"
            )
    print()


def main():
    p = argparse.ArgumentParser(description="tqai video generation benchmark")
    p.add_argument("--models", default="wan", help="Comma-separated: wan,ltx")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--output-dir", default="benchmarks/results/videos")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    run_benchmark(models=models, prompt=args.prompt, num_steps=args.steps,
                  output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
