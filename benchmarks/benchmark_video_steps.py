"""Step-sweep benchmark for video diffusion pipelines.

Sweeps `num_inference_steps` across all four presets ({quality, balanced,
fast, draft}) on WAN 2.2 5B and/or LTX-2, measuring wall-clock time and
PSNR vs the highest-step "reference" output. Tests the empirical
hypothesis that modern flow-matching video models handle step reduction
gracefully even without distillation.

Usage:
    python benchmarks/benchmark_video_steps.py
    python benchmarks/benchmark_video_steps.py --models wan
    python benchmarks/benchmark_video_steps.py --models wan,ltx --frames 33
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

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


@dataclass
class StepBenchResult:
    model: str
    preset: str
    num_inference_steps: int
    guidance_scale: float
    generation_time_s: float
    peak_rss_gb: float
    num_frames: int
    output_path: str
    psnr_vs_quality: float  # 0 for the reference itself


PROMPT = "A golden retriever running on a sunny beach with waves, cinematic, 4K"
NEGATIVE = "blurry, low quality, distorted"


def _to_frame_array(output) -> np.ndarray | None:
    """Convert pipeline output to a [T, H, W, C] uint8 numpy array."""
    try:
        frames = output.frames[0] if hasattr(output, "frames") else output
        return np.stack([np.array(f) for f in frames])
    except Exception:
        return None


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR between two uint8 image stacks."""
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    a_f = a.astype(np.float64) / 255.0
    b_f = b.astype(np.float64) / 255.0
    mse = float(np.mean((a_f - b_f) ** 2))
    if mse < 1e-12:
        return float("inf")
    return float(10 * np.log10(1.0 / mse))


def _save_video(output, path: Path):
    try:
        from diffusers.utils import export_to_video
        frames = output.frames[0] if hasattr(output, "frames") else output
        export_to_video(frames, str(path), fps=24)
        return True
    except Exception as e:
        print(f"    Export failed: {e}")
        return False


def _run_wan_sweep(num_frames: int, output_dir: Path, seed: int) -> list[StepBenchResult]:
    import torch
    from diffusers import WanPipeline

    from tqai.dit import list_presets, optimize_vae_memory

    print("\n=== WAN 2.2 5B step sweep ===")
    print("Loading Wan-AI/Wan2.2-TI2V-5B-Diffusers...")
    pipe = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers", torch_dtype=torch.bfloat16,
    )
    pipe.to("mps")
    optimize_vae_memory(pipe)
    print(f"Loaded. RSS: {_rss_gb():.1f} GB\n")

    results: list[StepBenchResult] = []
    reference_frames: np.ndarray | None = None
    presets = list_presets("WanPipeline")["WanPipeline"]

    # Quality (reference) first, then descending step counts
    order = ["quality", "balanced", "fast", "draft"]
    for mode in order:
        preset = presets[mode]
        print(f"--- WAN {mode} ({preset.num_inference_steps} steps, guidance {preset.guidance_scale}) ---")

        gen = torch.Generator(device="mps").manual_seed(seed)
        t0 = time.perf_counter()
        out = pipe(
            PROMPT,
            negative_prompt=NEGATIVE,
            num_frames=num_frames,
            height=480,
            width=832,
            num_inference_steps=preset.num_inference_steps,
            guidance_scale=preset.guidance_scale,
            generator=gen,
        )
        elapsed = time.perf_counter() - t0
        peak = _rss_gb()

        out_path = output_dir / f"wan_{mode}.mp4"
        _save_video(out, out_path)

        frames = _to_frame_array(out)
        if mode == "quality":
            reference_frames = frames
            psnr = 0.0
        else:
            psnr = _psnr(reference_frames, frames) if reference_frames is not None else 0.0

        result = StepBenchResult(
            model="wan",
            preset=mode,
            num_inference_steps=preset.num_inference_steps,
            guidance_scale=preset.guidance_scale,
            generation_time_s=round(elapsed, 1),
            peak_rss_gb=round(peak, 1),
            num_frames=num_frames,
            output_path=str(out_path),
            psnr_vs_quality=round(psnr, 1) if psnr != float("inf") else 999.9,
        )
        results.append(result)
        print(
            f"  time={elapsed:.1f}s  peak={peak:.1f}GB  "
            f"PSNR_vs_quality={result.psnr_vs_quality}dB\n"
        )

        del out
        gc.collect()
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    del pipe
    gc.collect()
    return results


def _run_ltx_sweep(num_frames: int, output_dir: Path, seed: int) -> list[StepBenchResult]:
    import torch
    from diffusers import LTX2Pipeline

    from tqai.dit import (
        list_presets,
        optimize_vae_memory,
        patch_mps_compatibility,
    )

    print("\n=== LTX-2 step sweep ===")
    print("Loading Lightricks/LTX-2...")
    pipe = LTX2Pipeline.from_pretrained(
        "Lightricks/LTX-2", torch_dtype=torch.bfloat16,
    )
    pipe.to("mps")
    patch_mps_compatibility(pipe)
    optimize_vae_memory(pipe, enable_tiling=False, enable_slicing=True)
    print(f"Loaded. RSS: {_rss_gb():.1f} GB\n")

    results: list[StepBenchResult] = []
    reference_frames: np.ndarray | None = None
    presets = list_presets("LTX2Pipeline")["LTX2Pipeline"]

    order = ["quality", "balanced", "fast", "draft"]
    for mode in order:
        preset = presets[mode]
        print(f"--- LTX {mode} ({preset.num_inference_steps} steps, guidance {preset.guidance_scale}) ---")

        gen = torch.Generator(device="mps").manual_seed(seed)
        t0 = time.perf_counter()
        out = pipe(
            PROMPT,
            negative_prompt=NEGATIVE,
            num_frames=num_frames,
            height=480,
            width=832,
            num_inference_steps=preset.num_inference_steps,
            guidance_scale=preset.guidance_scale,
            generator=gen,
        )
        elapsed = time.perf_counter() - t0
        peak = _rss_gb()

        out_path = output_dir / f"ltx_{mode}.mp4"
        _save_video(out, out_path)

        frames = _to_frame_array(out)
        if mode == "quality":
            reference_frames = frames
            psnr = 0.0
        else:
            psnr = _psnr(reference_frames, frames) if reference_frames is not None else 0.0

        result = StepBenchResult(
            model="ltx",
            preset=mode,
            num_inference_steps=preset.num_inference_steps,
            guidance_scale=preset.guidance_scale,
            generation_time_s=round(elapsed, 1),
            peak_rss_gb=round(peak, 1),
            num_frames=num_frames,
            output_path=str(out_path),
            psnr_vs_quality=round(psnr, 1) if psnr != float("inf") else 999.9,
        )
        results.append(result)
        print(
            f"  time={elapsed:.1f}s  peak={peak:.1f}GB  "
            f"PSNR_vs_quality={result.psnr_vs_quality}dB\n"
        )

        del out
        gc.collect()
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    del pipe
    gc.collect()
    return results


def _print_summary(results: list[StepBenchResult]):
    print("\n" + "=" * 80)
    print("STEP SWEEP SUMMARY")
    print("=" * 80)
    print(
        f"{'Model':<6} {'Preset':<10} {'Steps':>6} {'Time':>9} {'Speedup':>9} "
        f"{'PSNR':>10}"
    )
    print(f"{'-' * 6} {'-' * 10} {'-' * 6} {'-' * 9} {'-' * 9} {'-' * 10}")

    for model in sorted(set(r.model for r in results)):
        model_results = [r for r in results if r.model == model]
        ref = next((r for r in model_results if r.preset == "quality"), None)
        for r in model_results:
            speedup = (ref.generation_time_s / r.generation_time_s) if ref and r.generation_time_s > 0 else 0
            psnr_str = (
                f"{r.psnr_vs_quality:.1f}dB" if r.psnr_vs_quality < 999 else "(ref)"
            )
            print(
                f"{r.model:<6} {r.preset:<10} {r.num_inference_steps:>6} "
                f"{r.generation_time_s:>7.1f}s {speedup:>8.2f}x {psnr_str:>10}"
            )
        print()


def main():
    p = argparse.ArgumentParser(description="Video pipeline step-sweep benchmark")
    p.add_argument("--models", default="wan", help="Comma-separated: wan,ltx")
    p.add_argument("--frames", type=int, default=33)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir", default="benchmarks/results/videos_step_sweep",
    )
    p.add_argument(
        "--json", default="benchmarks/results/video_step_sweep.json",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",")]
    all_results: list[StepBenchResult] = []

    for model in models:
        if model == "wan":
            all_results.extend(_run_wan_sweep(args.frames, out_dir, args.seed))
        elif model == "ltx":
            all_results.extend(_run_ltx_sweep(args.frames, out_dir, args.seed))
        else:
            print(f"Unknown model: {model}, skipping")

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "frames": args.frames,
                "seed": args.seed,
                "results": [asdict(r) for r in all_results],
            },
            indent=2,
        )
    )
    print(f"\nResults saved: {json_path}")

    _print_summary(all_results)


if __name__ == "__main__":
    main()
