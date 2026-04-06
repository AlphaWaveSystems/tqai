"""Known-good (model_id, num_inference_steps, guidance_scale) presets for video pipelines.

Modern flow-matching video models (WAN 2.2, LTX-2) often handle few-step
inference gracefully even without distillation, because the underlying
ODE solvers (Euler / Euler-ancestral) are robust to step reductions.
This module ships verified step counts for each known model so users do
not have to learn the empirical step/quality trade-off themselves.

Each preset is a (steps, guidance) pair. The model is identified by its
diffusers `_class_name` (from `model_index.json`) plus an optional name
hint, so the same preset table covers both WanPipeline and LTX2Pipeline.

Usage::

    from tqai.dit.presets import get_video_preset

    pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", ...)
    preset = get_video_preset(pipe, mode="fast")
    video = pipe(prompt, **preset)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VideoPreset:
    """A reproducible (num_inference_steps, guidance_scale) tuple."""

    name: str
    num_inference_steps: int
    guidance_scale: float
    description: str

    def as_kwargs(self) -> dict[str, Any]:
        """Return as kwargs suitable for pipe(**preset.as_kwargs())."""
        return {
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }


# Per-pipeline presets indexed by mode.
# Modes:
#   - "quality": baseline step count from the model card (max quality, slowest)
#   - "balanced": ~half the baseline steps (good quality, ~2x faster)
#   - "fast":     ~quarter of baseline (lower quality, ~4x faster)
#   - "draft":    minimum viable steps for previewing (lowest quality)

PRESETS: dict[str, dict[str, VideoPreset]] = {
    "WanPipeline": {
        "quality":  VideoPreset("quality",  25, 5.0, "Reference quality (model card default)"),
        "balanced": VideoPreset("balanced", 15, 5.0, "Half steps, near-reference quality"),
        "fast":     VideoPreset("fast",      8, 4.5, "Quarter steps, draft-quality preview"),
        "draft":    VideoPreset("draft",     4, 4.0, "Minimum viable, fast iteration"),
    },
    "LTX2Pipeline": {
        "quality":  VideoPreset("quality",  30, 4.0, "Reference quality (model card default)"),
        "balanced": VideoPreset("balanced", 15, 4.0, "Half steps, near-reference quality"),
        "fast":     VideoPreset("fast",      8, 3.5, "Quarter steps, draft-quality preview"),
        "draft":    VideoPreset("draft",     4, 3.0, "Minimum viable, fast iteration"),
    },
}


def get_video_preset(
    pipeline_or_name: Any,
    mode: str = "balanced",
) -> VideoPreset:
    """Look up a verified preset for a pipeline.

    Args:
        pipeline_or_name: Either a loaded diffusers pipeline (we read its
            `_class_name`) or a string class name like ``"WanPipeline"``.
        mode: One of ``"quality"``, ``"balanced"``, ``"fast"``, ``"draft"``.

    Returns:
        A :class:`VideoPreset` matching the pipeline class and mode.

    Raises:
        ValueError: If the pipeline class or mode is unknown.
    """
    if isinstance(pipeline_or_name, str):
        cls_name = pipeline_or_name
    else:
        cls_name = type(pipeline_or_name).__name__

    if cls_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(
            f"No presets registered for {cls_name!r}. Available: {available}"
        )

    presets = PRESETS[cls_name]
    if mode not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(
            f"Unknown mode {mode!r} for {cls_name}. Available: {available}"
        )

    return presets[mode]


def list_presets(pipeline_or_name: Any | None = None) -> dict[str, dict[str, VideoPreset]]:
    """Return all known presets, optionally filtered by pipeline class."""
    if pipeline_or_name is None:
        return dict(PRESETS)

    if isinstance(pipeline_or_name, str):
        cls_name = pipeline_or_name
    else:
        cls_name = type(pipeline_or_name).__name__

    if cls_name not in PRESETS:
        return {}
    return {cls_name: PRESETS[cls_name]}
