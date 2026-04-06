"""Tests for tqai.dit.presets — verified video pipeline step counts."""

from __future__ import annotations

import pytest

from tqai.dit.presets import (
    PRESETS,
    VideoPreset,
    get_video_preset,
    list_presets,
)


class TestVideoPreset:
    def test_as_kwargs(self):
        p = VideoPreset("test", num_inference_steps=10, guidance_scale=4.5,
                        description="test")
        kw = p.as_kwargs()
        assert kw == {"num_inference_steps": 10, "guidance_scale": 4.5}

    def test_dataclass_fields(self):
        p = VideoPreset("balanced", 15, 5.0, "half steps")
        assert p.name == "balanced"
        assert p.num_inference_steps == 15
        assert p.guidance_scale == 5.0
        assert p.description == "half steps"


class TestGetVideoPreset:
    def test_wan_quality(self):
        p = get_video_preset("WanPipeline", mode="quality")
        assert p.name == "quality"
        assert p.num_inference_steps == 25

    def test_wan_balanced(self):
        p = get_video_preset("WanPipeline", mode="balanced")
        assert p.num_inference_steps == 15

    def test_wan_fast(self):
        p = get_video_preset("WanPipeline", mode="fast")
        assert p.num_inference_steps == 8

    def test_wan_draft(self):
        p = get_video_preset("WanPipeline", mode="draft")
        assert p.num_inference_steps == 4

    def test_ltx2_quality(self):
        p = get_video_preset("LTX2Pipeline", mode="quality")
        assert p.num_inference_steps == 30

    def test_ltx2_draft(self):
        p = get_video_preset("LTX2Pipeline", mode="draft")
        assert p.num_inference_steps == 4

    def test_unknown_pipeline_raises(self):
        with pytest.raises(ValueError, match="No presets registered"):
            get_video_preset("NonexistentPipeline", mode="balanced")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            get_video_preset("WanPipeline", mode="ultra")

    def test_pipeline_object_lookup(self):
        # Simulate a pipeline by creating a stub class
        class WanPipeline:
            pass
        p = get_video_preset(WanPipeline(), mode="fast")
        assert p.num_inference_steps == 8


class TestListPresets:
    def test_list_all(self):
        all_presets = list_presets()
        assert "WanPipeline" in all_presets
        assert "LTX2Pipeline" in all_presets
        assert "balanced" in all_presets["WanPipeline"]

    def test_list_filtered(self):
        filtered = list_presets("WanPipeline")
        assert set(filtered.keys()) == {"WanPipeline"}
        assert "fast" in filtered["WanPipeline"]

    def test_list_unknown(self):
        result = list_presets("NonexistentPipeline")
        assert result == {}


class TestStepCountInvariants:
    """Sanity checks that the preset table is internally consistent."""

    @pytest.mark.parametrize("pipeline", list(PRESETS.keys()))
    def test_quality_is_slowest(self, pipeline):
        modes = PRESETS[pipeline]
        assert (
            modes["quality"].num_inference_steps
            >= modes["balanced"].num_inference_steps
            >= modes["fast"].num_inference_steps
            >= modes["draft"].num_inference_steps
        )

    @pytest.mark.parametrize("pipeline", list(PRESETS.keys()))
    def test_all_modes_present(self, pipeline):
        modes = PRESETS[pipeline]
        assert set(modes.keys()) == {"quality", "balanced", "fast", "draft"}

    @pytest.mark.parametrize("pipeline", list(PRESETS.keys()))
    def test_all_steps_positive(self, pipeline):
        modes = PRESETS[pipeline]
        for preset in modes.values():
            assert preset.num_inference_steps >= 1
            assert preset.guidance_scale > 0
