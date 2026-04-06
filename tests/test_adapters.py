"""Tests for model adapter modules."""

from __future__ import annotations

import pytest


class TestLLMAdapter:
    def _make_adapter(self):
        from tqai.adapters.llm import LLMAdapter
        return LLMAdapter()

    def test_detect_hf_model(self):
        class FakeConfig:
            model_type = "llama"
        class FakeModel:
            config = FakeConfig()

        adapter = self._make_adapter()
        assert adapter.detect(FakeModel()) is True

    def test_detect_mlx_model(self):
        class FakeArgs:
            pass
        class FakeModel:
            layers = [1, 2, 3]
            args = FakeArgs()

        adapter = self._make_adapter()
        assert adapter.detect(FakeModel()) is True

    def test_detect_non_model(self):
        adapter = self._make_adapter()
        assert adapter.detect("not_a_model") is False

    def test_get_head_info(self):
        class FakeConfig:
            model_type = "llama"
            hidden_size = 4096
            num_attention_heads = 32
            num_key_value_heads = 8
            num_hidden_layers = 32
        class FakeModel:
            config = FakeConfig()

        adapter = self._make_adapter()
        info = adapter.get_head_info(FakeModel())
        assert info["head_dim"] == 128
        assert info["n_heads"] == 32
        assert info["n_kv_heads"] == 8
        assert info["n_layers"] == 32

    def test_registration(self):
        import tqai.adapters  # noqa: F401
        from tqai.pipeline.registry import list_available
        available = list_available()
        assert "llm" in available["adapters"]
        assert "dit" in available["adapters"]
        assert "wan" in available["adapters"]


class TestDiTAdapter:
    def _make_adapter(self):
        from tqai.adapters.dit import DiTAdapter
        return DiTAdapter()

    def test_detect_dit_pipeline(self):
        class FakeTransformer:
            transformer_blocks = [1, 2]
        class FakePipeline:
            transformer = FakeTransformer()

        adapter = self._make_adapter()
        assert adapter.detect(FakePipeline()) is True

    def test_detect_non_dit(self):
        adapter = self._make_adapter()
        assert adapter.detect("not_a_pipeline") is False


class TestWANAdapter:
    def _make_adapter(self):
        from tqai.adapters.wan import WANAdapter
        return WANAdapter()

    def test_detect_wan_model(self):
        class WanTransformer3DModel:
            transformer_blocks = []
        class FakePipeline:
            transformer = WanTransformer3DModel()

        adapter = self._make_adapter()
        assert adapter.detect(FakePipeline()) is True

    def test_detect_non_wan(self):
        adapter = self._make_adapter()
        assert adapter.detect("not_wan") is False


class TestAdapterAutoDetect:
    def test_auto_detect_llm(self):
        import tqai.adapters  # noqa: F401
        from tqai.pipeline.registry import get_adapter

        class FakeConfig:
            model_type = "qwen2"
        class FakeModel:
            config = FakeConfig()

        adapter = get_adapter(None, FakeModel())
        assert adapter.name == "llm"

    def test_auto_detect_dit(self):
        import tqai.adapters  # noqa: F401
        from tqai.pipeline.registry import get_adapter

        class FakeTransformer:
            transformer_blocks = [1]
        class FakePipeline:
            transformer = FakeTransformer()

        adapter = get_adapter(None, FakePipeline())
        # dit or wan could match; both have transformer_blocks
        assert adapter.name in ("dit", "wan")

    def test_unknown_model_raises(self):
        import tqai.adapters  # noqa: F401
        from tqai.pipeline.registry import get_adapter

        with pytest.raises(ValueError, match="No adapter found"):
            get_adapter(None, 42)
