"""End-to-end tests with larger models (5B, 8B, 14B).

Tests both MLX and HuggingFace backends with bf16, Q8, and Q4 quantized models.
Run with: python -m pytest tests/test_e2e_large_models.py -v -s --timeout=600
"""

from __future__ import annotations

import pytest

PROMPT = "Explain quantum entanglement in two sentences."


# ─── MLX tests (Apple Silicon, fast) ───

class TestMLXLargeModels:
    """Test tqai with larger MLX models."""

    MODELS = {
        # (model_id, label, expected_head_dim)
        "qwen-3b-bf16": ("mlx-community/Qwen2.5-3B-Instruct-bf16", "3B bf16"),
        "qwen-3b-q8": ("mlx-community/Qwen2.5-3B-Instruct-8bit", "3B Q8"),
        "qwen-3b-q4": ("mlx-community/Qwen2.5-3B-Instruct-4bit", "3B Q4"),
        "llama-8b-q4": ("mlx-community/Llama-3.1-8B-Instruct-4bit", "8B Q4"),
        "llama-8b-q8": ("mlx-community/Llama-3.1-8B-Instruct-8bit", "8B Q8"),
        "qwen-14b-q4": ("mlx-community/Qwen2.5-14B-Instruct-4bit", "14B Q4"),
    }

    @pytest.fixture(
        scope="class",
        params=list(MODELS.keys()),
    )
    def mlx_model(self, request):
        import mlx_lm

        model_id, label = self.MODELS[request.param]
        print(f"\n{'='*60}")
        print(f"Loading: {label} ({model_id})")
        print(f"{'='*60}")
        model, tokenizer = mlx_lm.load(model_id)
        return model, tokenizer, label, request.param

    def test_baseline(self, mlx_model):
        import mlx_lm

        model, tokenizer, label, _ = mlx_model
        result = mlx_lm.generate(model, tokenizer, prompt=PROMPT, max_tokens=60)
        print(f"\n[{label} baseline]\n{result}")
        assert len(result) > 10

    def test_tqai_k4v2(self, mlx_model):
        import mlx_lm

        import tqai

        model, tokenizer, label, _ = mlx_model
        tqai.patch(model, bits_k=4, bits_v=2, backend="mlx")
        try:
            result = mlx_lm.generate(model, tokenizer, prompt=PROMPT, max_tokens=60)
            print(f"\n[{label} + tqai K4/V2]\n{result}")
            assert len(result) > 5
        finally:
            tqai.unpatch(model)

    def test_tqai_k3v2(self, mlx_model):
        import mlx_lm

        import tqai

        model, tokenizer, label, _ = mlx_model
        tqai.patch(model, bits_k=3, bits_v=2, backend="mlx")
        try:
            result = mlx_lm.generate(model, tokenizer, prompt=PROMPT, max_tokens=60)
            print(f"\n[{label} + tqai K3/V2]\n{result}")
            assert len(result) > 5
        finally:
            tqai.unpatch(model)


# ─── HuggingFace / PyTorch tests ───

class TestHFLargeModels:
    """Test tqai with larger HuggingFace models (bf16 on CPU — slow but validates)."""

    MODELS = {
        "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    }

    @pytest.fixture(scope="class", params=list(MODELS.keys()))
    def hf_model(self, request):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.MODELS[request.param]
        print(f"\n{'='*60}")
        print(f"Loading HF: {model_id}")
        print(f"{'='*60}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        return model, tokenizer, request.param

    def test_baseline(self, hf_model):
        model, tokenizer, label = hf_model
        inputs = tokenizer(PROMPT, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n[HF {label} baseline]\n{result}")
        assert len(result) > len(PROMPT)

    def test_tqai_k4v2(self, hf_model):
        import tqai

        model, tokenizer, label = hf_model
        cache = tqai.patch(model, bits_k=4, bits_v=2, backend="torch")
        inputs = tokenizer(PROMPT, return_tensors="pt")
        output = model.generate(**inputs, past_key_values=cache, max_new_tokens=60, do_sample=False)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n[HF {label} + tqai K4/V2]\n{result}")
        print(f"  Cache seq length: {cache.get_seq_length()}")
        assert len(result) > len(PROMPT)
