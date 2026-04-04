"""End-to-end tests with real models.

Downloads small models and tests inference with and without TurboQuant.
Run with: python -m pytest tests/test_e2e_models.py -v -s
"""

from __future__ import annotations

import pytest

# ─── HuggingFace / PyTorch tests ───

class TestHuggingFace:
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    PROMPT = "What is 2+2? Answer in one word:"

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.bfloat16
        )
        return model, tokenizer

    def _generate(self, model, tokenizer, past_key_values=None, max_new_tokens=30):
        inputs = tokenizer(self.PROMPT, return_tensors="pt")
        output = model.generate(
            **inputs,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)

    def test_baseline_bf16(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        result = self._generate(model, tokenizer)
        print(f"\n[HF bf16 baseline] {result}")
        assert len(result) > len(self.PROMPT)

    def test_with_tqai_k4v2(self, model_and_tokenizer):
        import tqai

        model, tokenizer = model_and_tokenizer
        cache = tqai.patch(model, bits_k=4, bits_v=2, backend="torch")
        result = self._generate(model, tokenizer, past_key_values=cache)
        print(f"\n[HF tqai K4/V2] {result}")
        assert len(result) > len(self.PROMPT)
        print(f"  Cache seq length: {cache.get_seq_length()}")

    def test_with_tqai_k3v2(self, model_and_tokenizer):
        import tqai

        model, tokenizer = model_and_tokenizer
        cache = tqai.patch(model, bits_k=3, bits_v=2, backend="torch")
        result = self._generate(model, tokenizer, past_key_values=cache)
        print(f"\n[HF tqai K3/V2] {result}")
        assert len(result) > len(self.PROMPT)


# ─── MLX tests ───

class TestMLX:
    MODEL_BF16 = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
    MODEL_Q8 = "mlx-community/Qwen2.5-0.5B-Instruct-8bit"
    MODEL_Q4 = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    PROMPT = "What is 2+2? Answer in one word:"

    @pytest.fixture(scope="class", params=["bf16", "q8", "q4"])
    def mlx_model(self, request):
        import mlx_lm

        model_map = {"bf16": self.MODEL_BF16, "q8": self.MODEL_Q8, "q4": self.MODEL_Q4}
        model_id = model_map[request.param]
        print(f"\nLoading MLX model: {model_id}")
        model, tokenizer = mlx_lm.load(model_id)
        return model, tokenizer, request.param

    def test_baseline(self, mlx_model):
        import mlx_lm

        model, tokenizer, variant = mlx_model
        result = mlx_lm.generate(model, tokenizer, prompt=self.PROMPT, max_tokens=30)
        print(f"\n[MLX {variant} baseline] {result}")
        assert len(result) > 0

    def test_with_tqai(self, mlx_model):
        import mlx_lm

        import tqai

        model, tokenizer, variant = mlx_model
        tqai.patch(model, bits_k=4, bits_v=2, backend="mlx")
        try:
            result = mlx_lm.generate(model, tokenizer, prompt=self.PROMPT, max_tokens=30)
            print(f"\n[MLX {variant} + tqai K4/V2] {result}")
            assert len(result) > 0
        finally:
            tqai.unpatch(model)
