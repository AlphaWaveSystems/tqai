"""Standard HF/mlx-lm autoregressive LLM adapter.

Extracts the LLM-specific patching logic from ``_patch.py`` into the
``ModelAdapter`` protocol so the pipeline can auto-detect model type.
"""

from __future__ import annotations

from typing import Any


class LLMAdapter:
    """Adapter for standard autoregressive LLMs (HuggingFace / mlx-lm).

    Handles Llama, Qwen2, Mistral, Phi, Gemma, Falcon, GPT-NeoX, OPT,
    GPT-2, and other HuggingFace causal LM architectures.
    """

    name = "llm"

    def detect(self, model: Any) -> bool:
        # HF: has config.model_type or is CausalLM
        if hasattr(model, "config") and hasattr(model.config, "model_type"):
            return True
        # mlx-lm: has model.layers and model.args
        if hasattr(model, "layers") and hasattr(model, "args"):
            return True
        return False

    def get_attention_modules(self, model: Any):
        from tqai.module_utils import iter_attention_modules

        yield from iter_attention_modules(model)

    def get_head_info(self, model: Any) -> dict:
        config = getattr(model, "config", None)
        args = getattr(model, "args", None)
        source = config or args

        if source is None:
            raise ValueError("Cannot extract head info from model")

        hidden = getattr(source, "hidden_size", None) or getattr(source, "n_embd", 0)
        n_heads = getattr(source, "num_attention_heads", None) or getattr(source, "n_head", 1)
        head_dim = getattr(source, "head_dim", None)
        if head_dim is None and hidden and n_heads:
            head_dim = hidden // n_heads
        n_kv_heads = getattr(source, "num_key_value_heads", n_heads)
        n_layers = getattr(source, "num_hidden_layers", None) or getattr(source, "n_layer", 1)

        return {
            "head_dim": head_dim,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "n_layers": n_layers,
        }

    def patch(self, model: Any, pipeline: Any, config: Any) -> Any:
        from tqai._patch import _patch

        return _patch(model, config)

    def unpatch(self, model: Any) -> None:
        from tqai._patch import _unpatch

        _unpatch(model)
