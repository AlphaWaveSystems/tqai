"""Model family adapters."""

from tqai.pipeline.registry import register_adapter
from tqai.adapters.llm import LLMAdapter
from tqai.adapters.dit import DiTAdapter
from tqai.adapters.wan import WANAdapter

register_adapter("llm", LLMAdapter)
register_adapter("dit", DiTAdapter)
register_adapter("wan", WANAdapter)
