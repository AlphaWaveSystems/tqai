"""Model family adapters."""

from tqai.adapters.dit import DiTAdapter
from tqai.adapters.llm import LLMAdapter
from tqai.adapters.wan import WANAdapter
from tqai.pipeline.registry import register_adapter

register_adapter("llm", LLMAdapter)
register_adapter("dit", DiTAdapter)
register_adapter("wan", WANAdapter)
