"""Example: TurboQuant with mlx-lm on Apple Silicon.

Usage:
    pip install tqai[mlx]
    python examples/mlx_example.py
"""

import mlx_lm

import tqai

MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

print(f"Loading {MODEL_ID}...")
model, tokenizer = mlx_lm.load(MODEL_ID)

# Patch model to use TurboQuant-compressed KV cache
tqai.patch(model, bits_k=4, bits_v=2, backend="mlx")

prompt = "Explain the theory of relativity in simple terms."
print(f"\nPrompt: {prompt}")

response = mlx_lm.generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=200,
)

print(f"\nResponse:\n{response}")

# Unpatch when done
tqai.unpatch(model)
