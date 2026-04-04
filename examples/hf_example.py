"""Example: TurboQuant with HuggingFace Transformers (PyTorch).

Usage:
    pip install tqai[torch] transformers accelerate
    python examples/hf_example.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

import tqai

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")

# Create TurboQuant-compressed KV cache (K4/V2 = ~3 bits avg)
cache = tqai.patch(model, bits_k=4, bits_v=2)

prompt = "Explain the theory of relativity in simple terms."
print(f"\nPrompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(
    **inputs,
    past_key_values=cache,
    max_new_tokens=200,
    do_sample=False,
)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\nResponse:\n{response}")
print(f"\nCache sequence length: {cache.get_seq_length()}")
