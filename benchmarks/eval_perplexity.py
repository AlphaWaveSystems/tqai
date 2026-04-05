"""Perplexity evaluation for tqai benchmarks.

Measures perplexity on a fixed text sample using sliding window evaluation.
Works with both HuggingFace (PyTorch) and MLX backends.
"""

from __future__ import annotations

import math

# Fixed eval text — first ~1000 tokens of a Wikipedia-style passage
EVAL_TEXT = (
    "The theory of general relativity, published by Albert Einstein in 1915, "
    "fundamentally changed our understanding of gravity. Rather than describing "
    "gravity as a force between masses, Einstein proposed that massive objects "
    "cause a distortion in space-time, which is felt as gravity. This was a "
    "radical departure from Newton's law of universal gravitation, which had "
    "successfully described gravitational phenomena for over two centuries. "
    "The key insight of general relativity is the equivalence principle, which "
    "states that the effects of gravity are indistinguishable from the effects "
    "of acceleration. Einstein realized that a person in a closed elevator "
    "could not tell whether the elevator was stationary on Earth's surface or "
    "accelerating through space at 9.8 meters per second squared. This "
    "seemingly simple observation led to profound consequences for our "
    "understanding of the universe. One of the most striking predictions of "
    "general relativity is the bending of light by gravity. When light passes "
    "near a massive object, its path is curved by the warping of space-time. "
    "This effect was first confirmed during the solar eclipse of 1919, when "
    "Arthur Eddington observed that stars near the Sun appeared to shift "
    "position, exactly as Einstein had predicted. Another prediction is "
    "gravitational time dilation: clocks run slower in stronger gravitational "
    "fields. This effect has been confirmed with atomic clocks on aircraft and "
    "satellites, and is essential for the accuracy of GPS systems. Without "
    "corrections for both special and general relativistic effects, GPS "
    "positions would drift by about 10 kilometers per day. Perhaps the most "
    "dramatic prediction of general relativity is the existence of black holes, "
    "regions of space where gravity is so strong that nothing, not even light, "
    "can escape. Black holes form when massive stars collapse at the end of "
    "their lives. The boundary of a black hole is called the event horizon. "
    "In 2019, the Event Horizon Telescope collaboration produced the first "
    "direct image of a black hole, located in the galaxy M87, confirming "
    "decades of theoretical work. General relativity also predicts the "
    "existence of gravitational waves, ripples in space-time caused by "
    "accelerating massive objects. These waves were first directly detected "
    "in 2015 by the LIGO observatory, from the merger of two black holes "
    "about 1.3 billion light-years away. This detection opened a new era "
    "of gravitational wave astronomy and earned the 2017 Nobel Prize in "
    "Physics. The equations of general relativity also describe the evolution "
    "of the universe as a whole. When Einstein applied his equations to the "
    "entire cosmos, he found that the universe could not be static — it must "
    "be either expanding or contracting."
)


def perplexity_hf(model, tokenizer, text: str = EVAL_TEXT, max_length: int = 512) -> float:
    """Measure perplexity using HuggingFace model."""
    import torch

    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss

    return math.exp(neg_log_likelihood.item())


def perplexity_mlx(model, tokenizer, text: str = EVAL_TEXT, max_length: int = 512) -> float:
    """Measure perplexity using MLX model."""
    import mlx.core as mx
    import mlx.nn as nn

    tokens = tokenizer.encode(text, add_special_tokens=True)[:max_length]
    tokens = mx.array(tokens)[None]  # (1, seq_len)

    logits = model(tokens)  # (1, seq_len, vocab_size)
    logits = logits[:, :-1, :]  # shift: predict next token
    targets = tokens[:, 1:]

    loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
    mx.eval(loss)
    return math.exp(loss.item())


def generate_tokens(
    model,
    tokenizer,
    text: str = EVAL_TEXT,
    max_new_tokens: int = 100,
    backend: str = "torch",
) -> list[int]:
    """Generate tokens greedily. Returns token IDs for match comparison."""
    if backend == "mlx":
        import mlx_lm

        output = mlx_lm.generate(
            model, tokenizer, prompt=text[:200], max_tokens=max_new_tokens
        )
        return tokenizer.encode(output)
    else:
        import torch

        inputs = tokenizer(text[:200], return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        return output[0].tolist()


def compute_match_rate(baseline_tokens: list, compressed_tokens: list) -> float:
    """Fraction of tokens that match between two sequences."""
    min_len = min(len(baseline_tokens), len(compressed_tokens))
    if min_len == 0:
        return 0.0
    matches = sum(
        1 for a, b in zip(baseline_tokens[:min_len], compressed_tokens[:min_len]) if a == b
    )
    return matches / min_len
