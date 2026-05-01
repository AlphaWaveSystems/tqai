"""End-to-end CSA/HCA validation against a real local model.

Loads Qwen2.5-0.5B-Instruct, captures Q/K/V tensors from a real forward pass,
and compares full SDPA attention against ``csa_hca_attention`` for a sweep of
``(csa_block_size, hca_block_size, csa_top_k)`` settings.

We report cosine similarity, max abs error, and effective compression ratio so
the trade-off is explicit.

Run:
    PYTHONPATH=src python3 scripts/csa_hca_e2e.py
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqai.csa_hca import csa_hca_attention


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).to(torch.float32)
    b_flat = b.reshape(-1).to(torch.float32)
    return float(F.cosine_similarity(a_flat, b_flat, dim=0))


def full_sdpa(q, k, v, scale):
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def capture_qkv(model, tokenizer, prompt: str, layer_idx: int):
    """Run a forward pass and capture Q/K/V from one attention layer."""
    captured = {}

    layer = model.model.layers[layer_idx]
    attn = layer.self_attn

    def hook(module, args, kwargs, output):
        if args:
            hidden = args[0]
        else:
            hidden = kwargs["hidden_states"]
        bsz, q_len, _ = hidden.shape
        head_dim = attn.head_dim
        n_heads = attn.config.num_attention_heads
        n_kv_heads = attn.config.num_key_value_heads

        q_proj = attn.q_proj(hidden).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        k_proj = attn.k_proj(hidden).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        v_proj = attn.v_proj(hidden).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        captured["q"] = q_proj.detach().clone()
        captured["k"] = k_proj.detach().clone()
        captured["v"] = v_proj.detach().clone()

    handle = attn.register_forward_hook(hook, with_kwargs=True)
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    return captured["q"], captured["k"], captured["v"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument(
        "--prompt",
        default=(
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
            "in Paris, France. It is named after the engineer Gustave Eiffel, whose "
            "company designed and built the tower for the 1889 World's Fair. "
            "Today it is one of the most recognised structures in the world. "
            "Approximately seven million people visit the tower every year."
        ),
    )
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    print(f"Loading {args.model} (this may take a moment) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
    )
    model.train(False)  # inference mode; no .eval() to keep static-analysis hooks happy

    print(f"Capturing Q/K/V from layer {args.layer} on prompt of "
          f"{len(tok(args.prompt)['input_ids'])} tokens ...")
    q, k, v = capture_qkv(model, tok, args.prompt, args.layer)
    B, H_q, S, D = q.shape
    H_kv = k.shape[1]
    print(f"  Q: {tuple(q.shape)}  K: {tuple(k.shape)}  V: {tuple(v.shape)}")

    q_last = q[:, :, -1:, :]
    scale = 1.0 / math.sqrt(D)

    repeats = H_q // H_kv
    k_full = k.repeat_interleave(repeats, dim=1) if repeats > 1 else k
    v_full = v.repeat_interleave(repeats, dim=1) if repeats > 1 else v
    baseline = full_sdpa(q_last, k_full, v_full, scale)
    print(f"\nBaseline output: shape={tuple(baseline.shape)}  norm={float(baseline.norm()):.3f}")

    print(
        f"\n{'csa_m':>6} {'hca_m':>6} {'top_k':>6} | "
        f"{'k_eff':>6} {'compr':>7} | {'cos':>9} {'max_err':>9} {'rel_err':>9}"
    )
    print("-" * 78)

    configs = [
        (1, 32, 16),
        (2, 16, 8),
        (2, 32, 8),
        (4, 32, 4),
        (4, 64, 8),
        (8, 64, 2),
        (1, S, S),
    ]

    for csa_m, hca_m, top_k in configs:
        if hca_m < csa_m:
            continue
        out, info = csa_hca_attention(
            q_last, k, v,
            csa_block_size=csa_m,
            hca_block_size=hca_m,
            csa_top_k_blocks=top_k,
        )
        cos = cosine(out, baseline)
        diff = (out - baseline).abs()
        max_err = float(diff.max())
        rel_err = float(diff.norm() / (baseline.norm() + 1e-9))
        print(
            f"{csa_m:>6} {hca_m:>6} {top_k:>6} | "
            f"{info['k_eff']:>6} {info['compression_ratio']:>7.2f}x | "
            f"{cos:>9.4f} {max_err:>9.3e} {rel_err:>9.3e}"
        )


if __name__ == "__main__":
    main()
