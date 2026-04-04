"""tqai CLI — generate text, run benchmarks, and validate.

Usage:
    tqai run "Your prompt here" --model mlx-community/Llama-3.1-8B-Instruct-4bit
    tqai run "Your prompt" --model Qwen/Qwen2.5-3B-Instruct --backend torch
    tqai benchmark
    tqai benchmark --bits-k 3 --bits-v 2 --head-dim 128
    tqai info
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np


def cmd_info(args):
    """Print library and environment info."""
    import tqai

    print(f"tqai v{tqai.__version__}")
    print()

    from tqai.backend import detect_backend

    try:
        backend = detect_backend()
        print(f"Default backend: {backend}")
    except RuntimeError as e:
        print(f"No backend: {e}")

    for name, pkg in [("torch", "torch"), ("mlx", "mlx.core")]:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "?")
            print(f"  {name}: {version}")
        except ImportError:
            print(f"  {name}: not installed")

    from tqai.codebook.registry import CodebookRegistry

    reg = CodebookRegistry()
    dims = [64, 96, 128, 256]
    bits_list = [2, 3, 4]
    available = []
    for d in dims:
        for b in bits_list:
            fname = reg.codebook_filename(d, b)
            try:
                from importlib.resources import files

                data_files = files("tqai.codebook") / "data"
                (data_files / fname).open("rb").close()
                available.append(f"d{d}/b{b}")
            except (FileNotFoundError, TypeError):
                pass
    print(f"\nShipped codebooks: {len(available)}")
    print(f"  {', '.join(available)}")


def cmd_run(args):
    """Generate text with TurboQuant-compressed KV cache."""
    model_id = args.model
    prompt = args.prompt
    bits_k = args.bits_k
    bits_v = args.bits_v
    max_tokens = args.max_tokens
    backend = args.backend
    no_tqai = args.no_tqai
    tqai_config = getattr(args, "tqai_config", None)
    compress_hidden = getattr(args, "compress_hidden", False)
    compress_ffn = getattr(args, "compress_ffn", False)
    bits_hidden = getattr(args, "bits_hidden", 8)
    bits_ffn = getattr(args, "bits_ffn", 8)
    if getattr(args, "compress_all", False):
        compress_hidden = True
        compress_ffn = True

    from tqai.backend import detect_backend

    detected = backend or detect_backend()

    print(f"Model:   {model_id}")
    print(f"Backend: {detected}")
    if not no_tqai:
        kv_str = f"K{bits_k}/V{bits_v}"
        fwd_parts = []
        if compress_hidden:
            fwd_parts.append(f"hidden={bits_hidden}b")
        if compress_ffn:
            fwd_parts.append(f"ffn={bits_ffn}b")
        fwd_str = "  +" + "+".join(fwd_parts) if fwd_parts else ""
        print(f"Config:  {kv_str}{fwd_str}")
    else:
        print("Config:  baseline (no compression)")
    print(f"Tokens:  {max_tokens}")
    print()

    if detected == "mlx":
        _run_mlx(model_id, prompt, bits_k, bits_v, max_tokens, no_tqai, tqai_config)
    else:
        _run_hf(
            model_id, prompt, bits_k, bits_v, max_tokens, no_tqai, backend, tqai_config,
            compress_hidden=compress_hidden, compress_ffn=compress_ffn,
            bits_hidden=bits_hidden, bits_ffn=bits_ffn,
        )


def _run_mlx(model_id, prompt, bits_k, bits_v, max_tokens, no_tqai, tqai_config=None):
    import mlx_lm

    import tqai

    print(f"Loading {model_id}...")
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(model_id)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s\n")

    if not no_tqai:
        tqai.patch(model, bits_k=bits_k, bits_v=bits_v, backend="mlx", config_path=tqai_config)

    print(f"Prompt: {prompt}\n")
    print("--- Response ---")
    t0 = time.perf_counter()
    response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    elapsed = time.perf_counter() - t0
    print(response)
    print(f"\n--- {len(response.split())} words in {elapsed:.1f}s ---")

    if not no_tqai:
        tqai.unpatch(model)


def _run_hf(
    model_id, prompt, bits_k, bits_v, max_tokens, no_tqai, backend, tqai_config=None,
    compress_hidden=False, compress_ffn=False, bits_hidden=8, bits_ffn=8,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import tqai

    print(f"Loading {model_id}...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    try:
        import torch
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s\n")

    cache = None
    if not no_tqai:
        cache = tqai.patch(
            model,
            bits_k=bits_k, bits_v=bits_v,
            backend=backend or "torch",
            config_path=tqai_config,
            compress_hidden=compress_hidden, bits_hidden=bits_hidden,
            compress_ffn=compress_ffn, bits_ffn=bits_ffn,
        )

    inputs = tokenizer(prompt, return_tensors="pt")

    print(f"Prompt: {prompt}\n")
    print("--- Response ---")
    t0 = time.perf_counter()
    kwargs = {**inputs, "max_new_tokens": max_tokens, "do_sample": False}
    if cache is not None:
        kwargs["past_key_values"] = cache
    output = model.generate(**kwargs)
    elapsed = time.perf_counter() - t0
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(result)
    print(f"\n--- {len(result.split())} words in {elapsed:.1f}s ---")

    if cache is not None:
        print(f"Cache seq length: {cache.get_seq_length()}")


def cmd_compare(args):
    """Compare baseline vs tqai output for the same prompt."""
    model_id = args.model
    prompt = args.prompt
    bits_k = args.bits_k
    bits_v = args.bits_v
    max_tokens = args.max_tokens

    from tqai.backend import detect_backend

    detected = detect_backend()

    print(f"Comparing: {model_id}")
    print(f"Config:    K{bits_k}/V{bits_v}")
    print(f"Backend:   {detected}")
    print()

    if detected == "mlx":
        import mlx_lm

        import tqai

        print(f"Loading {model_id}...")
        model, tokenizer = mlx_lm.load(model_id)

        print("\n=== BASELINE ===")
        baseline = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
        print(baseline)

        print(f"\n=== tqai K{bits_k}/V{bits_v} ===")
        tqai.patch(model, bits_k=bits_k, bits_v=bits_v, backend="mlx")
        compressed = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
        print(compressed)
        tqai.unpatch(model)

        # Simple similarity metric
        baseline_words = set(baseline.lower().split())
        compressed_words = set(compressed.lower().split())
        if baseline_words:
            overlap = len(baseline_words & compressed_words) / len(baseline_words | compressed_words)
            print(f"\n=== Word overlap: {overlap:.0%} ===")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import tqai

        print(f"Loading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:
            import torch
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(model_id)

        inputs = tokenizer(prompt, return_tensors="pt")

        print("\n=== BASELINE ===")
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

        print(f"\n=== tqai K{bits_k}/V{bits_v} ===")
        cache = tqai.patch(model, bits_k=bits_k, bits_v=bits_v, backend="torch")
        output = model.generate(**inputs, past_key_values=cache, max_new_tokens=max_tokens, do_sample=False)
        print(tokenizer.decode(output[0], skip_special_tokens=True))


def cmd_convert(args):
    """Pre-convert a model for faster TurboQuant inference."""
    from tqai.convert import convert_model

    convert_model(
        model_id=args.model,
        output_dir=args.output,
        bits_k=args.bits_k,
        bits_v=args.bits_v,
        seed=args.seed,
        backend=args.backend,
    )


def cmd_benchmark(args):
    """Run quantization accuracy benchmark."""
    from tqai.backend import get_backend
    from tqai.quantizer import PolarQuantizer

    backend = args.backend
    bits_k = args.bits_k
    bits_v = args.bits_v
    head_dim = args.head_dim
    n_vectors = args.n_vectors

    print("tqai benchmark")
    print(f"  backend:  {backend or 'auto'}")
    print(f"  head_dim: {head_dim}")
    print(f"  bits_k:   {bits_k}")
    print(f"  bits_v:   {bits_v}")
    print(f"  vectors:  {n_vectors}")
    print()

    ops = get_backend(backend)
    print(f"Using backend: {type(ops).__name__}")
    print()

    for label, bits in [("Keys", bits_k), ("Values", bits_v)]:
        pq = PolarQuantizer(head_dim=head_dim, bits=bits, seed=42, ops=ops)
        x = ops.randn((n_vectors, head_dim), seed=123)

        t0 = time.perf_counter()
        indices, norms = pq.quantize(x)
        t_quant = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_hat = pq.dequantize(indices, norms)
        t_dequant = time.perf_counter() - t0

        x_np = ops.to_numpy(x)
        x_hat_np = ops.to_numpy(x_hat)

        mse = np.mean((x_np - x_hat_np) ** 2)
        signal = np.mean(x_np ** 2)
        nmse = mse / signal
        snr = 10 * np.log10(signal / mse) if mse > 0 else float("inf")

        dot = np.sum(x_np * x_hat_np, axis=-1)
        na = np.linalg.norm(x_np, axis=-1)
        nb = np.linalg.norm(x_hat_np, axis=-1)
        cos_sims = dot / (na * nb + 1e-30)

        orig_bits = 16 * head_dim
        comp_bits = bits * head_dim + 16
        ratio = orig_bits / comp_bits
        saved = (1 - comp_bits / orig_bits) * 100

        bound = (math.sqrt(3) * math.pi / 2.0) / (4.0 ** bits)

        print(f"--- {label} ({bits}-bit) ---")
        print(f"  Compression:     {ratio:.1f}x ({saved:.0f}% saved)")
        print(f"  NMSE:            {nmse:.6f}  (bound: {bound:.6f})")
        print(f"  SNR:             {snr:.1f} dB")
        print(f"  Cosine sim:      mean={np.mean(cos_sims):.4f}  min={np.min(cos_sims):.4f}  p5={np.percentile(cos_sims, 5):.4f}")
        print(f"  Quantize time:   {t_quant*1000:.1f} ms")
        print(f"  Dequantize time: {t_dequant*1000:.1f} ms")
        print()

    k_bits = bits_k * head_dim + 16
    v_bits = bits_v * head_dim + 16
    total_orig = 2 * 16 * head_dim
    total_comp = k_bits + v_bits
    print(f"--- Combined K{bits_k}/V{bits_v} ---")
    print(f"  Per-token: {total_orig} -> {total_comp} bits  ({total_orig/8:.0f} -> {total_comp/8:.0f} bytes)")
    print(f"  Ratio:     {total_orig/total_comp:.1f}x")
    print(f"  Saved:     {(1 - total_comp/total_orig)*100:.0f}%")


def main():
    parser = argparse.ArgumentParser(
        prog="tqai",
        description="TurboQuant KV cache compression for local LLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  tqai info
  tqai benchmark
  tqai convert -m mlx-community/Llama-3.1-8B-Instruct-4bit -o ./llama-8b-tqai/
  tqai run "Explain gravity" -m mlx-community/Llama-3.1-8B-Instruct-4bit
  tqai run "Explain gravity" -m mlx-community/Llama-3.1-8B-Instruct-4bit --tqai-config ./llama-8b-tqai/
  tqai compare "Explain gravity" -m mlx-community/Llama-3.1-8B-Instruct-4bit
""",
    )
    sub = parser.add_subparsers(dest="command")

    # info
    sub.add_parser("info", help="Show library and environment info")

    # benchmark
    bench = sub.add_parser("benchmark", help="Run quantization accuracy benchmark")
    bench.add_argument("--backend", default=None, help="torch or mlx (auto)")
    bench.add_argument("--bits-k", type=int, default=4, help="Key bits (default: 4)")
    bench.add_argument("--bits-v", type=int, default=2, help="Value bits (default: 2)")
    bench.add_argument("--head-dim", type=int, default=128, help="Head dimension (default: 128)")
    bench.add_argument("--n-vectors", type=int, default=1000, help="Test vectors (default: 1000)")

    # run
    run = sub.add_parser("run", help="Generate text with TurboQuant compression")
    run.add_argument("prompt", help="Input prompt")
    run.add_argument("--model", "-m", required=True, help="Model ID (HuggingFace or mlx-community)")
    run.add_argument("--bits-k", type=int, default=4, help="Key bits (default: 4)")
    run.add_argument("--bits-v", type=int, default=2, help="Value bits (default: 2)")
    run.add_argument("--max-tokens", type=int, default=200, help="Max tokens (default: 200)")
    run.add_argument("--backend", default=None, help="torch or mlx (auto)")
    run.add_argument("--no-tqai", action="store_true", help="Run without compression (baseline)")
    run.add_argument("--tqai-config", default=None, help="Path to pre-converted tqai config dir")
    run.add_argument("--compress-all", action="store_true",
                     help="Enable all forward-pass compression (hidden + FFN, PyTorch only)")
    run.add_argument("--compress-hidden", action="store_true",
                     help="Compress residual stream (hidden states, PyTorch only)")
    run.add_argument("--compress-ffn", action="store_true",
                     help="Compress FFN intermediate activations (PyTorch only)")
    run.add_argument("--bits-hidden", type=int, default=8,
                     help="Bits for hidden state compression (default: 8)")
    run.add_argument("--bits-ffn", type=int, default=8,
                     help="Bits for FFN compression (default: 8)")

    # compare
    comp = sub.add_parser("compare", help="Compare baseline vs tqai output side by side")
    comp.add_argument("prompt", help="Input prompt")
    comp.add_argument("--model", "-m", required=True, help="Model ID")
    comp.add_argument("--bits-k", type=int, default=4, help="Key bits (default: 4)")
    comp.add_argument("--bits-v", type=int, default=2, help="Value bits (default: 2)")
    comp.add_argument("--max-tokens", type=int, default=100, help="Max tokens (default: 100)")

    # convert
    conv = sub.add_parser("convert", help="Pre-convert a model for faster tqai inference")
    conv.add_argument("--model", "-m", required=True, help="Model ID (HuggingFace or mlx-community)")
    conv.add_argument("--output", "-o", required=True, help="Output directory")
    conv.add_argument("--bits-k", type=int, default=4, help="Key bits (default: 4)")
    conv.add_argument("--bits-v", type=int, default=2, help="Value bits (default: 2)")
    conv.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    conv.add_argument("--backend", default=None, help="torch or mlx (auto)")

    args = parser.parse_args()
    if args.command == "info":
        cmd_info(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "convert":
        cmd_convert(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
