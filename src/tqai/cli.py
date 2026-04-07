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


def cmd_plugins(args):
    """List available pipeline plugins."""
    # Ensure all modules are imported so registrations happen
    import tqai.scorers  # noqa: F401
    import tqai.strategies  # noqa: F401

    try:
        import tqai.monitors  # noqa: F401
    except ImportError:
        pass
    try:
        import tqai.adapters  # noqa: F401
    except ImportError:
        pass

    from tqai.pipeline.registry import list_available

    available = list_available()
    print("tqai pipeline plugins\n")
    for category, items in available.items():
        label = category.capitalize()
        print(f"  {label:<12} {', '.join(items) if items else '(none)'}")
    print()


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
    use_qjl = getattr(args, "use_qjl", False)
    qjl_sketch_size = getattr(args, "qjl_sketch_size", 64)
    if getattr(args, "compress_all", False):
        compress_hidden = True
        compress_ffn = True

    # Build pipeline config from CLI flags
    scorer_name = getattr(args, "scorer", None)
    strategy_name = getattr(args, "strategy", None)
    pipeline_cfg = None
    if scorer_name or strategy_name:
        # Ensure registrations happen
        import tqai.scorers  # noqa: F401
        import tqai.strategies  # noqa: F401

        pipeline_cfg = {}
        if scorer_name:
            pipeline_cfg["scorer"] = scorer_name
        if strategy_name:
            pipeline_cfg["strategy"] = strategy_name

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
        _run_mlx(model_id, prompt, bits_k, bits_v, max_tokens, no_tqai, tqai_config, pipeline=pipeline_cfg)
    else:
        _run_hf(
            model_id, prompt, bits_k, bits_v, max_tokens, no_tqai, backend, tqai_config,
            compress_hidden=compress_hidden, compress_ffn=compress_ffn,
            bits_hidden=bits_hidden, bits_ffn=bits_ffn,
            use_qjl=use_qjl, qjl_sketch_size=qjl_sketch_size,
            pipeline=pipeline_cfg,
        )


def _run_mlx(model_id, prompt, bits_k, bits_v, max_tokens, no_tqai, tqai_config=None, pipeline=None):
    import mlx_lm

    import tqai

    print(f"Loading {model_id}...")
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(model_id)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s\n")

    if not no_tqai:
        tqai.patch(model, bits_k=bits_k, bits_v=bits_v, backend="mlx", config_path=tqai_config, pipeline=pipeline)

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
    use_qjl=False, qjl_sketch_size=64, pipeline=None,
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
            use_qjl=use_qjl, qjl_sketch_size=qjl_sketch_size,
            pipeline=pipeline,
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


def cmd_calibrate(args):
    """Run offline gradient-based Fisher Information calibration on a HF model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tqai.optimization.fisher_calibration import calibrate_fisher

    # Default calibration prompts — generic, broad-domain coverage
    default_prompts = [
        "The history of computing began with mechanical calculators.",
        "Machine learning is the study of algorithms that learn from data.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The French Revolution began in 1789 and reshaped European politics.",
        "Quantum mechanics describes the behavior of matter at the smallest scales.",
        "Climate change is driven by greenhouse gas emissions from human activity.",
        "The human brain contains approximately 86 billion neurons.",
        "Software engineering is both a craft and a discipline.",
        "Music theory explains the relationships between pitches and rhythms.",
        "Economics studies how societies allocate scarce resources.",
        "The genetic code is read in triplets called codons.",
        "Artificial neural networks are loosely inspired by biological neurons.",
        "The speed of light in vacuum is approximately 299,792 kilometers per second.",
        "Plate tectonics explains the slow movement of continents over time.",
        "Linear algebra underlies most of modern machine learning.",
        "The internet emerged from research projects in the 1960s and 1970s.",
    ]

    prompts = default_prompts[: args.num_samples]
    if args.prompts_file:
        from pathlib import Path
        prompts = Path(args.prompts_file).read_text().strip().splitlines()
        prompts = [p for p in prompts if p.strip()][: args.num_samples]

    print(f"Loading {args.model}...")
    import time
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    try:
        import torch
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    print(f"Calibrating with {len(prompts)} prompts (max_length={args.max_length})...")
    t0 = time.perf_counter()
    cal = calibrate_fisher(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        output_path=args.output,
        max_length=args.max_length,
        notes=f"Calibrated via tqai calibrate CLI on {args.num_samples} samples",
    )
    print(f"Calibration done in {time.perf_counter() - t0:.1f}s")
    print(f"  Layers:  {cal.num_layers}")
    print(f"  Samples: {cal.num_samples}")
    print(f"  Saved:   {args.output}")
    print()
    print("Per-layer Fisher (K projection):")
    for i, v in enumerate(cal.layer_fisher_k):
        bar = "#" * int(min(v / max(cal.layer_fisher_k) * 40, 40)) if max(cal.layer_fisher_k) > 0 else ""
        print(f"  layer {i:>3}: {v:.6e}  {bar}")


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

    # plugins
    sub.add_parser("plugins", help="List available pipeline plugins (scorers, strategies, monitors, adapters)")

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
    run.add_argument("--use-qjl", action="store_true",
                     help="Enable QJL Stage 2 residual sketch (research/non-softmax use only)")
    run.add_argument("--qjl-sketch-size", type=int, default=64,
                     help="Number of 1-bit JL projections for QJL (default: 64)")
    run.add_argument("--scorer", default=None,
                     help="Pipeline scorer plugin (e.g., palm, fisher, snr)")
    run.add_argument("--strategy", default=None,
                     help="Pipeline compression strategy (e.g., tiered, delta, delta2)")

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

    # calibrate
    cal = sub.add_parser(
        "calibrate",
        help="Run offline gradient-based Fisher calibration (PyTorch only)",
    )
    cal.add_argument("--model", "-m", required=True, help="HuggingFace model ID")
    cal.add_argument("--output", "-o", required=True, help="Output JSON path")
    cal.add_argument(
        "--num-samples", type=int, default=16,
        help="Number of calibration prompts (default: 16, max: built-in default set)",
    )
    cal.add_argument(
        "--max-length", type=int, default=512,
        help="Maximum tokens per calibration prompt (default: 512)",
    )
    cal.add_argument(
        "--prompts-file", default=None,
        help="Optional path to a text file with one prompt per line "
             "(overrides the built-in defaults)",
    )

    args = parser.parse_args()
    if args.command == "info":
        cmd_info(args)
    elif args.command == "plugins":
        cmd_plugins(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "convert":
        cmd_convert(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
