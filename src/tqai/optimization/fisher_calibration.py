"""Offline gradient-based Fisher Information calibration.

The runtime ``FisherScorer`` in ``tqai.scorers.fisher`` uses ``mean(x^2)``
as a proxy for the Fisher Information diagonal because computing real
gradients per attention call is too expensive. The proxy over-estimates
importance and routes everything to the high-bit tier — see
``benchmarks/results/pipeline_benchmark.json`` (NMSE 0.116 for fisher+tiered
vs 0.009 baseline).

This module implements **proper offline calibration**: load the model,
run a small calibration dataset through forward+backward passes, collect
the squared gradients of the K and V projection weights at each
attention layer, aggregate them, and save a per-layer Fisher importance
table to JSON. The companion :class:`tqai.scorers.fisher_static.FisherStaticScorer`
loads this JSON and serves precomputed scores at runtime — no proxy,
no per-call gradient computation, no overhead beyond a dictionary lookup.

The Fisher Information diagonal of a parameter ``theta`` is::

    F[theta] = E[(dL/dtheta)^2]

where ``L`` is the loss and the expectation is over the data distribution.
For KV cache quantization we want a per-LAYER importance score (not
per-element), so we average the squared gradients over each projection
weight matrix.

References:
    - Fisher Information for Neural Network Compression: arXiv:1906.08589
    - APTQ attention-aware quantization: arXiv:2402.14866
    - Optimal Brain Surgeon (Hassibi & Stork, 1992) — original use of
      Fisher diagonal as a parameter importance estimator
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class FisherCalibration:
    """Per-layer Fisher importance table from offline calibration."""

    model_id: str
    timestamp: str
    num_samples: int
    num_layers: int
    layer_fisher_k: list[float]  # per-layer mean squared gradient for K proj
    layer_fisher_v: list[float]  # per-layer mean squared gradient for V proj
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FisherCalibration:
        return cls(
            model_id=data["model_id"],
            timestamp=data["timestamp"],
            num_samples=data["num_samples"],
            num_layers=data["num_layers"],
            layer_fisher_k=list(data["layer_fisher_k"]),
            layer_fisher_v=list(data["layer_fisher_v"]),
            notes=data.get("notes", ""),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> FisherCalibration:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _find_kv_projections(model: Any) -> list[tuple[int, Any, Any]]:
    """Walk attention modules and return (layer_idx, k_proj, v_proj) tuples.

    Handles two common naming conventions:
      - Standard HF (Llama, Qwen, Mistral, etc.): ``q_proj``, ``k_proj``,
        ``v_proj`` as separate child modules of an attention module.
      - Fused QKV (GPT-2 style): ``c_attn`` containing all three projections
        in a single weight matrix. We treat the whole matrix as one Fisher
        target since we cannot cheaply split out the K and V slices without
        knowing the model-specific layout.

    Returns:
        A list of ``(layer_idx, k_proj, v_proj)`` tuples. For fused models,
        ``k_proj`` and ``v_proj`` will be the same module (the fused
        ``c_attn``) and the calibrator will only count it once per layer.
    """
    from tqai.module_utils import iter_attention_modules

    results: list[tuple[int, Any, Any]] = []
    for layer_idx, (_name, module) in enumerate(iter_attention_modules(model)):
        k_proj = getattr(module, "k_proj", None)
        v_proj = getattr(module, "v_proj", None)
        if k_proj is not None and v_proj is not None:
            results.append((layer_idx, k_proj, v_proj))
            continue
        # Fused QKV
        c_attn = getattr(module, "c_attn", None)
        if c_attn is not None:
            results.append((layer_idx, c_attn, c_attn))
            continue
        qkv = getattr(module, "qkv", None)
        if qkv is not None:
            results.append((layer_idx, qkv, qkv))
    return results


def _squared_grad_mean(projection: Any) -> float:
    """Read the mean of squared gradients from a projection module's weight.

    Returns the mean of the squared gradient elements (the Fisher
    Information diagonal averaged over the weight matrix). Returns 0.0
    if the gradient is None (e.g., the projection wasn't reached or the
    backward pass was skipped).
    """
    weight = getattr(projection, "weight", None)
    if weight is None:
        return 0.0
    grad = getattr(weight, "grad", None)
    if grad is None:
        return 0.0
    return float((grad.float() ** 2).mean().item())


def calibrate_fisher(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    output_path: str | Path | None = None,
    max_length: int = 512,
    device: str | None = None,
    notes: str = "",
) -> FisherCalibration:
    """Run offline Fisher calibration on a HuggingFace model.

    Performs forward + backward passes on each calibration prompt,
    collects per-layer mean-squared gradients for the K and V
    projection weights, aggregates across all samples, and optionally
    saves the result to JSON.

    The calibration is **gradient-based**: it computes ``dL/dtheta`` where
    ``L`` is the next-token cross-entropy loss and ``theta`` is each
    projection weight. This is the actual Fisher Information diagonal,
    not the runtime activation proxy used by the regular ``FisherScorer``.

    Args:
        model: HuggingFace causal LM (must have a ``.forward()`` that
            accepts ``input_ids`` and returns logits or a struct with
            ``.logits``).
        tokenizer: HuggingFace tokenizer for encoding the prompts.
        prompts: List of calibration prompts. 8-32 typically suffices.
        output_path: If provided, write the calibration JSON here.
        max_length: Maximum sequence length per prompt (longer prompts
            are truncated).
        device: Optional device override (``"cpu"``, ``"cuda"``, ``"mps"``).
            If None, uses the model's existing device.
        notes: Optional free-form note saved alongside the calibration
            for provenance.

    Returns:
        A :class:`FisherCalibration` instance with per-layer K and V
        Fisher diagonals.

    Raises:
        ValueError: If no attention layers can be found in the model.
        RuntimeError: If torch is not available.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Fisher calibration requires PyTorch. Install with: pip install tqai[torch]"
        ) from exc

    if not prompts:
        raise ValueError("At least one calibration prompt is required")

    kv_projections = _find_kv_projections(model)
    if not kv_projections:
        raise ValueError(
            "No attention K/V projections found. Calibration requires a model "
            "with detectable q_proj/k_proj/v_proj or c_attn modules."
        )

    num_layers = len(kv_projections)
    fisher_k = [0.0] * num_layers
    fisher_v = [0.0] * num_layers

    # Determine device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
        model = model.to(device)

    # Set training mode (needs grads enabled).  We use train(True) instead
    # of the .train() shorthand to be unambiguous.
    model.train(True)

    # We don't want to update params; just collect grads
    for p in model.parameters():
        p.requires_grad_(False)
    # Re-enable grads only on the K/V projections we care about
    for _, k_proj, v_proj in kv_projections:
        if hasattr(k_proj, "weight") and k_proj.weight is not None:
            k_proj.weight.requires_grad_(True)
        if v_proj is not k_proj and hasattr(v_proj, "weight") and v_proj.weight is not None:
            v_proj.weight.requires_grad_(True)

    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(device)

        # Zero existing grads
        for _, k_proj, v_proj in kv_projections:
            if hasattr(k_proj, "weight") and k_proj.weight.grad is not None:
                k_proj.weight.grad.zero_()
            if v_proj is not k_proj and hasattr(v_proj, "weight"):
                if v_proj.weight.grad is not None:
                    v_proj.weight.grad.zero_()

        # Forward pass with labels = input_ids (HF computes shifted CE loss)
        try:
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        except TypeError:
            # Fallback for models that don't accept labels
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if loss is None:
            continue

        loss.backward()

        # Accumulate squared gradients per layer
        for layer_idx, k_proj, v_proj in kv_projections:
            fisher_k[layer_idx] += _squared_grad_mean(k_proj)
            if v_proj is not k_proj:
                fisher_v[layer_idx] += _squared_grad_mean(v_proj)
            else:
                # Fused: charge half to V (rough approximation)
                fisher_v[layer_idx] += _squared_grad_mean(v_proj) * 0.5

    # Average across samples
    n = len(prompts)
    fisher_k = [v / n for v in fisher_k]
    fisher_v = [v / n for v in fisher_v]

    # Back to eval/inference mode
    model.train(False)

    calibration = FisherCalibration(
        model_id=getattr(getattr(model, "config", None), "_name_or_path", "<unknown>"),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        num_samples=n,
        num_layers=num_layers,
        layer_fisher_k=fisher_k,
        layer_fisher_v=fisher_v,
        notes=notes,
    )

    if output_path is not None:
        calibration.save(output_path)

    return calibration
