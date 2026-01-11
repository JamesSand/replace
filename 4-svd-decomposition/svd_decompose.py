#!/usr/bin/env python3
# compare_layer10_singular_values_llama_hf_3rows.py
#
# Based on your current script:
# - Load TWO HuggingFace LLaMA-family checkpoints from your DEFAULT PATHS
# - Only layer=10 (0-based)
# - Only Attention q_proj + MLP gate_proj (NO up/down; MLP uses gate only)
#
# Plot 3 rows x 2 cols:
#   Row 1: singular values (Before vs After)
#   Row 2: relative change   (after - before) / before
#   Row 3: absolute change   (after - before)
#   Col 1: Attention q_proj
#   Col 2: MLP gate_proj
#
# Usage:
#   python compare_layer10_singular_values_llama_hf_3rows.py
#   # or override
#   python compare_layer10_singular_values_llama_hf_3rows.py --layer 10 --topk 256 --out fig.png

import argparse
import os
from typing import Tuple

import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM


# ----------------------------
# Utils
# ----------------------------
def basename_like(s: str) -> str:
    s = s.rstrip("/").rstrip("\\")
    if "/" in s:
        return s.split("/")[-1]
    if "\\" in s:
        return s.split("\\")[-1]
    return s


def svd_singular_values(weight: torch.Tensor) -> torch.Tensor:
    """SVD singular values for 2D+ weight tensor: reshape to [out_dim, -1]."""
    if weight.ndim < 2:
        raise ValueError(f"Expected weight.ndim >= 2, got {weight.shape}")
    mat = weight.detach().to(torch.float32).reshape(weight.shape[0], -1).cpu()
    return torch.linalg.svdvals(mat).cpu()


def relative_change(after: torch.Tensor, before: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """(after - before) / before"""
    min_len = min(int(after.numel()), int(before.numel()))
    a = after[:min_len]
    b = before[:min_len]
    return (a - b) / (b.abs() + eps)


def absolute_change(after: torch.Tensor, before: torch.Tensor) -> torch.Tensor:
    """after - before"""
    min_len = min(int(after.numel()), int(before.numel()))
    return after[:min_len] - before[:min_len]


def load_llama_causal_lm(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    return model


def get_layer_weights(model, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    LLaMA-style HF:
      model.model.layers[layer].self_attn.q_proj.weight
      model.model.layers[layer].mlp.gate_proj.weight
    """
    try:
        layers = model.model.layers
    except Exception as e:
        raise AttributeError(
            "Cannot access model.model.layers. This checkpoint might not be LLaMA-style. "
            "Try: print(model) and adjust access paths."
        ) from e

    if layer < 0 or layer >= len(layers):
        raise IndexError(f"layer={layer} out of range: [0, {len(layers)-1}]")

    lyr = layers[layer]
    q_w = lyr.self_attn.q_proj.weight
    gate_w = lyr.mlp.gate_proj.weight
    return q_w, gate_w


# ----------------------------
# Plotting
# ----------------------------
def plot_grid_3rows(
    s_before_q: torch.Tensor,
    s_after_q: torch.Tensor,
    s_before_gate: torch.Tensor,
    s_after_gate: torch.Tensor,
    model_before_name: str,
    model_after_name: str,
    layer: int,
    topk: int,
    out_path: str,
):
    def _topk(x: torch.Tensor) -> torch.Tensor:
        if topk <= 0:
            return x
        return x[: min(int(x.numel()), topk)]

    bq = _topk(s_before_q)
    aq = _topk(s_after_q)
    bg = _topk(s_before_gate)
    ag = _topk(s_after_gate)

    rq = relative_change(aq, bq)
    rg = relative_change(ag, bg)
    dq = absolute_change(aq, bq)
    dg = absolute_change(ag, bg)

    mean_abs_rq = float(rq.abs().mean().item())
    mean_abs_rg = float(rg.abs().mean().item())
    mean_abs_dq = float(dq.abs().mean().item())
    mean_abs_dg = float(dg.abs().mean().item())

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex="col")

    # Row 1: singular values
    ax = axes[0, 0]
    ax.plot(range(1, len(bq) + 1), bq.numpy(), label="Before")
    ax.plot(range(1, len(aq) + 1), aq.numpy(), label="After")
    ax.set_title("Attention q_proj: singular values")
    ax.set_ylabel(r"$\sigma_i$")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(range(1, len(bg) + 1), bg.numpy(), label="Before")
    ax.plot(range(1, len(ag) + 1), ag.numpy(), label="After")
    ax.set_title("MLP gate_proj: singular values")
    ax.set_ylabel(r"$\sigma_i$")
    ax.legend()

    # Row 2: relative change
    ax = axes[1, 0]
    ax.plot(range(1, len(rq) + 1), rq.numpy())
    ax.axhline(0.0)
    ax.set_title("Attention q_proj: relative change")
    ax.set_ylabel(r"$(\sigma^{after}-\sigma^{before})/\sigma^{before}$")
    ax.text(0.02, 0.92, f"mean |Δσ/σ| = {mean_abs_rq*100:.4f}%", transform=ax.transAxes, va="top")

    ax = axes[1, 1]
    ax.plot(range(1, len(rg) + 1), rg.numpy())
    ax.axhline(0.0)
    ax.set_title("MLP gate_proj: relative change")
    ax.set_ylabel(r"$(\sigma^{after}-\sigma^{before})/\sigma^{before}$")
    ax.text(0.02, 0.92, f"mean |Δσ/σ| = {mean_abs_rg*100:.4f}%", transform=ax.transAxes, va="top")

    # Row 3: absolute change
    ax = axes[2, 0]
    ax.plot(range(1, len(dq) + 1), dq.numpy())
    ax.axhline(0.0)
    ax.set_title("Attention q_proj: absolute change")
    ax.set_xlabel(f"Rank (top-k={len(dq)})")
    ax.set_ylabel(r"$\sigma^{after}-\sigma^{before}$")
    ax.text(0.02, 0.92, f"mean |Δσ| = {mean_abs_dq:.6g}", transform=ax.transAxes, va="top")

    ax = axes[2, 1]
    ax.plot(range(1, len(dg) + 1), dg.numpy())
    ax.axhline(0.0)
    ax.set_title("MLP gate_proj: absolute change")
    ax.set_xlabel(f"Rank (top-k={len(dg)})")
    ax.set_ylabel(r"$\sigma^{after}-\sigma^{before}$")
    ax.text(0.02, 0.92, f"mean |Δσ| = {mean_abs_dg:.6g}", transform=ax.transAxes, va="top")

    suptitle = f"LLaMA Layer {layer} Spectrum | Before: {model_before_name}  →  After: {model_after_name}"
    fig.suptitle(suptitle)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"[OK] saved figure -> {out_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_before",
        type=str,
        default="/home/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Llama-3.2-3B-Instruct",
        help="HF model id/path (before)",
    )
    parser.add_argument(
        "--model_after",
        type=str,
        default="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted/llama-muon-muonlr1e-4-spectral_norm-muonadamlr1e-6-20260110_005142-global_step_200",
        help="HF model id/path (after)",
    )
    parser.add_argument("--layer", type=int, default=10, help="layer index (0-based), default=10")
    parser.add_argument("--topk", type=int, default=0, help="plot only top-k ranks (0 = all)")
    parser.add_argument("--out", type=str, default="llama_muon_layer10_q_gate_3rows.png", help="output figure path")
    args = parser.parse_args()

    before_name = basename_like(args.model_before)
    after_name = basename_like(args.model_after)

    print(f"[LOAD] before: {args.model_before}")
    model_b = load_llama_causal_lm(args.model_before)

    print(f"[LOAD] after : {args.model_after}")
    model_a = load_llama_causal_lm(args.model_after)

    q_b, gate_b = get_layer_weights(model_b, args.layer)
    q_a, gate_a = get_layer_weights(model_a, args.layer)

    print("[FOUND] shapes:")
    print(f"  q_proj   before {tuple(q_b.shape)} | after {tuple(q_a.shape)}")
    print(f"  gate_proj before {tuple(gate_b.shape)} | after {tuple(gate_a.shape)}")

    print("[SVD] q_proj ...")
    s_before_q = svd_singular_values(q_b)
    s_after_q = svd_singular_values(q_a)

    print("[SVD] gate_proj ...")
    s_before_gate = svd_singular_values(gate_b)
    s_after_gate = svd_singular_values(gate_a)

    plot_grid_3rows(
        s_before_q=s_before_q,
        s_after_q=s_after_q,
        s_before_gate=s_before_gate,
        s_after_gate=s_after_gate,
        model_before_name=before_name,
        model_after_name=after_name,
        layer=args.layer,
        topk=args.topk,
        out_path=args.out,
    )

    del model_b, model_a
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
