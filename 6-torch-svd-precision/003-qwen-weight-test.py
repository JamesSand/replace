
#!/usr/bin/env python3
# real_weight_svd_study.py
import os
import json
import time
import math
import argparse

import torch
import matplotlib.pyplot as plt
from safetensors.torch import load_file


# ----------------------------
# Metrics
# ----------------------------
def relative_change(after: torch.Tensor, before: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (after - before) / (before.abs() + eps)

def absolute_change(after: torch.Tensor, before: torch.Tensor) -> torch.Tensor:
    return after - before


# ----------------------------
# Plotting (ONE matrix only)
# 3 rows x 1 col: (sigma), (relative change), (absolute change)
# ----------------------------
def plot_grid_3rows_one(
    s_before: torch.Tensor,
    s_after: torch.Tensor,
    model_before_name: str,
    model_after_name: str,
    layer: int,
    topk: int,
    out_path: str,
    title_prefix: str = "SVD Spectrum",
):
    def _topk(x: torch.Tensor) -> torch.Tensor:
        if topk <= 0:
            return x
        return x[: min(int(x.numel()), topk)]

    # move to CPU float64 for stable numpy()
    b = _topk(s_before.detach().to("cpu", torch.float64))
    a = _topk(s_after.detach().to("cpu", torch.float64))

    r = relative_change(a, b)
    d = absolute_change(a, b)

    mean_abs_r = float(r.abs().mean().item())
    mean_abs_d = float(d.abs().mean().item())

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Row 1: singular values
    ax = axes[0]
    ax.plot(range(1, len(b) + 1), b.numpy(), label="Before")
    ax.plot(range(1, len(a) + 1), a.numpy(), label="After")
    ax.set_title(f"{title_prefix}: singular values")
    ax.set_ylabel(r"$\sigma_i$")
    ax.legend()

    # Row 2: relative change
    ax = axes[1]
    ax.plot(range(1, len(r) + 1), r.numpy())
    ax.axhline(0.0)
    ax.set_title(f"{title_prefix}: relative change")
    ax.set_ylabel(r"$(\sigma^{after}-\sigma^{before})/\sigma^{before}$")
    ax.text(0.02, 0.92, f"mean |Δσ/σ| = {mean_abs_r*100:.4f}%", transform=ax.transAxes, va="top")

    # Row 3: absolute change
    ax = axes[2]
    ax.plot(range(1, len(d) + 1), d.numpy())
    ax.axhline(0.0)
    ax.set_title(f"{title_prefix}: absolute change")
    ax.set_xlabel(f"Rank (top-k={len(d)})")
    ax.set_ylabel(r"$\sigma^{after}-\sigma^{before}$")
    ax.text(0.02, 0.92, f"mean |Δσ| = {mean_abs_d:.6g}", transform=ax.transAxes, va="top")

    suptitle = f"Layer {layer} | Before: {model_before_name}  →  After: {model_after_name}"
    fig.suptitle(suptitle)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"[OK] saved figure -> {out_path}")


# ----------------------------
# Load ONE tensor from sharded safetensors
# ----------------------------
def pick_tensor_name(all_names):
    """
    Prefer a "real" 2D weight that is common in Qwen-style models.
    Fallback: first 2D-like name (we'll check shape after loading).
    """
    preferred = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
    for n in preferred:
        if n in all_names:
            return n
    # fallback: something that endswith ".weight" under model.layers.0
    for n in all_names:
        if n.startswith("model.layers.0.") and n.endswith(".weight"):
            return n
    # ultimate fallback
    return all_names[0]


def load_one_weight(model_dir: str, tensor_name: str | None = None):
    """
    Works with HuggingFace sharded safetensors:
      - model.safetensors.index.json maps tensor -> shard file
    Also supports single-file model.safetensors.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]  # tensor_name -> shard filename
        all_names = list(weight_map.keys())

        if tensor_name is None:
            tensor_name = pick_tensor_name(all_names)

        shard_file = weight_map[tensor_name]
        shard_path = os.path.join(model_dir, shard_file)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        # load only that shard (still may contain many tensors, but much smaller than whole model)
        sd = load_file(shard_path, device="cpu")
        if tensor_name not in sd:
            raise KeyError(f"{tensor_name} not in loaded shard {shard_file}")
        w = sd[tensor_name]
        return tensor_name, w

    if os.path.exists(single_path):
        sd = load_file(single_path, device="cpu")
        all_names = list(sd.keys())
        if tensor_name is None:
            tensor_name = pick_tensor_name(all_names)
        w = sd[tensor_name]
        return tensor_name, w

    raise FileNotFoundError(
        f"Neither {index_path} nor {single_path} exists. Is this a HF safetensors model dir?"
    )


# ----------------------------
# Core experiment: W -> SVD -> W' -> SVD, compare S vs S'
# ----------------------------
def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

@torch.no_grad()
def run_case(W_cpu: torch.Tensor, case_name: str, device: torch.device, dtype: torch.dtype, topk: int, out_dir: str):
    # move & cast
    W = W_cpu.to(device=device, dtype=dtype, non_blocking=True)
    if not W.is_contiguous():
        W = W.contiguous()

    # SVD(W)
    sync_if_cuda(device)
    t0 = time.time()
    U, S1, Vh = torch.linalg.svd(W, full_matrices=False)
    sync_if_cuda(device)
    t_svd1 = time.time() - t0

    # reconstruct W' = U diag(S1) Vh
    
    # change to fp32
    U = U.to(dtype=torch.float32)
    S1 = S1.to(dtype=torch.float32)
    Vh = Vh.to(dtype=torch.float32)
    
    sync_if_cuda(device)
    t0 = time.time()
    Wp = (U * S1.unsqueeze(0)) @ Vh
    sync_if_cuda(device)
    t_recon = time.time() - t0

    # SVD(W')
    
    # cast back to original dtype
    Wp = Wp.to(dtype=dtype)
    
    sync_if_cuda(device)
    t0 = time.time()
    Up, S2, Vhp = torch.linalg.svd(Wp, full_matrices=False)
    sync_if_cuda(device)
    t_svd2 = time.time() - t0

    # stats
    r = relative_change(S2.to(torch.float64), S1.to(torch.float64))
    d = absolute_change(S2.to(torch.float64), S1.to(torch.float64))
    mean_abs_r = float(r.abs().mean().item())
    mean_abs_d = float(d.abs().mean().item())

    print(f"\n===== {case_name} =====")
    print(f"device={device}, dtype={dtype}, shape={tuple(W.shape)}")
    print(f"time: svd(W)={t_svd1:.3f}s | recon={t_recon:.3f}s | svd(W')={t_svd2:.3f}s")
    print(f"S1 vs S2: mean|Δσ/σ|={mean_abs_r*100:.4f}% | mean|Δσ|={mean_abs_d:.6g}")

    out_path = os.path.join(out_dir, f"spectrum_{case_name.replace(' ', '_')}.png")
    plot_grid_3rows_one(
        s_before=S1,
        s_after=S2,
        model_before_name=f"{case_name}: S from W",
        model_after_name=f"{case_name}: S' from W'",
        layer=10,
        topk=topk,
        out_path=out_path,
        title_prefix=f"{case_name}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, required=True, help="HF model directory containing safetensors")
    ap.add_argument("--tensor-name", type=str, default="", help="optional exact tensor name")
    ap.add_argument("--out-dir", type=str, default="./real_weight_svd_study_out")
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--disable-tf32", action="store_true", help="disable TF32 on CUDA (recommended)")
    args = ap.parse_args()

    if args.disable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    tensor_name = args.tensor_name.strip() or None
    name, W = load_one_weight(args.model_dir, tensor_name=tensor_name)

    if W.ndim != 2:
        raise RuntimeError(f"Picked tensor is not 2D: {name} has shape {tuple(W.shape)}")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] loaded tensor: {name} | shape={tuple(W.shape)} | dtype={W.dtype}")

    # Normalize: use CPU float32 as the common source
    W_cpu = W.detach().to("cpu", torch.float32).contiguous()

    # CPU fp32
    run_case(W_cpu, "CPU_fp32", device=torch.device("cpu"), dtype=torch.float32, topk=args.topk, out_dir=args.out_dir)

    # GPU cases if available
    if torch.cuda.is_available():
        run_case(W_cpu, "GPU_fp32", device=torch.device("cuda"), dtype=torch.float32, topk=args.topk, out_dir=args.out_dir)
        run_case(W_cpu, "GPU_fp64", device=torch.device("cuda"), dtype=torch.float64, topk=args.topk, out_dir=args.out_dir)
    else:
        print("[WARN] CUDA not available; skip GPU_fp32 / GPU_fp64")

    print(f"\n[OK] all figures saved in: {os.path.abspath(args.out_dir)}")
    print(f"[INFO] tensor used: {name}")


if __name__ == "__main__":
    main()


