

#!/usr/bin/env python3
# cond_hist_qwen.py
import os
import argparse
import math
import torch
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoModelForCausalLM


# ====== HARD CODE (same as your script) ======
MODEL_PATH = "/ssd2/zhizhou/workspace/rotation-project/Lucky_RL/szz_dion_debug/models/Qwen/Qwen3-1.7B-Base"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Qwen3-1.7B-Base"
# ============================================


@torch.no_grad()
def pick_hidden_weights_like_optimizer(model):
    """
    Exactly mirror your build_optimizer() selection for hidden_weights:
      - exclude embed_tokens / lm_head
      - include params under "model.layers." with p.ndim >= 2
    """
    hidden_weights = []
    seen = set()  # avoid duplicates if any tied/shared params

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embed_tokens" in name or "lm_head" in name:
            continue
        if "model.layers." in name and p.ndim >= 2:
            if id(p) not in seen:
                hidden_weights.append((name, p))
                seen.add(id(p))
    return hidden_weights


@torch.no_grad()
def cond_number_svdvals(mat: torch.Tensor, eps: float = 1e-12) -> float:
    """
    cond = smax / smin using svdvals.
    Use float64 for svd if input is fp32 to reduce numerical issues.
    """
    x = mat
    # If you want faster but less stable: comment next line
    if x.dtype != torch.float64:
        x = x.to(torch.float64)
    s = torch.linalg.svdvals(x)  # descending
    smax = float(s[0].item())
    smin = float(s[-1].item())
    smin = max(smin, eps)
    return smax / smin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp64"])
    ap.add_argument("--attn-impl", type=str, default="eager")
    ap.add_argument("--max-mats", type=int, default=-1, help="limit number of matrices (for speed). -1 = all")
    ap.add_argument("--out", type=str, default="./cond_hist_qwen3_1p7b.png")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--plot-log10", action="store_true", help="plot log10(cond) histogram (recommended)")
    ap.add_argument("--disable-tf32", action="store_true", help="disable TF32 on CUDA")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available, use --device cpu")

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "fp32" else torch.float64

    if device.type == "cuda":
        if args.disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    print(f"[INFO] loading model from: {MODEL_PATH}")
    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, attn_implementation=args.attn_impl)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=cfg,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    mats = pick_hidden_weights_like_optimizer(model)
    if args.max_mats > 0:
        mats = mats[: args.max_mats]

    print(f"[INFO] selected hidden 2D matrices (Muon group): {len(mats)}")

    conds = []
    meta = []  # (name, shape, cond)

    for i, (name, p) in enumerate(mats):
        x = p.detach()
        # Some params can be non-contiguous; make contiguous for SVD
        x = x.to(device)
        if not x.is_contiguous():
            x = x.contiguous()

        kappa = cond_number_svdvals(x)
        conds.append(kappa)
        meta.append((name, tuple(x.shape), kappa))

        if (i + 1) % 10 == 0:
            print(f"[INFO] done {i+1}/{len(mats)} ... last cond={kappa:.3e}")

    # stats
    meta_sorted = sorted(meta, key=lambda t: t[2])
    min_item = meta_sorted[0]
    max_item = meta_sorted[-1]

    conds_t = torch.tensor(conds, dtype=torch.float64)
    print("\n==================== STATS ====================")
    print(f"count = {len(conds)}")
    print(f"min   = {min_item[2]:.6e} | {min_item[0]} | shape={min_item[1]}")
    print(f"max   = {max_item[2]:.6e} | {max_item[0]} | shape={max_item[1]}")
    print(f"mean  = {float(conds_t.mean().item()):.6e}")
    print(f"median= {float(conds_t.median().item()):.6e}")
    print("==============================================\n")

    # plot
    vals = conds_t.numpy()
    if args.plot_log10:
        vals = [math.log10(max(v, 1e-300)) for v in vals]
        xlabel = "log10(condition number)"
        title = "Condition Number Distribution (log10) | Qwen3-1.7B hidden 2D weights"
    else:
        xlabel = "condition number"
        title = "Condition Number Distribution | Qwen3-1.7B hidden 2D weights"

    plt.figure(figsize=(10, 6))
    plt.hist(vals, bins=args.bins)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print(f"[OK] saved histogram -> {args.out}")


if __name__ == "__main__":
    main()


