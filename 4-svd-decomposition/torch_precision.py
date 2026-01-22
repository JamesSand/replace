

#!/usr/bin/env python3
import os
import torch
import matplotlib.pyplot as plt
from torch import Tensor


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



def _msign_real(mat: Tensor, compute_dtype: torch.dtype) -> Tensor:
    """
    "True" msign via SVD:
        mat = U @ diag(S) @ V^T
        msign(mat) = U @ V^T

    Works for 2D (m,n) or batched 3D (..., m, n).
    Returns float32 for stability (like your _msign).
    """
    original_dtype = mat.dtype
    # Do SVD in compute_dtype for stability
    X = mat.to(dtype=compute_dtype)

    # full_matrices=False gives:
    # U: (..., m, k), Vh: (..., k, n), k=min(m,n)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    # msign = U @ V^T  == U @ Vh
    Q = U @ Vh
    # Convert back to original dtype
    result = Q.to(dtype=original_dtype)
    
    # Free X and S immediately as we don't need them
    del X
    del S
    del U
    del Vh
    del Q
    
    return result


# ----------------------------
# Main experiment (your 1~6 steps)
# ----------------------------
def main():
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the fp64 GPU SVD step.")

    device = "cuda"
    m, n = 2048, 2048
    k = min(m, n)
    lr = 1e-3
    topk = -1  # plot top-k singular values; set -1 for all

    # (1) fp32 W
    W = torch.randn(m, n, device=device, dtype=torch.float32)

    # (2) fp64 GPU SVD -> U,S,V
    W64 = W.to(torch.float64)
    U64, S64, Vh64 = torch.linalg.svd(W64, full_matrices=False)
    V64 = Vh64.transpose(-2, -1).contiguous()

    # (3) U,V -> fp32, take one gradient step -> U',V'
    U32 = U64.to(torch.float32)
    V32 = V64.to(torch.float32)

    # random "gradient" (replace with real gradient if you want)
    gU = torch.randn_like(U32)
    gV = torch.randn_like(V32)

    U32_p = U32 - lr * gU
    V32_p = V32 - lr * gV
    
    U32_p = _msign_real(U32_p, torch.float64)
    V32_p = _msign_real(V32_p, torch.float64)

    S_32 = S64.to(torch.float32)
    W64_p = (U32_p * S_32.unsqueeze(0)) @ V32_p.transpose(-2, -1)

    # (5) W' -> fp32 CPU SVD -> U'',S'',V''
    Wcpu = W64_p.to(torch.float32).cpu()
    _, S_cpu2, _ = torch.linalg.svd(Wcpu, full_matrices=False)

    # (6) compare S (from step 2) vs S'' (from step 5) + plot
    S_ref = S64.detach().cpu()        # before
    S_new = S_cpu2.detach().cpu()     # after

    abs_diff = (S_new.to(torch.float64) - S_ref.to(torch.float64)).abs()
    rel_diff = abs_diff / (S_ref.to(torch.float64).abs() + 1e-12)

    print(f"m,n={m},{n} k={k} lr={lr}")
    print(f"max|ΔS|={abs_diff.max().item():.6e}, mean|ΔS|={abs_diff.mean().item():.6e}")
    print(f"max rel|ΔS|={rel_diff.max().item():.6e}, mean rel|ΔS|={rel_diff.mean().item():.6e}")

    out_path = "svd_spectrum_change.png"
    plot_grid_3rows_one(
        s_before=S_ref,
        s_after=S_new,
        model_before_name="GPU fp64 SVD on W (S)",
        model_after_name="CPU fp32 SVD on W' (S'')",
        layer=0,
        topk=topk,
        out_path=out_path,
        title_prefix="S vs S''",
    )


if __name__ == "__main__":
    main()




