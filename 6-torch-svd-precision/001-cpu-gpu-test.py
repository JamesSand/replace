

#!/usr/bin/env python3
import os
import time
import math
import argparse

import torch
import matplotlib.pyplot as plt


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
# Construct a hard matrix:
# W = U diag(S) V^T with log-spaced S (high condition number)
# ----------------------------
@torch.no_grad()
def make_hard_matrix(n: int, cond_exp: float, device: torch.device, dtype: torch.dtype, seed: int = 0):
    """
    cond_exp controls condition number ~ 10^(cond_exp).
    S_i = 10^linspace(0, -cond_exp, n)
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # Build orthogonal U,V on CPU in float64 for stability
    A = torch.randn(n, n, generator=g, dtype=torch.float64, device="cpu")
    B = torch.randn(n, n, generator=g, dtype=torch.float64, device="cpu")
    U, _ = torch.linalg.qr(A)  # (n,n)
    V, _ = torch.linalg.qr(B)  # (n,n)

    # log-spaced singular values
    exps = torch.linspace(0.0, -cond_exp, steps=n, dtype=torch.float64, device="cpu")
    S = (10.0 ** exps)  # (n,)

    # Move to target device/dtype
    U = U.to(device=device, dtype=dtype)
    V = V.to(device=device, dtype=dtype)
    S = S.to(device=device, dtype=dtype)

    # W = U diag(S) V^T
    W = (U * S.unsqueeze(0)) @ V.transpose(-2, -1)
    return W, S  # return "ground-truth" S used to construct W


def sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


@torch.no_grad()
def run_case(
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    n: int,
    cond_exp: float,
    out_dir: str,
    topk: int,
    seed: int,
    disable_tf32: bool,
):
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = not disable_tf32
        torch.backends.cudnn.allow_tf32 = not disable_tf32

    print(f"\n===== {name} | device={device} dtype={dtype} tf32={'OFF' if disable_tf32 else 'ON'} =====")
    W, S_true = make_hard_matrix(n=n, cond_exp=cond_exp, device=device, dtype=dtype, seed=seed)

    # Step 1: W -> U,S,V
    sync_if_cuda(device)
    t0 = time.time()
    U, S1, Vh = torch.linalg.svd(W, full_matrices=False)
    sync_if_cuda(device)
    t_svd1 = time.time() - t0

    # Step 2: W' <- U S V
    t0 = time.time()
    Wp = (U * S1.unsqueeze(0)) @ Vh
    sync_if_cuda(device)
    t_recon = time.time() - t0

    # Step 3: W' -> U' S' V'
    sync_if_cuda(device)
    t0 = time.time()
    Up, S2, Vhp = torch.linalg.svd(Wp, full_matrices=False)
    sync_if_cuda(device)
    t_svd2 = time.time() - t0

    # Compare S1 vs S2 (your slide's "S = S'")
    # Also compare to S_true (constructed spectrum) for sanity
    r12 = relative_change(S2.to(torch.float64), S1.to(torch.float64))
    d12 = absolute_change(S2.to(torch.float64), S1.to(torch.float64))
    mean_abs_r12 = float(r12.abs().mean().item())
    mean_abs_d12 = float(d12.abs().mean().item())

    rt = relative_change(S1.to(torch.float64), S_true.to(torch.float64))
    mean_abs_rt = float(rt.abs().mean().item())

    print(f"timing: svd(W)={t_svd1:.3f}s | recon={t_recon:.3f}s | svd(W')={t_svd2:.3f}s")
    print(f"S1 vs S2: mean |Δσ/σ| = {mean_abs_r12*100:.4f}% | mean |Δσ| = {mean_abs_d12:.6g}")
    print(f"S_true vs S1 sanity: mean |Δσ/σ| = {mean_abs_rt*100:.4f}%")

    # Plot (use S1 as "before", S2 as "after")
    tag = f"{device.type}_{str(dtype).replace('torch.', '')}"
    fig_path = os.path.join(out_dir, f"spectrum_{name.replace(' ', '_')}_{tag}.png")
    plot_grid_3rows_one(
        s_before=S1,
        s_after=S2,
        model_before_name=f"{name} (S from W)",
        model_after_name=f"{name} (S' from W')",
        layer=0,
        topk=topk,
        out_path=fig_path,
        title_prefix=f"{name} | n={n} cond~1e{cond_exp:g}",
    )

    return {
        "name": name,
        "device": str(device),
        "dtype": str(dtype),
        "tf32_disabled": disable_tf32,
        "t_svd1": t_svd1,
        "t_recon": t_recon,
        "t_svd2": t_svd2,
        "mean_abs_rel_S1_S2": mean_abs_r12,
        "mean_abs_abs_S1_S2": mean_abs_d12,
        "mean_abs_rel_Strue_S1": mean_abs_rt,
        "fig": fig_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024, help="matrix size (n x n)")
    parser.add_argument("--cond-exp", type=float, default=10.0, help="condition number approx 10^(cond-exp)")
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default="./svd_minproof_out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-tf32", action="store_true", help="disable TF32 on CUDA (recommended for analysis)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = []

    # CPU cases
    results.append(
        run_case(
            name="CPU fp32",
            device=torch.device("cpu"),
            dtype=torch.float32,
            n=args.n,
            cond_exp=args.cond_exp,
            out_dir=args.out_dir,
            topk=args.topk,
            seed=args.seed,
            disable_tf32=args.disable_tf32,
        )
    )
    # results.append(
    #     run_case(
    #         name="CPU fp64",
    #         device=torch.device("cpu"),
    #         dtype=torch.float64,
    #         n=args.n,
    #         cond_exp=args.cond_exp,
    #         out_dir=args.out_dir,
    #         topk=args.topk,
    #         seed=args.seed,
    #         disable_tf32=args.disable_tf32,
    #     )
    # )

    # GPU cases (if available)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        results.append(
            run_case(
                name="GPU fp32",
                device=dev,
                dtype=torch.float32,
                n=args.n,
                cond_exp=args.cond_exp,
                out_dir=args.out_dir,
                topk=args.topk,
                seed=args.seed,
                disable_tf32=args.disable_tf32,
            )
        )
        results.append(
            run_case(
                name="GPU fp64",
                device=dev,
                dtype=torch.float64,
                n=args.n,
                cond_exp=args.cond_exp,
                out_dir=args.out_dir,
                topk=args.topk,
                seed=args.seed,
                disable_tf32=args.disable_tf32,
            )
        )
    else:
        print("\n[WARN] CUDA not available: skip GPU fp32/fp64 cases.")

    # Print a compact summary
    print("\n================ Summary (S1 vs S2) ================")
    for r in results:
        rel_pct = r["mean_abs_rel_S1_S2"] * 100.0
        print(
            f"{r['name']:<8} | {r['dtype']:<13} | "
            f"svd1={r['t_svd1']:.3f}s svd2={r['t_svd2']:.3f}s | "
            f"mean|Δσ/σ|={rel_pct:.4f}% | fig={r['fig']}"
        )
    print("====================================================\n")
    print(f"[OK] all figures saved under: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()





