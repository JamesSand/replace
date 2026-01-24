# ns_precision_sim.py
import os
import torch
from torch import Tensor
import matplotlib.pyplot as plt


# ----------------------------
# Newton–Schulz (given)
# ----------------------------
try:
    _compile = torch.compile  # torch 2.x
except Exception:
    _compile = None

def _zeropower_via_newtonschulz5(G: Tensor, epsilon: float = 1e-7):
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # (note: this is Frobenius norm scaling; matches your snippet)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

if _compile is not None:
    zeropower_via_newtonschulz5 = _compile(_zeropower_via_newtonschulz5, fullgraph=True)
else:
    zeropower_via_newtonschulz5 = _zeropower_via_newtonschulz5


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
# Plot: sigma / rel / abs
# ----------------------------
def relative_change(after: torch.Tensor, before: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (after - before) / (before.abs() + eps)

def absolute_change(after: torch.Tensor, before: torch.Tensor) -> torch.Tensor:
    return after - before

def plot_grid_3rows_one(
    s_before: torch.Tensor,
    s_after: torch.Tensor,
    topk: int,
    out_path: str,
    title: str,
):
    def _topk(x: torch.Tensor) -> torch.Tensor:
        if topk <= 0:
            return x
        return x[: min(int(x.numel()), topk)]

    b = _topk(s_before.detach().to("cpu", torch.float64))
    a = _topk(s_after.detach().to("cpu", torch.float64))
    r = relative_change(a, b)
    d = absolute_change(a, b)

    mean_abs_r = float(r.abs().mean().item())
    mean_abs_d = float(d.abs().mean().item())

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(range(1, len(b) + 1), b.numpy(), label="S (from W)")
    ax.plot(range(1, len(a) + 1), a.numpy(), label="S' (from W')")
    ax.set_title("singular values")
    ax.set_ylabel(r"$\sigma_i$")
    ax.legend()

    ax = axes[1]
    ax.plot(range(1, len(r) + 1), r.numpy())
    ax.axhline(0.0)
    ax.set_title("relative change")
    ax.set_ylabel(r"$(\sigma' - \sigma)/\sigma$")
    ax.text(0.02, 0.92, f"mean |Δσ/σ| = {mean_abs_r*100:.4f}%", transform=ax.transAxes, va="top")

    ax = axes[2]
    ax.plot(range(1, len(d) + 1), d.numpy())
    ax.axhline(0.0)
    ax.set_title("absolute change")
    ax.set_xlabel(f"Rank (top-k={len(d)})")
    ax.set_ylabel(r"$\sigma' - \sigma$")
    ax.text(0.02, 0.92, f"mean |Δσ| = {mean_abs_d:.6g}", transform=ax.transAxes, va="top")

    fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"[OK] saved figure -> {out_path}")


# ----------------------------
# Simulate your 6-step pipeline
# ----------------------------
def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # matrix size (keep moderate; change if you want)
    m, n = 2048, 2048
    lr = 1e-3
    topk = -1
    out_path = "./ns_precision_sim.png"

    # 1) W -> U S V (fp64)
    W = torch.randn(m, n, device=device, dtype=torch.float64)
    U64, S64, Vh64 = torch.linalg.svd(W, full_matrices=False)  # Vh = V^T
    # We'll treat "V" as Vh^T
    V64 = Vh64.mT

    # 2) U, V <- SGD (fp32)
    U32 = U64.to(torch.float32)
    V32 = V64.to(torch.float32)

    # mock gradients (fp32) for U and V
    Gu = torch.randn_like(U32)
    Gv = torch.randn_like(V32)

    U32 = U32 - lr * Gu
    V32 = V32 - lr * Gv

    # 3) U <- NS(U), V <- NS(V) (fp32)
    # to fp64
    U32 = U32.to(torch.float64)
    V32 = V32.to(torch.float64)

    # Start: here you choose which algo to perform projection
    # U32o = zeropower_via_newtonschulz5(U32)
    # V32o = zeropower_via_newtonschulz5(V32)
    
    U32o = _msign_real(U32, compute_dtype=torch.float64)
    V32o = _msign_real(V32, compute_dtype=torch.float64)
    # End: here you choose which algo to perform projection

    # calcualte the f norm of U and V, and print them out
    exptected_f_norm = m ** 0.5  # since U and V should be orthogonal matrices
    fnorm_U = torch.norm(U32o, p='fro')
    fnorm_V = torch.norm(V32o, p='fro')
    print('-' * 50)
    print(f"Expected Frobenius norm (sqrt of matrix size): {exptected_f_norm:.6f}")
    print(f"Frobenius norm of U after projection: {fnorm_U.item()}")
    print(f"Frobenius norm of V after projection: {fnorm_V.item()}")
    print('-' * 50)
    print(f"diff U F norm and exptected F norm {exptected_f_norm - fnorm_U.item()}")
    print(f"diff V F norm and exptected F norm {exptected_f_norm - fnorm_V.item()}")
    breakpoint()
    print()
    
    # back to fp32
    U32o = U32o.to(torch.float32)
    V32o = V32o.to(torch.float32)
    

    # 4) W' <- U S V (fp32)
    # use S from step1; cast to fp32
    S32 = S64.to(torch.float32)
    # W' = U diag(S) V^T ; we have V (not V^T)
    Wp32 = (U32o * S32.unsqueeze(0)) @ V32o.mT  # fp32

    # 5) W' -> U' S' V' (fp64)
    Wp64 = Wp32.to(torch.float64)
    Up64, Sp64, Vhp64 = torch.linalg.svd(Wp64, full_matrices=False)

    # 6) compare S (fp32) vs S' (fp32)
    Sp32 = Sp64.to(torch.float32)

    title = f"NS precision sim | m=n={m} | device={device.type} | lr={lr}"
    plot_grid_3rows_one(S32, Sp32, topk=topk, out_path=out_path, title=title)


if __name__ == "__main__":
    main()






