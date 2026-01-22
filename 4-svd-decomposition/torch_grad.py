import torch
from torch import Tensor

@torch.compile(fullgraph=True)
def zeropower_via_newtonschulz5(G: Tensor, epsilon: float = 1e-7):
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]
    X = G
    transposed = False
    if G.size(-2) > G.size(-1):
        X = X.mT
        transposed = True

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X

def _msign_real(mat: Tensor, compute_dtype: torch.dtype = torch.float64) -> Tensor:
    original_dtype = mat.dtype
    X = mat.to(dtype=compute_dtype)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    Q = U @ Vh
    return Q.to(dtype=original_dtype)

def vjp_example(f, W: Tensor, g_prime: Tensor):
    """
    给定 W' = f(W) 和上游梯度 g' = dL/dW'，计算 g = dL/dW
    """
    W_prime = f(W)  # 这一步要在 autograd 图里
    (gW,) = torch.autograd.grad(
        outputs=W_prime,
        inputs=W,
        grad_outputs=g_prime,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    return gW

def main(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32):
    torch.manual_seed(0)

    W = torch.randn(128, 64, device=device, dtype=dtype, requires_grad=True)

    # 你在优化器里“已有的”上游梯度：dL/dW'
    # （这里用随机张量模拟；真实情况就是你手里那份 grad）
    g_prime = torch.randn_like(W)

    # Case A: W' = NS(W)
    gW_ns = vjp_example(zeropower_via_newtonschulz5, W, g_prime)
    print("NS:   gW finite:", torch.isfinite(gW_ns).all().item(), "norm:", float(gW_ns.norm()))

    # Case B: W' = msign(W)
    gW_ms = vjp_example(_msign_real, W, g_prime)
    print("msign gW finite:", torch.isfinite(gW_ms).all().item(), "norm:", float(gW_ms.norm()))

if __name__ == "__main__":
    main()
