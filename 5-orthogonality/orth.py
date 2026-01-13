# orthogonality_uv_one_layer.py
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def safe_key_to_filename(key: str) -> str:
    return key.replace(".", "__").replace("/", "_slash_")

# ====== 你只需要改这几行 ======
LAYER_KEY = "model.layers.0.mlp.down_proj.weight"   # <-- hardcode 你要分析的 layer name
CACHE_DIR = Path("/fast/sliu/zhizhou/workspace/rotation-project/replace/2-blend/llama-adam-cache/layers")           # <-- before 的 per-layer pt 存放目录
# AFTER_DIR  = Path("svd_uv_after/layers")            # <-- after  的 per-layer pt 存放目录
# TOP_R = 24                                          # <-- 画前 r 个奇异向量(像你图里 0~23)
# TOP_R = 99999                                          # <-- 画前 r 个奇异向量(像你图里 0~23)

# =================================

def load_UV(pt_path: Path):
    d = torch.load(pt_path, map_location="cpu")
    
    
    
    U_after = d.get("U_after", None)
    Vh_after = d.get("Vh_after", None)
    
    U_before = d.get("U_before", None)
    Vh_before = d.get("Vh_before", None)
    
    V_after = Vh_after.T
    V_before = Vh_before.T
    
    return (U_before.to(torch.float32), V_before.to(torch.float32),
            U_after.to(torch.float32), V_after.to(torch.float32))

def main():
    fn = safe_key_to_filename(LAYER_KEY) + ".pt"
    cache_pt = CACHE_DIR / fn
    # after_pt  = AFTER_DIR / fn
    
    U_before, V_before, U_after, V_after = load_UV(cache_pt)
    
    
    
    total_r = min(U_before.shape[1], U_after.shape[1], V_before.shape[1], V_after.shape[1])
    
    middle = total_r // 2
    
    test_list = [
        (0, 50), 
        (0, 9999999), 
        (middle - 25, middle + 25),
        (-50, 0)
    ]
    
    for start_r, end_r in test_list:
        if start_r < 0:
            start_r = total_r + start_r
            end_r = total_r + end_r
            
        start_r = max(0, start_r)
        end_r = min(total_r, end_r)
        
        process(U_before, V_before, U_after, V_after, start_r, end_r)
    
    
    
def process(U_before, V_before, U_after, V_after, start_r=None, end_r=None):
    
    
    if start_r is None:
        start_r = 0
    if end_r is None:
        end_r = min(U_before.shape[1], U_after.shape[1], V_before.shape[1], V_after.shape[1])
        
    r = end_r - start_r
    print(f"Analyzing orthogonality of top r={r} singular vectors (from index {start_r} to {end_r})")
    
    U_before, U_after = U_before[:, start_r:end_r], U_after[:, start_r:end_r]
    V_before, V_after = V_before[:, start_r:end_r], V_after[:, start_r:end_r]

    # U_before Q1 =  U_after, 
    # Q2 = V_before^T V_after
    Q1 = U_before.T @ U_after          # (r,r)
    Q2 = V_before.T @ V_after          # (r,r)
    
    # U_before Q_1 = U_middle
    # U_middle Q_2 = U_after

    # Eq.(7): I_orth = Q1^T Q2
    I_orth = Q1.T @ Q2        # (r,r)
    
    # print(I_orth)
    
    # # print diagonal values
    # diag_values = I_orth.diagonal().cpu().numpy()
    # print("Diagonal values of I_orth:")
    # for i, val in enumerate(diag_values):
    #     print(f"  {start_r + i:4d}: {val:.6f}")
    
    # breakpoint()
    # print()
    
    # ===== Frobenius distance to identity =====
    I = torch.eye(r, dtype=I_orth.dtype)
    
    diff_I = I_orth - I
    
    dF_raw = torch.linalg.norm(diff_I, ord="fro").item()
    dF_norm = dF_raw / (r ** 0.5)
    print(f"Frobenius norm (raw): {dF_raw:.6f}")
    print(f"Frobenius norm (normalized): {dF_norm:.6f}")
    
    step = max(1, r // 8)
    # ===== plot: I_orth (left) vs diff_I (right) =====
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), dpi=200)

    im0 = axes[0].imshow(I_orth)
    axes[0].set_title("I_orth")
    
    ticks = list(range(0, r, step))
    labels = [str(t + start_r) for t in ticks]
    axes[0].set_xticks(ticks); axes[0].set_xticklabels(labels)
    axes[0].set_yticks(ticks); axes[0].set_yticklabels(labels)
    
    fig.colorbar(im0, ax=axes[0])
    
    


    im1 = axes[1].imshow(diff_I)
    axes[1].set_title("diff_I = I_orth - I")
    
    ticks = list(range(0, r, step))
    labels = [str(t + start_r) for t in ticks]
    axes[1].set_xticks(ticks); axes[1].set_xticklabels(labels)
    axes[1].set_yticks(ticks); axes[1].set_yticklabels(labels
                                                       )
    fig.colorbar(im1, ax=axes[1])
    
    subtitle = f"Start {start_r} to {end_r}\nFrobenius raw={dF_raw:.6f}\n F norm / r ={dF_norm:.6f}"

    fig.suptitle(subtitle, y=0.93)
    fig.tight_layout()
    
    # output figrue to dir named orth_fig
    OUT_DIR = Path("orth_fig_uv_one_layer")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    OUT_PNG = OUT_DIR / f"orthogonality_uv_{start_r}_{end_r}.png"
    
    fig.savefig(OUT_PNG)
    print(f"[saved] {OUT_PNG}")

if __name__ == "__main__":
    
    test_list = [
        (0, 50),
        (0, 48),
        (0, 96),
        (0, 192),
    ]
    
    main()




