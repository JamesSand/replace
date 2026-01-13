# svsm_one_weight_type.py
# Compute SVSM for ONE weight type (hardcoded suffix), across ALL layer indices and ALL sigma dimensions.
#
# SVSM definition from your screenshot:
#   For each layer i (same matrix type W^i in two models A=before, B=after):
#       Div^(i) = [ sigma_B,1^(i) / sigma_A,1^(i), ..., sigma_B,n^(i) / sigma_A,n^(i) ]^T
#   Then SVSM is stacking Div^(i) across layers i.
#
# Here we use your cache files produced by svd_sigma_blend_cache_merge.py:
#   each layer file contains S_before and S_after (and also U_after/Vh_after, but SVSM only needs S).
#
# Output:
#   - svsm matrix saved to .pt
#   - 2D heatmap png
#   - 3D surface png (paper-like)

import re
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def safe_key_to_filename(key: str) -> str:
    return key.replace(".", "__").replace("/", "_slash_")

# ===================== 你改这几行就行 =====================
CACHE_DIR = Path("/fast/sliu/zhizhou/workspace/rotation-project/replace/2-blend/llama-adam-cache")      # 你的 cache 根目录（里面应有 svd_cache_meta.json 和 layers/）
# WEIGHT_SUFFIX = "self_attn.q_proj.weight"     # 写死：你要分析的“某一种权重类型”
OUT_DIR = Path("svsm_outputs_one_type")       # 输出目录

# 画图相关（可不改）
# EPS = 1e-12                                   # 防止除 0
EPS = 0.0                                   # 防止除 0
# CLIP_PERCENTILE = (1.0, 99.0)                 # 为了画图更像论文，按分位数裁剪色阶；设为 None 表示不裁剪
CLIP_PERCENTILE = None                 # 为了画图更像论文，按分位数裁剪色阶；设为 None 表示不裁剪
SURFACE_TARGET_SIG_SAMPLES = 300              # 3D surface y 轴采样密度（太大就很慢）；不影响“计算全维度”
# ==========================================================

def layer_index_from_key(k: str):
    """
    尽量兼容不同模型命名：
      - model.layers.{i}.xxx
      - layers.{i}.xxx
      - transformer.h.{i}.xxx (GPT类)
    """
    m = re.search(r"(?:model\.)?layers\.(\d+)\.", k)
    if m:
        return int(m.group(1))
    m = re.search(r"transformer\.h\.(\d+)\.", k)
    if m:
        return int(m.group(1))
    return None

def load_cached_keys(meta_path: Path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    keys = meta.get("cached_2d_keys", [])
    if not keys:
        raise RuntimeError(f"No cached_2d_keys found in {meta_path}")
    return keys

# core function for computing SVSM matrix
def compute_svsm_matrix(cache_dir: Path, weight_suffix: str):
    meta_path = cache_dir / "svd_cache_meta.json"
    layers_dir = cache_dir / "layers"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta: {meta_path}")
    if not layers_dir.exists():
        raise FileNotFoundError(f"Missing layers dir: {layers_dir}")

    all_keys = load_cached_keys(meta_path)
    target_keys = [k for k in all_keys if k.endswith(weight_suffix)]

    if not target_keys:
        raise RuntimeError(
            f"No keys end with suffix '{weight_suffix}'. "
            f"Check suffix or cache content."
        )

    rows = []
    for k in target_keys:
        idx = layer_index_from_key(k)
        if idx is None:
            continue

        pt_path = layers_dir / (safe_key_to_filename(k) + ".pt")
        if not pt_path.exists():
            # cache 里缺文件就跳过
            continue

        pack = torch.load(pt_path, map_location="cpu")
        S_before = pack["S_before"].to(torch.float32)
        S_after = pack["S_after"].to(torch.float32)

        # ratio = sigma_after / sigma_before
        ratio = (S_after + EPS) / (S_before + EPS)

        rows.append((idx, ratio))

        # 释放（避免累积内存）
        del pack, S_before, S_after

    if not rows:
        raise RuntimeError("No valid (layer_idx, sigma_ratio) rows collected.")

    rows.sort(key=lambda x: x[0])
    layer_indices = [i for i, _ in rows]
    num_layers = max(layer_indices) + 1
    sig_dim = max(r.numel() for _, r in rows)

    svsm = torch.full((num_layers, sig_dim), float("nan"), dtype=torch.float32)
    for i, r in rows:
        svsm[i, : r.numel()] = r

    return svsm  # shape: [num_layers, sig_dim]

def percentile_clip(arr: np.ndarray, q_low: float, q_high: float):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return arr, None, None
    vmin = np.percentile(valid, q_low)
    vmax = np.percentile(valid, q_high)
    clipped = np.clip(arr, vmin, vmax)
    return clipped, vmin, vmax

def plot_heatmap(Z: np.ndarray, out_path: Path, title: str, vmin=None, vmax=None):
    # Z expected shape: (sig_dim, num_layers)
    plt.figure(figsize=(7.0, 4.0), dpi=220)
    im = plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        # cmap="jet",  # 论文里那种偏“彩虹色”的观感更接近 jet
    )
    plt.title(title)
    plt.xlabel("layer index")
    plt.ylabel("sig dimension")
    cbar = plt.colorbar(im)
    cbar.set_label("sigma_after / sigma_before")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_surface(Z: np.ndarray, out_path: Path, title: str, vmin=None, vmax=None):
    # Z expected shape: (sig_dim, num_layers)
    sig_dim, num_layers = Z.shape

    # 3D surface 太密会慢：只对“绘图采样”，不影响你“分析所有维度”的计算
    if sig_dim > SURFACE_TARGET_SIG_SAMPLES:
        step = max(1, sig_dim // SURFACE_TARGET_SIG_SAMPLES)
        Zp = Z[::step, :]
        y = np.arange(0, sig_dim, step)
    else:
        Zp = Z
        y = np.arange(sig_dim)

    x = np.arange(num_layers)
    X, Y = np.meshgrid(x, y)  # shapes: (len(y), num_layers)

    # NaN 处理：用 1.0 填（“无变化”中性值），避免 plot_surface 报错
    Zp = np.nan_to_num(Zp, nan=1.0)

    fig = plt.figure(figsize=(7.2, 5.4), dpi=220)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Zp,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xlabel("layer index")
    ax.set_ylabel("sig dimension")
    ax.set_zlabel("ratio")

    # 视角调得接近论文那种
    ax.view_init(elev=28, azim=-55)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label("sigma_after / sigma_before")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main(WEIGHT_SUFFIX):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    svsm = compute_svsm_matrix(CACHE_DIR, WEIGHT_SUFFIX)  # [num_layers, sig_dim]
    # 为了符合你说的轴定义：x=layer index, y=sig dimension
    Z = svsm.T.numpy()  # -> (sig_dim, num_layers)

    # 保存原始矩阵（全维度、全层）
    out_pt = OUT_DIR / f"svsm_{WEIGHT_SUFFIX.replace('.', '_')}.pt"
    torch.save({"weight_suffix": WEIGHT_SUFFIX, "svsm": svsm}, out_pt)
    print(f"[saved] raw SVSM matrix -> {out_pt}  shape={tuple(svsm.shape)}")

    # 画图裁剪（只影响可视化，不影响保存的原始矩阵）
    if CLIP_PERCENTILE is not None:
        Z_plot, vmin, vmax = percentile_clip(Z, CLIP_PERCENTILE[0], CLIP_PERCENTILE[1])
    else:
        Z_plot, vmin, vmax = Z, None, None

    title = f"SVSM: {WEIGHT_SUFFIX}"

    out_heat = OUT_DIR / f"svsm_{WEIGHT_SUFFIX.replace('.', '_')}_heatmap.png"
    plot_heatmap(Z_plot, out_heat, title, vmin=vmin, vmax=vmax)
    print(f"[saved] heatmap -> {out_heat}")

    out_surf = OUT_DIR / f"svsm_{WEIGHT_SUFFIX.replace('.', '_')}_surface.png"
    plot_surface(Z_plot, out_surf, title, vmin=vmin, vmax=vmax)
    print(f"[saved] surface -> {out_surf}")

if __name__ == "__main__":
    
    weight_suffix_list = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.down_proj.weight",
        "mlp.up_proj.weight",
        "mlp.gate_proj.weight",
    ]
    
    for WEIGHT_SUFFIX in weight_suffix_list:
        print(f"\n=== Processing weight type: {WEIGHT_SUFFIX} ===")
        main(WEIGHT_SUFFIX)


