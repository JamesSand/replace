# svsm_row_figure.py
# Plot SVSM 3D surfaces for multiple weight types in ONE horizontal figure (1 x N),
# sharing the same color scale and one colorbar on the right.

import re
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def safe_key_to_filename(key: str) -> str:
    return key.replace(".", "__").replace("/", "_slash_")

CACHE_DIR = Path("/fast/sliu/zhizhou/workspace/rotation-project/replace/2-blend/llama-adam-cache")
OUT_DIR = Path("svsm_outputs_row")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12
SURFACE_TARGET_SIG_SAMPLES = 250  # 画图采样密度（只影响渲染速度）

weight_suffix_list = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
]

def layer_index_from_key(k: str):
    m = re.search(r"(?:model\.)?layers\.(\d+)\.", k)
    if m: return int(m.group(1))
    m = re.search(r"transformer\.h\.(\d+)\.", k)
    if m: return int(m.group(1))
    return None

def load_cached_keys(meta_path: Path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    keys = meta.get("cached_2d_keys", [])
    if not keys:
        raise RuntimeError(f"No cached_2d_keys found in {meta_path}")
    return keys

def compute_svsm_matrix(cache_dir: Path, weight_suffix: str):
    meta_path = cache_dir / "svd_cache_meta.json"
    layers_dir = cache_dir / "layers"
    all_keys = load_cached_keys(meta_path)
    target_keys = [k for k in all_keys if k.endswith(weight_suffix)]
    if not target_keys:
        raise RuntimeError(f"No keys end with suffix '{weight_suffix}'")

    rows = []
    for k in target_keys:
        idx = layer_index_from_key(k)
        if idx is None:
            continue
        pt_path = layers_dir / (safe_key_to_filename(k) + ".pt")
        if not pt_path.exists():
            continue

        pack = torch.load(pt_path, map_location="cpu")
        S_before = pack["S_before"].to(torch.float32)
        S_after  = pack["S_after"].to(torch.float32)
        ratio = (S_after + EPS) / (S_before + EPS)
        rows.append((idx, ratio))

        del pack, S_before, S_after

    rows.sort(key=lambda x: x[0])
    layer_indices = [i for i, _ in rows]
    num_layers = max(layer_indices) + 1
    sig_dim = max(r.numel() for _, r in rows)

    svsm = torch.full((num_layers, sig_dim), float("nan"), dtype=torch.float32)
    for i, r in rows:
        svsm[i, : r.numel()] = r
    return svsm  # [num_layers, sig_dim]

def downsample_for_surface(Z: np.ndarray, target_sig_samples: int):
    # Z: (sig_dim, num_layers)
    sig_dim, num_layers = Z.shape
    if sig_dim <= target_sig_samples:
        y = np.arange(sig_dim)
        return Z, y
    step = max(1, sig_dim // target_sig_samples)
    y = np.arange(0, sig_dim, step)
    return Z[::step, :], y

def plot_surfaces_row(cache_dir: Path, suffixes, out_png: Path, title="SVSMs"):
    # 1) compute all Z's first (so we can share vmin/vmax)
    Z_list = []
    for suf in suffixes:
        svsm = compute_svsm_matrix(cache_dir, suf)   # [L, D]
        Z = svsm.T.numpy()                           # (D, L)
        # 填 NaN（中性值 ratio=1）
        Z = np.nan_to_num(Z, nan=1.0)
        
        # Z = Z - 1.0 # 中心化到 0 附近，更好看一些
        
        Z_list.append(Z)

    # 2) global vmin/vmax for shared colorbar with percentile clamp
    all_values = np.concatenate([Z.flatten() for Z in Z_list])
    global_min = float(np.percentile(all_values, 1))   # 1st percentile
    global_max = float(np.percentile(all_values, 99))  # 99th percentile
    
    # global_abs = max(float(np.max(np.abs(Z))) for Z in Z_list)
    # vmin, vmax = -global_abs, global_abs
    # norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # 3) make one wide figure
    n = len(suffixes)
    fig = plt.figure(figsize=(3.2 * n + 1.2, 4.2), dpi=220)
    axes = []

    mappable = None
    for i, (suf, Z) in enumerate(zip(suffixes, Z_list), start=1):
        ax = fig.add_subplot(1, n, i, projection="3d")
        axes.append(ax)

        Zp, y = downsample_for_surface(Z, SURFACE_TARGET_SIG_SAMPLES)
        sig_dim_s, num_layers = Zp.shape
        x = np.arange(num_layers)
        X, Y = np.meshgrid(x, y)

        surf = ax.plot_surface(
            X, Y, Zp,
            cmap="jet",
            linewidth=0,
            antialiased=True,
            vmin=global_min,
            vmax=global_max,
        )
        if mappable is None:
            mappable = surf

        # 标题用论文那种短名字
        short = suf.replace("self_attn.", "layers.").replace("mlp.", "layers.")
        short = short.replace(".weight", "")
        ax.set_title(short, pad=2)

        ax.set_xlabel("layer index")
        ax.set_ylabel("sig dim")
        ax.set_zlabel("ratio")

        ax.view_init(elev=28, azim=-55)

        # 让子图更紧凑
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.tick_params(axis="z", which="major", labelsize=7)

    fig.suptitle(title, y=0.98, fontsize=14)

    # 4) one shared colorbar on the right
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.62, pad=0.01)
    cbar.set_label("sigma_after / sigma_before", rotation=90)

    plt.tight_layout(rect=[0, 0, 0.98, 0.95])
    plt.savefig(out_png)
    plt.close()
    print(f"[saved] {out_png}  vmin={global_min:.6g} vmax={global_max:.6g}")

if __name__ == "__main__":
    plot_surfaces_row(
        CACHE_DIR,
        weight_suffix_list,
        OUT_DIR / "svsm_row_surface.png",
        title="Llama adam 1e-6 weight SVSMs (row projections)",
    )
