#!/usr/bin/env python
# minimal_distance_stats.py
#
# 对每一层 layer：
#   - 读入 base / RL1 / RL2 的 U
#   - 计算 U 子空间之间的距离（frobenius / geo / chordal）
#   - 把结果缓存到磁盘，避免重复计算
#   - 对同一层里的多个 module（q/k/v/o/mlp 等）做平均
#   - 分 metric 画出 base-RL1 / base-RL2 / RL1-RL2 的曲线

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# 模型与 SVD 路径配置
# ==========================

# MODEL_BASE_EXT = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_RL1_EXT  = "agentica-org/DeepScaleR-1.5B-Preview"
# MODEL_RL2_EXT  = "agentica-org/DeepCoder-1.5B-Preview"

MODEL_BASE_EXT = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_RL1_EXT  = "IffYuan/Embodied-R1-3B-Stage1"
MODEL_RL2_EXT  = "IffYuan/Embodied-R1-3B-v1"

fig_base_dir = os.path.join("figs", "qwenvl-distance")
os.makedirs(fig_base_dir, exist_ok=True)

MODEL_BASE = os.path.basename(MODEL_BASE_EXT)
MODEL_RL1  = os.path.basename(MODEL_RL1_EXT)
MODEL_RL2  = os.path.basename(MODEL_RL2_EXT)

# 你的 SVD 结果根目录
SVD_ROOT = "qwenvl-svd"

# 距离结果缓存目录
DIST_CACHE_ROOT = os.path.join(SVD_ROOT, "distance_cache")


# ==========================
# 工具函数：加载 SVD
# ==========================

def svd_file_path(model_dir: str, layer_name: str) -> str:
    """SVD 保存时用 name.replace('.', '__') + '.pt' 作为文件名。"""
    file_name = layer_name.replace(".", "__") + ".pt"
    return os.path.join(SVD_ROOT, model_dir, file_name)


def load_svd_matrices(model_dir: str, layer_name: str) -> dict:
    """加载某一层的 SVD 结果：dict(U, S, Vh)。"""
    path = svd_file_path(model_dir, layer_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"SVD file not found: {path}")
    return torch.load(path, map_location="cpu")


# ==========================
# 距离计算 + 缓存
# ==========================

def distance_cache_path(layer_name: str, r_cap: int | None) -> str:
    """根据 layer_name 和 r_cap 构造缓存文件路径。"""
    safe_name = layer_name.replace(".", "__")
    suffix = "full" if r_cap is None else f"rcap{r_cap}"
    file_name = f"{safe_name}_{suffix}.pt"
    return os.path.join(DIST_CACHE_ROOT, file_name)


def _pairwise_distances_from_U(
    U1: torch.Tensor,
    U2: torch.Tensor,
    r_cap: int | None = None,
) -> dict:
    """
    给定两个列正交矩阵 U1, U2，计算：
        - frobenius: ||U1 - U2||_F
        - geo      : Grassmann geodesic distance
        - chordal  : ||sin(theta)||_2
    """
    U1 = U1.to(torch.float32)
    U2 = U2.to(torch.float32)

    # 子空间截断（可选，建议用来加速；None 表示不用截断）
    if r_cap is not None and U1.shape[1] > r_cap:
        U1 = U1[:, :r_cap].contiguous()
        U2 = U2[:, :r_cap].contiguous()

    # Frobenius 距离
    d_frob = torch.linalg.norm(U1 - U2, ord="fro").item()

    # principal angles （只需要 U1^T U2 的奇异值）
    M = U1.T @ U2              # [r, r]
    s = torch.linalg.svdvals(M)
    s = torch.clamp(s, -1.0, 1.0)
    theta = torch.arccos(s)
    sin_theta = torch.sin(theta)

    d_geo = torch.linalg.norm(theta, ord=2).item()
    d_chordal = torch.linalg.norm(sin_theta, ord=2).item()

    return {
        "frobenius": d_frob,
        "geo":       d_geo,
        "chordal":   d_chordal,
    }


def compute_layer_distances(
    layer_name: str,
    r_cap: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    对单个 weight 的 U 做距离分析，并把结果缓存下来。
    
    返回格式：
    {
        'frobenius': {'base-rl1': x, 'base-rl2': y, 'rl1-rl2': z},
        'geo':       {...},
        'chordal':   {...},
    }
    """
    os.makedirs(DIST_CACHE_ROOT, exist_ok=True)
    cache_path = distance_cache_path(layer_name, r_cap)

    # --- 1. 先看缓存 ---
    if os.path.exists(cache_path):
        dists = torch.load(cache_path, map_location="cpu")
        if verbose:
            print(f"[Cache hit] layer={layer_name}, r_cap={r_cap}")
        return dists

    # --- 2. 没缓存就真正算一遍 ---
    data_base = load_svd_matrices(MODEL_BASE, layer_name)
    data_rl1  = load_svd_matrices(MODEL_RL1, layer_name)
    data_rl2  = load_svd_matrices(MODEL_RL2, layer_name)

    U_base = data_base["U"]
    U_rl1  = data_rl1["U"]
    U_rl2  = data_rl2["U"]

    if U_base.shape != U_rl1.shape or U_base.shape != U_rl2.shape:
        raise ValueError(
            f"[Shape mismatch] layer={layer_name}, "
            f"U_base={U_base.shape}, U_rl1={U_rl1.shape}, U_rl2={U_rl2.shape}"
        )

    if verbose:
        print(f"[Compute] {layer_name}, U shape={U_base.shape}, r_cap={r_cap}")

    dist_base_rl1 = _pairwise_distances_from_U(U_base, U_rl1, r_cap=r_cap)
    dist_base_rl2 = _pairwise_distances_from_U(U_base, U_rl2, r_cap=r_cap)
    dist_rl1_rl2  = _pairwise_distances_from_U(U_rl1,  U_rl2,  r_cap=r_cap)

    dists = {name: {} for name in dist_base_rl1.keys()}
    for metric_name in dists.keys():
        dists[metric_name]["base-rl1"] = dist_base_rl1[metric_name]
        dists[metric_name]["base-rl2"] = dist_base_rl2[metric_name]
        dists[metric_name]["rl1-rl2"]  = dist_rl1_rl2[metric_name]

    # 存盘
    torch.save(dists, cache_path)
    if verbose:
        print(f"[Saved] {layer_name} -> {cache_path}")

    return dists


# ==========================
# 按 layer 聚合多个 module 的均值 & 画图
# ==========================

def analyze_and_plot_by_layer(
    layer_indices,
    module_templates,
    r_cap: int | None = 256,
    out_png_prefix: str = "distance_layer_avg",
):
    """
    对每一层：
      - 对该层的多个 module（比如 q/k/v/o/mlp）分别调用 compute_layer_distances
      - 对同一层内的 module 做平均：指标是分开的（frobenius / geo / chordal 分别平均）
      - 得到：metric -> pair -> [per-layer mean over modules]
      - 为每个 metric 单独画一张图
    """
    metrics = ["frobenius", "geo", "chordal"]
    pairs = ["base-rl1", "base-rl2", "rl1-rl2"]

    # metric -> pair -> list of per-layer mean
    metric_pair_to_values = {
        m: {p: [] for p in pairs} for m in metrics
    }

    layer_indices = list(layer_indices)

    for layer_idx in layer_indices:
        # 这一层里：metric -> pair -> 该层所有 module 的值
        per_layer_values = {
            m: {p: [] for p in pairs} for m in metrics
        }

        for tmpl in module_templates:
            layer_name = tmpl.format(layer_idx)
            
            dists = compute_layer_distances(layer_name, r_cap=r_cap, verbose=False)
            # try:
            #     dists = compute_layer_distances(layer_name, r_cap=r_cap, verbose=False)
            # except FileNotFoundError:
            #     # 某个 module 在这个模型里不存在/没做 SVD，就跳过
            #     print(f"[Skip] SVD not found for {layer_name}")
            #     continue

            for m in metrics:
                for p in pairs:
                    per_layer_values[m][p].append(dists[m][p])

        # 对该层的多个 module 取平均
        for m in metrics:
            for p in pairs:
                vals = per_layer_values[m][p]
                if len(vals) == 0:
                    mean_val = np.nan
                else:
                    mean_val = float(np.mean(vals))
                metric_pair_to_values[m][p].append(mean_val)

    # 打印 summary
    print("\n=== Summary over layers (module-avg per layer, per metric, per pair) ===")
    for m in metrics:
        print(f"\n[Metric: {m}]")
        for p in pairs:
            arr = np.array(metric_pair_to_values[m][p], dtype=float)
            # 过滤掉 nan（如果某层所有 module 都缺失）
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                print(f"  {p:8s} | no data")
            else:
                print(f"  {p:8s} | mean={arr.mean():.6e}, std={arr.std():.6e}")

    # 画图：每个 metric 一张图
    for m in metrics:
        plt.figure(figsize=(8, 4))
        for p in pairs:
            ys = np.array(metric_pair_to_values[m][p], dtype=float)
            plt.plot(layer_indices, ys, marker="o", label=p)
        plt.xlabel("Layer index")
        plt.ylabel(f"{m} distance (avg over modules)")
        plt.title(f"{m} distance vs layer (module-averaged)")
        plt.grid(alpha=0.3)
        plt.legend()
        out_path = f"{out_png_prefix}_{m}_rcap{r_cap if r_cap is not None else 'full'}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Saved plot] {out_path}")


# ==========================
# main：自己改 layer_indices / module_templates 即可
# ==========================

if __name__ == "__main__":
    # ===== Vision Blocks Analysis =====
    print("\n" + "=" * 80)
    print("PART 1: Vision Blocks Analysis")
    print("=" * 80)
    
    r_cap = None
    
    layer_start = 0
    layer_end = 32
    layer_indices = range(layer_start, layer_end)

    # Vision 模块：参与平均的 modules
    vision_module_templates = [
        "model.visual.blocks.{}.attn.qkv.weight",
        "model.visual.blocks.{}.attn.proj.weight",
        "model.visual.blocks.{}.mlp.gate_proj.weight",
        "model.visual.blocks.{}.mlp.up_proj.weight",
        "model.visual.blocks.{}.mlp.down_proj.weight",
    ]

    analyze_and_plot_by_layer(
        layer_indices=layer_indices,
        module_templates=vision_module_templates,
        r_cap=r_cap,
        out_png_prefix=os.path.join(fig_base_dir, "vision_blocks_distance"),
    )
    
    # ===== Language Blocks Analysis =====
    print("\n" + "=" * 80)
    print("PART 2: Language Blocks Analysis")
    print("=" * 80)
    
    layer_start = 0
    layer_end = 32
    layer_indices = range(layer_start, layer_end)

    # Language 模块：参与平均的 modules
    language_module_templates = [
        "model.language_model.layers.{}.self_attn.q_proj.weight",
        "model.language_model.layers.{}.self_attn.k_proj.weight",
        "model.language_model.layers.{}.self_attn.v_proj.weight",
        "model.language_model.layers.{}.self_attn.o_proj.weight",
        "model.language_model.layers.{}.mlp.down_proj.weight",
        "model.language_model.layers.{}.mlp.gate_proj.weight",
        "model.language_model.layers.{}.mlp.up_proj.weight",
    ]

    analyze_and_plot_by_layer(
        layer_indices=layer_indices,
        module_templates=language_module_templates,
        r_cap=r_cap,
        out_png_prefix=os.path.join(fig_base_dir, "lang_blocks_distance"),
    )
