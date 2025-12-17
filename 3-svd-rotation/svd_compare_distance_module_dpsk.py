#!/usr/bin/env python
# module_level_distance_stats.py
#
# 功能修改：
# - 不再按 Layer 画曲线。
# - 而是统计特定 Module (如 q_proj) 在所有层上的平均距离。
# - 输出分组柱状图 (Bar Chart)。

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# 模型与 SVD 路径配置
# ==========================

MODEL_BASE_EXT = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_RL1_EXT  = "agentica-org/DeepScaleR-1.5B-Preview"
MODEL_RL2_EXT  = "agentica-org/DeepCoder-1.5B-Preview"

MODEL_BASE = os.path.basename(MODEL_BASE_EXT)
MODEL_RL1  = os.path.basename(MODEL_RL1_EXT)
MODEL_RL2  = os.path.basename(MODEL_RL2_EXT)

# 你的 SVD 结果根目录
SVD_ROOT = "qwenvl-svd"

# 距离结果缓存目录
DIST_CACHE_ROOT = os.path.join(SVD_ROOT, "distance_cache")


# ==========================
# 工具函数：加载 SVD (保持不变)
# ==========================

def svd_file_path(model_dir: str, layer_name: str) -> str:
    """SVD 保存时用 name.replace('.', '__') + '.pt' 作为文件名。"""
    file_name = layer_name.replace(".", "__") + ".pt"
    return os.path.join(SVD_ROOT, model_dir, file_name)


def load_svd_matrices(model_dir: str, layer_name: str) -> dict:
    path = svd_file_path(model_dir, layer_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"SVD file not found: {path}")
    return torch.load(path, map_location="cpu")


# ==========================
# 距离计算 + 缓存 (保持不变)
# ==========================

def distance_cache_path(layer_name: str, r_cap: int | None) -> str:
    safe_name = layer_name.replace(".", "__")
    suffix = "full" if r_cap is None else f"rcap{r_cap}"
    file_name = f"{safe_name}_{suffix}.pt"
    return os.path.join(DIST_CACHE_ROOT, file_name)


def _pairwise_distances_from_U(
    U1: torch.Tensor,
    U2: torch.Tensor,
    r_cap: int | None = None,
) -> dict:
    U1 = U1.to(torch.float32)
    U2 = U2.to(torch.float32)

    if r_cap is not None and U1.shape[1] > r_cap:
        U1 = U1[:, :r_cap].contiguous()
        U2 = U2[:, :r_cap].contiguous()

    # Frobenius
    d_frob = torch.linalg.norm(U1 - U2, ord="fro").item()

    # Geo / Chordal
    M = U1.T @ U2
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
    os.makedirs(DIST_CACHE_ROOT, exist_ok=True)
    cache_path = distance_cache_path(layer_name, r_cap)

    if os.path.exists(cache_path):
        # print(f"[Cache hit] {layer_name}") # 减少刷屏，注视掉
        return torch.load(cache_path, map_location="cpu")

    # 如果没有缓存，则加载 SVD 计算
    try:
        data_base = load_svd_matrices(MODEL_BASE, layer_name)
        data_rl1  = load_svd_matrices(MODEL_RL1, layer_name)
        data_rl2  = load_svd_matrices(MODEL_RL2, layer_name)
    except FileNotFoundError as e:
        # 如果文件不存在，向上抛出，由调用者处理
        raise e

    U_base = data_base["U"]
    U_rl1  = data_rl1["U"]
    U_rl2  = data_rl2["U"]

    if verbose:
        print(f"[Compute] {layer_name} (r_cap={r_cap})")

    dist_base_rl1 = _pairwise_distances_from_U(U_base, U_rl1, r_cap=r_cap)
    dist_base_rl2 = _pairwise_distances_from_U(U_base, U_rl2, r_cap=r_cap)
    dist_rl1_rl2  = _pairwise_distances_from_U(U_rl1,  U_rl2,  r_cap=r_cap)

    dists = {name: {} for name in dist_base_rl1.keys()}
    for metric_name in dists.keys():
        dists[metric_name]["base-rl1"] = dist_base_rl1[metric_name]
        dists[metric_name]["base-rl2"] = dist_base_rl2[metric_name]
        dists[metric_name]["rl1-rl2"]  = dist_rl1_rl2[metric_name]

    torch.save(dists, cache_path)
    return dists


# ==========================
# 新逻辑：按 Module 聚合 (Cross-Layer Mean)
# ==========================

def analyze_and_plot_aggregated_by_module(
    layer_indices,
    module_map,
    r_cap: int | None = 256,
    out_png_prefix: str = "module_aggregated_dist",
):
    """
    参数:
    module_map: dict, e.g. {"q_proj": "layers.{}.self_attn.q_proj.weight"}
    
    逻辑:
    1. 遍历 module_map 中的每种 module (key)。
    2. 遍历 layer_indices，收集该 module 在每一层的 distance。
    3. 计算该 module 跨层 (Cross-Layer) 的 Mean 和 Std。
    4. 画 Grouped Bar Chart。
    """
    metrics = ["frobenius", "geo", "chordal"]
    pairs = ["base-rl1", "base-rl2", "rl1-rl2"]
    
    module_names = list(module_map.keys())
    
    # 存储结构: stats[metric][pair][module_name] = [list of values across layers]
    raw_data = {
        m: {p: {mod: [] for mod in module_names} for p in pairs}
        for m in metrics
    }

    print(f"Collecting data for modules: {module_names} across {len(layer_indices)} layers...")

    for layer_idx in layer_indices:
        for mod_short_name, mod_template in module_map.items():
            full_layer_name = mod_template.format(layer_idx)
            
            try:
                dists = compute_layer_distances(full_layer_name, r_cap=r_cap, verbose=False)
                # 将结果塞入列表
                for m in metrics:
                    for p in pairs:
                        val = dists[m][p]
                        raw_data[m][p][mod_short_name].append(val)
            except FileNotFoundError:
                # 某些层可能没有某些 module，或者没跑 SVD，跳过
                pass

    # ==========================
    # 数据聚合与打印
    # ==========================
    # aggregated[metric][pair][module_name] = (mean, std)
    aggregated = {
        m: {p: {} for p in pairs} for m in metrics
    }

    print("\n" + "="*60)
    print(f"Aggregated Stats (Mean across {len(layer_indices)} layers)")
    print("="*60)

    for m in metrics:
        print(f"\nMetric: [{m}]")
        print(f"{'Module':<15} | {'Pair':<10} | {'Mean':<12} | {'Std':<12}")
        print("-" * 55)
        
        for p in pairs:
            for mod in module_names:
                vals = np.array(raw_data[m][p][mod], dtype=float)
                if len(vals) > 0:
                    mean_val = np.mean(vals)
                    std_val = np.std(vals)
                    aggregated[m][p][mod] = (mean_val, std_val)
                    print(f"{mod:<15} | {p:<10} | {mean_val:.4e}   | {std_val:.4e}")
                else:
                    aggregated[m][p][mod] = (0.0, 0.0)

    # ==========================
    # 画图：Grouped Bar Chart
    # ==========================
    # 为每个 Metric 画一张图
    
    x = np.arange(len(module_names))  # label locations
    width = 0.25  # the width of the bars
    
    for m in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 针对每个 pair 画一组柱子
        # 偏移量计算：如果有3个pair，偏移依次为 -width, 0, +width
        offsets = [-width, 0, width]
        
        for idx, p in enumerate(pairs):
            # 收集该 metric 下，该 pair 对应所有 module 的 mean 值
            means = [aggregated[m][p][mod][0] for mod in module_names]
            stds  = [aggregated[m][p][mod][1] for mod in module_names] # 也可以画 error bar
            
            rects = ax.bar(x + offsets[idx], means, width, label=p, alpha=0.85) # yerr=stds 可选

        ax.set_ylabel(f'Mean {m} Distance (avg across layers)')
        ax.set_title(f'Module-Level Change Comparison ({m})\n(Averaged over layers {list(layer_indices)[0]}-{list(layer_indices)[-1]})')
        ax.set_xticks(x)
        ax.set_xticklabels(module_names, rotation=45)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        fig.tight_layout()
        out_path = f"{out_png_prefix}_{m}_rcap{r_cap if r_cap is not None else 'full'}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[Saved Plot] {out_path}")


# ==========================
# Main
# ==========================

if __name__ == "__main__":
    # 1. 设定要统计的 Layer 范围
    layer_indices = range(0, 28) 

    # 2. 设定 Module 映射 (Short Name -> Template)
    # 这样图表上的 label 比较好看 (例如显示 'q_proj' 而不是长字符串)
    module_map = {
        "q_proj":    "layers.{}.self_attn.q_proj.weight",
        "k_proj":    "layers.{}.self_attn.k_proj.weight",
        "v_proj":    "layers.{}.self_attn.v_proj.weight",
        "o_proj":    "layers.{}.self_attn.o_proj.weight",
        "gate_proj": "layers.{}.mlp.gate_proj.weight",
        "up_proj":   "layers.{}.mlp.up_proj.weight",
        "down_proj": "layers.{}.mlp.down_proj.weight",
    }

    analyze_and_plot_aggregated_by_module(
        layer_indices=layer_indices,
        module_map=module_map,
        r_cap=None,
        out_png_prefix="module_level_stats"
    )