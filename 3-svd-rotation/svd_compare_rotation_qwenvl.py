
#!/usr/bin/env python
# compare_rotation.py
#
# 读取三个模型（base, rl1, rl2）在同一层的 U 矩阵，
# 计算旋转矩阵 R1, R2, R3, R4，并输出 R4 和 R3 的 MSE。

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ==========================
# 硬编码参数（你只需要改 LAYER_NAME）
# ==========================

# 对应你做 SVD 的三个模型 ID / 输出目录
MODEL_BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_RL1  = "IffYuan/Embodied-R1-3B-Stage1"
MODEL_RL2  = "IffYuan/Embodied-R1-3B-v1"

# chang above threee to its basename
MODEL_BASE = os.path.basename(MODEL_BASE)
MODEL_RL1 = os.path.basename(MODEL_RL1)
MODEL_RL2 = os.path.basename(MODEL_RL2)

# 如果你之前的 svd_decompose_hf.py 用默认 output_dir，
# 那么 SVD 结果就放在 ./Qwen/Qwen2.5-VL-3B-Instruct 这类目录下。
# 如果你当时改过 output_dir，这里把路径改成对应的就行。
SVD_ROOT = "qwenvl-svd"  # 以当前目录为根路径

# 要对比的 layer 名，必须和 model.named_parameters() 里的 name 一致
# 比如: "model.layers.0.mlp.up_proj.weight"
# LAYER_NAME = "layers.0.mlp.up_proj.weight"  # TODO: 改成你想分析的那一层


# ==========================
# 工具函数
# ==========================

def svd_file_path(model_dir: str, layer_name: str) -> str:
    """
    给定模型目录名和 layer 参数名，构造 SVD 结果文件路径。
    我们之前保存时，用的是 name.replace('.', '__') + '.pt'
    """
    file_name = layer_name.replace(".", "__") + ".pt"
    return os.path.join(SVD_ROOT, model_dir, file_name)


def load_svd_matrices(model_dir: str, layer_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从指定模型的 SVD 结果中加载某一层的 U 和 Vh 矩阵。
    
    Returns:
        U: shape [m, r]
        Vh: shape [r, n]
    """
    path = svd_file_path(model_dir, layer_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"SVD file not found: {path}")
    data = torch.load(path, map_location="cpu")
    
    return data
    
    # return data["U"], data["Vh"]


def compute_rotation(U_src: torch.Tensor, U_tgt: torch.Tensor) -> torch.Tensor:
    """
    给定两个正交基 U_src, U_tgt，计算旋转矩阵 R，使得:
        U_tgt ≈ U_src @ R
    在理想情况下，R = U_src^T @ U_tgt 是一个正交矩阵。
    """
    if U_src.shape != U_tgt.shape:
        raise ValueError(
            f"Shape mismatch between U_src {tuple(U_src.shape)} and U_tgt {tuple(U_tgt.shape)}"
        )
    # [m, r]^T @ [m, r] -> [r, r]
    return U_src.T @ U_tgt


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """均方误差"""
    return torch.mean((a - b) ** 2).item()


# ==========================
# 主逻辑
# ==========================

verbose = False
debug = False
random_projection = False
optimal_projection = False
optimal_projection_base = False

def compute_layer_mse(LAYER_NAME: str, verbose=True):
    """
    计算单层的 U 和 Vh 旋转 MSE，返回 (mse_U, mse_Vh)
    """
    if verbose:
        print("+" * 50)
        print(f"Layer name: {LAYER_NAME}")
    
    data_base = load_svd_matrices(MODEL_BASE, LAYER_NAME)
    data_rl1  = load_svd_matrices(MODEL_RL1, LAYER_NAME)
    data_rl2  = load_svd_matrices(MODEL_RL2, LAYER_NAME)
    
    if debug:
        print(data_base.keys())
        print(f"layer name: {LAYER_NAME}")
        print(f"original shape: {data_base['orig_shape']}")
        breakpoint()
        print()
        return 0.0, 0.0
    
    U_base, Vh_base = data_base["U"], data_base["Vh"]
    U_rl1, Vh_rl1   = data_rl1["U"], data_rl1["Vh"]
    U_rl2, Vh_rl2   = data_rl2["U"], data_rl2["Vh"]
    
    if random_projection:
        # change U_rl1, and Vh_rl1 into random orthogonal matrix with same shape
        # For U_rl1: shape (m, r), generate random orthogonal columns
        m, r = U_rl1.shape
        random_matrix = torch.randn(m, r, dtype=U_rl1.dtype)
        U_rl1, _ = torch.linalg.qr(random_matrix)  # QR decomposition gives orthogonal Q
        
        # For Vh_rl1: shape (r, n), generate random orthogonal rows
        r, n = Vh_rl1.shape
        random_matrix = torch.randn(r, n, dtype=Vh_rl1.dtype)
        Vh_rl1, _ = torch.linalg.qr(random_matrix.T)  # QR on transpose, then transpose back
        Vh_rl1 = Vh_rl1.T
        
    if optimal_projection:
        # use optimal projection matrix, where this should be U_rl2
        U_rl1 = U_rl2
        Vh_rl1 = Vh_rl2
        
    if optimal_projection_base:
        U_rl1 = U_base
        Vh_rl1 = Vh_base
    
    # === U 矩阵的旋转分析 ===
    R1_U = U_base.T @ U_rl1
    R2_U = U_rl1.T @ U_rl2
    R3_U = U_base.T @ U_rl2
    R4_U = R1_U @ R2_U 
    
    # R3 = U_base.T @ U_rl2
    
    # R4 = (U_base^T @ U_rl1) @ (U_rl1^T @ U_rl2) 
    #    = U_base^T @ (U_rl1 @ U_rl1.T) @ U_rl2 
    
    # U: shape (3840, 1280)
    # (U_rl1 @ U_rl1.T) shape (3840, 3840), which only first 1280 diagno entry is 1, rest all 0.
    
    # === Vh 矩阵的旋转分析 ===
    R1_Vh = Vh_base @ Vh_rl1.T
    R2_Vh = Vh_rl1 @ Vh_rl2.T
    R3_Vh = Vh_base @ Vh_rl2.T
    R4_Vh = R1_Vh @ R2_Vh
    
    if verbose:
        print("U matrix shape", tuple(U_base.shape))
        print(f"U Rotation matrix shape: {tuple(R1_U.shape)}")
        print("Vh matrix shape", tuple(Vh_base.shape))
        print(f"Vh Rotation matrix shape: {tuple(R1_Vh.shape)}")
        
        x = U_rl1 @ U_rl1.T
        print(x.shape)
        
        # save the matrix x
        torch.save(x, "debug_U_rl1_U_rl1T.pt")
        
        breakpoint()

    # 计算 MSE
    rotation_mse_U = mse(R4_U, R3_U)
    rotation_mse_Vh = mse(R4_Vh, R3_Vh)
    
    if verbose:
        print(f"\n[U] MSE between R4 (R1 @ R2) and R3 (direct base -> rl2): {rotation_mse_U:.6e}")
        print(f"[Vh] MSE between R4 (R1 @ R2) and R3 (direct base -> rl2): {rotation_mse_Vh:.6e}")
    
    return rotation_mse_U, rotation_mse_Vh


def analyze_module_layers(layer_list, module_name="Module"):
    """
    分析一个模块内所有层的旋转 MSE，计算均值并打印
    返回: (avg_mse_U, avg_mse_Vh, all_mse_U, all_mse_Vh)
    """
    print("=" * 80)
    print(f"{module_name}")
    print("=" * 80)
    
    mse_U_list = []
    mse_Vh_list = []
    
    for layer_name in layer_list:
        mse_U, mse_Vh = compute_layer_mse(layer_name, verbose=True)
        mse_U_list.append(mse_U)
        mse_Vh_list.append(mse_Vh)
    
    avg_mse_U = np.mean(mse_U_list)
    avg_mse_Vh = np.mean(mse_Vh_list)
    
    print("\n" + "=" * 80)
    print(f"{module_name} - Summary Statistics")
    print("=" * 80)
    print(f"Average [U] MSE: {avg_mse_U:.6e}")
    print(f"Average [Vh] MSE: {avg_mse_Vh:.6e}")
    print(f"Std [U] MSE: {np.std(mse_U_list):.6e}")
    print(f"Std [Vh] MSE: {np.std(mse_Vh_list):.6e}")
    
    return avg_mse_U, avg_mse_Vh, mse_U_list, mse_Vh_list


def plot_layer_mse_curves(layer_configs, output_path="rotation_mse_curves.png"):
    """
    绘制不同层之间旋转 MSE 的变化曲线
    
    layer_configs: list of dict, 每个 dict 包含:
        - 'name': 层的名称模板（如 "visual.blocks.{}.attn.qkv.weight"）
        - 'range': 层的范围（如 range(0, 10)）
        - 'label': 图例标签
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for config in layer_configs:
        name_template = config['name']
        layer_range = config['range']
        label = config['label']
        
        mse_U_list = []
        mse_Vh_list = []
        layer_indices = []
        
        for i in layer_range:
            layer_name = name_template.format(i)
            
            mse_U, mse_Vh = compute_layer_mse(layer_name, verbose=False)
            mse_U_list.append(mse_U)
            mse_Vh_list.append(mse_Vh)
            layer_indices.append(i)
            print(f"Processed {layer_name}: U={mse_U:.6e}, Vh={mse_Vh:.6e}")
            
            # try:
            #     mse_U, mse_Vh = compute_layer_mse(layer_name, verbose=False)
            #     mse_U_list.append(mse_U)
            #     mse_Vh_list.append(mse_Vh)
            #     layer_indices.append(i)
            #     print(f"Processed {layer_name}: U={mse_U:.6e}, Vh={mse_Vh:.6e}")
            # except Exception as e:
            #     print(f"Skipped {layer_name}: {e}")
        
        # 绘制 U 矩阵 MSE
        ax1.plot(layer_indices, mse_U_list, marker='o', label=label, linewidth=2, markersize=6)
        
        # 绘制 Vh 矩阵 MSE
        ax2.plot(layer_indices, mse_Vh_list, marker='s', label=label, linewidth=2, markersize=6)
    
    # 设置 U 矩阵图
    ax1.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax1.set_title('U Matrix Rotation MSE Across Layers', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 设置 Vh 矩阵图
    ax2.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax2.set_title('Vh Matrix Rotation MSE Across Layers', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved rotation MSE curves to {output_path}")
    plt.close()

    # # 可选：检查一下 R1, R2, R3 的正交性误差
    # def orthogonality_error(R: torch.Tensor) -> float:
    #     I = torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
    #     return torch.mean((R.T @ R - I) ** 2).item()

    # print("\nOrthogonality MSE (R^T R vs I):")
    # print(f"  R1: {orthogonality_error(R1):.6e}")
    # print(f"  R2: {orthogonality_error(R2):.6e}")
    # print(f"  R3: {orthogonality_error(R3):.6e}")
    # print(f"  R4: {orthogonality_error(R4):.6e}")


if __name__ == "__main__":
    
    # # ===== 1. 分析单个模块的所有层（计算模块均值）=====
    # print("\n")
    # print("=" * 80)
    # print("PART 1: Module-wise Analysis (Layer 0 Only)")
    # print("=" * 80)
    
    # vision_layer_list = [
    #     "visual.blocks.0.attn.qkv.weight",
    #     "visual.blocks.0.attn.proj.weight",
    #     "visual.blocks.0.mlp.gate_proj.weight",
    #     "visual.blocks.0.mlp.up_proj.weight",
    #     "visual.blocks.0.mlp.down_proj.weight",
    # ]
    
    # analyze_module_layers(vision_layer_list, module_name="Vision Module (Layer 0)")
    
    # language_layer_list = [
    #     "model.layers.0.self_attn.q_proj.weight",
    #     "model.layers.0.self_attn.k_proj.weight",
    #     "model.layers.0.self_attn.v_proj.weight",
    #     "model.layers.0.self_attn.o_proj.weight",
    #     "model.layers.0.mlp.up_proj.weight",
    #     "model.layers.0.mlp.down_proj.weight",
    #     "model.layers.0.mlp.gate_proj.weight",
    # ]
    
    # analyze_module_layers(language_layer_list, module_name="Language Module (Layer 0)")
    
    
    # ===== 2. 绘制不同层之间的 MSE 曲线 =====
    print("\n\n")
    print("=" * 80)
    print("PART 2: Cross-Layer Analysis")
    print("=" * 80)
    
    # for vision parts
    layer_start = 0
    layer_end = 32  
    
    output_path = "qwenvl_rotation_mse_curves_vision.png"
    
    if random_projection:
        output_path = f"random_projection_{output_path}"
        
    if optimal_projection:
        output_path = f"optimal_projection_{output_path}"
        
    if optimal_projection_base:
        output_path = f"optimal_projection_base_{output_path}"
    
    layer_configs = [
        {
            'name': 'model.visual.blocks.{}.attn.qkv.weight',
            'range': range(layer_start, layer_end),  # 分析前10层
            'label': 'Vision QKV'
        },
        {
            'name': 'model.visual.blocks.{}.attn.proj.weight',
            'range': range(layer_start, layer_end),
            'label': 'Vision Proj'
        },
        {
            'name': 'model.visual.blocks.{}.mlp.gate_proj.weight',
            'range': range(layer_start, layer_end),
            'label': 'Vision MLP Gate'
        },
        {
            'name': 'model.visual.blocks.{}.mlp.up_proj.weight',
            'range': range(layer_start, layer_end),
            'label': 'Vision MLP Up'
        },
        {
            'name': 'model.visual.blocks.{}.mlp.down_proj.weight',
            'range': range(layer_start, layer_end),
            'label': 'Vision MLP Down'
        },
        # 如果需要分析 language 层，取消下面的注释
    ]
    
    # plot_layer_mse_curves(layer_configs, output_path=output_path)
    
    
    # for language parts
    layer_start = 0
    layer_end = 32
    
    output_path = "qwenvl_rotation_mse_curves_language.png"
    
    if random_projection:
        output_path = f"random_projection_{output_path}"
        
    if optimal_projection:
        output_path = f"optimal_projection_{output_path}"
        
    if optimal_projection_base:
        output_path = f"optimal_projection_base_{output_path}"
    
    layer_configs = [
        {
            'name': 'model.language_model.layers.{}.self_attn.q_proj.weight',
            'range': range(layer_start, layer_end),
            'label': 'Language Q'
        },
        {
            'name': 'model.language_model.layers.{}.self_attn.k_proj.weight',
            'range': range(layer_start, layer_end),
            'label': 'Language K'
        },
        {
            'name': 'model.language_model.layers.{}.self_attn.v_proj.weight',
            'range': range(layer_start, layer_end),
            'label': 'Language V'
        },
        {
            'name': 'model.language_model.layers.{}.self_attn.o_proj.weight',
            'range': range(layer_start, layer_end),
            'label': 'Language O'
        },
        {
            'name': 'model.language_model.layers.{}.mlp.down_proj.weight',
            'range': range(layer_start, layer_end),
            'label': 'Language MLP Down'
        },
        {
            'name': 'model.language_model.layers.{}.mlp.gate_proj.weight',
            'range': range(layer_start, layer_end),
            'label': 'Language MLP Gate'
        },
        {
            'name': 'model.language_model.layers.{}.mlp.up_proj.weight',
            'range': range(layer_start, layer_end),
            'label': 'Language MLP Up'
        },
    ]
    
    plot_layer_mse_curves(layer_configs, output_path=output_path)
    
    
