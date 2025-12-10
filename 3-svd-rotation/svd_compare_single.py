
#!/usr/bin/env python
# compare_rotation.py
#
# 读取三个模型（base, rl1, rl2）在同一层的 U 矩阵，
# 计算旋转矩阵 R1, R2, R3, R4，并输出 R4 和 R3 的 MSE。

import os
import torch

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


def load_U(model_dir: str, layer_name: str) -> torch.Tensor:
    """
    从指定模型的 SVD 结果中加载某一层的 U 矩阵。
    """
    path = svd_file_path(model_dir, layer_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"SVD file not found: {path}")
    data = torch.load(path, map_location="cpu")
    U = data["U"]  # shape: [m, r]
    return U


def load_Vh(model_dir: str, layer_name: str) -> torch.Tensor:
    """
    从指定模型的 SVD 结果中加载某一层的 Vh 矩阵。
    """
    path = svd_file_path(model_dir, layer_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"SVD file not found: {path}")
    data = torch.load(path, map_location="cpu")
    Vh = data["Vh"]  # shape: [r, n]
    return Vh


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

def main(LAYER_NAME: str):
    print("+" * 50)
    print(f"Layer name: {LAYER_NAME}")
    # print("Loading U matrices...")

    U_base = load_U(MODEL_BASE, LAYER_NAME)
    U_rl1  = load_U(MODEL_RL1,  LAYER_NAME)
    U_rl2  = load_U(MODEL_RL2,  LAYER_NAME)
    
    Vh_base = load_Vh(MODEL_BASE, LAYER_NAME)
    Vh_rl1  = load_Vh(MODEL_RL1,  LAYER_NAME)
    Vh_rl2  = load_Vh(MODEL_RL2,  LAYER_NAME)

    # print(f"U_base shape: {tuple(U_base.shape)}")
    # print(f"U_rl1  shape: {tuple(U_rl1.shape)}")
    # print(f"U_rl2  shape: {tuple(U_rl2.shape)}")

    # U_rl1 = U_base @ R1
    # R = U_base.T @ U_rl1
    
    # U_rl2 = U_rl1 @ R2
    # R2 = U_rl1.T @ U_rl2
    
    # U_rl2 = U_base @ (R1 @ R2)

    # === U 矩阵的旋转分析 ===
    # R1: base -> rl1
    R1_U = U_base.T @ U_rl1
    # R2: rl1 -> rl2
    R2_U = U_rl1.T @ U_rl2
    # R3: base -> rl2
    R3_U = U_base.T @ U_rl2
    # R4: 先从 base -> rl1，再 rl1 -> rl2
    R4_U = R1_U @ R2_U
    
    # === Vh 矩阵的旋转分析 ===
    # 注意：Vh 是 [r, n]，V = Vh.T 是 [n, r]
    # 旋转矩阵：R = Vh1 @ Vh2.T (因为 Vh 已经是转置形式)
    # R1: base -> rl1
    R1_Vh = Vh_base @ Vh_rl1.T
    # R2: rl1 -> rl2
    R2_Vh = Vh_rl1 @ Vh_rl2.T
    # R3: base -> rl2
    R3_Vh = Vh_base @ Vh_rl2.T
    # R4: 先从 base -> rl1，再 rl1 -> rl2
    R4_Vh = R1_Vh @ R2_Vh
    
    print("U matrix shape", tuple(U_base.shape))
    print(f"U Rotation matrix shape: {tuple(R1_U.shape)}")
    print("Vh matrix shape", tuple(Vh_base.shape))
    print(f"Vh Rotation matrix shape: {tuple(R1_Vh.shape)}")

    # print(f"R1 shape: {tuple(R1.shape)}")
    # print(f"R2 shape: {tuple(R2.shape)}")
    # print(f"R3 shape: {tuple(R3.shape)}")
    # print(f"R4 shape: {tuple(R4.shape)}")

    # 计算 R4 和 R3 的 MSE (U 矩阵)
    rotation_mse_U = mse(R4_U, R3_U)
    print(f"\n[U] MSE between R4 (R1 @ R2) and R3 (direct base -> rl2): {rotation_mse_U:.6e}")
    
    # 计算 R4 和 R3 的 MSE (Vh 矩阵)
    rotation_mse_Vh = mse(R4_Vh, R3_Vh)
    print(f"[Vh] MSE between R4 (R1 @ R2) and R3 (direct base -> rl2): {rotation_mse_Vh:.6e}")

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
    
    # LAYER_NAME = "layers.0.mlp.up_proj.weight"  # TODO: 改成你想分析的那一层

    print("=" * 50)
    print("Vision Parts")
    print()

    layer_list = [
        "visual.blocks.0.attn.qkv.weight",
        "visual.blocks.0.attn.proj.weight",
        "visual.blocks.0.mlp.gate_proj.weight",
        "visual.blocks.0.mlp.up_proj.weight",
        "visual.blocks.0.mlp.down_proj.weight",
    ]
    
    for LAYER_NAME in layer_list:
        main(LAYER_NAME)
        
        
    # print("\n\n\n\n\n\n")
    # print("=" * 50)
    # print("Language Parts")
    
    # layer_list = [
    #     "model.layers.0.self_attn.q_proj.weight",
    #     "model.layers.0.self_attn.k_proj.weight",
    #     "model.layers.0.self_attn.v_proj.weight",
    #     "model.layers.0.self_attn.o_proj.weight",
    #     "model.layers.0.mlp.up_proj.weight",
    #     "model.layers.0.mlp.down_proj.weight",
    #     "model.layers.0.mlp.gate_proj.weight "
    # ]
    
    # for LAYER_NAME in layer_list:
    #     main(LAYER_NAME)
    
    
