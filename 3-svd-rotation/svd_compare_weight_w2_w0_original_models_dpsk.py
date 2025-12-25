import torch
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os
import gc

# ==========================
# 配置部分
# ==========================
# 使用指定的 1.5B 模型
MODEL_BASE_ID = "Qwen/Qwen2.5-Math-1.5B"
MODEL_RL2_ID  = "agentica-org/DeepScaleR-1.5B-Preview"

SAVE_FIGURE_NAME = "dpsk_w2_w0_ori_weights_mse_curves_language.png"


# 计算设备配置
# 对于 1.5B 模型，如果有 16G 以上显存的 GPU，可以使用 "cuda" 加快加载和计算
# 如果显存不足或只有 CPU，使用 "cpu"。
# 为了通用性和防止 OOM，默认设置为 "cpu"，但计算会稍慢。
DEVICE = "cpu" 
# 计算精度，保持 float32 以确保 MSE 计算的准确性
Run_DTYPE = torch.float32

# ==========================
# 核心计算函数
# ==========================

def load_model_weights(model_id):
    """加载模型并返回其 state_dict (权重字典)"""
    print(f"Loading model: {model_id} on {DEVICE}...")
    try:
        # 我们只需要权重，不需要完整的模型图来进行前向传播
        # 所以使用 device_map="auto" 或指定 device 都可以
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=Run_DTYPE,
            device_map=DEVICE,
            trust_remote_code=True,
            low_cpu_mem_usage=True # 优化 CPU 内存使用
        )
        # 获取 state_dict 后立即删除模型以释放内存
        state_dict = model.state_dict()
        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return state_dict
    except Exception as e:
        print(f"Error loading {model_id}: {e}")
        exit(1)

def calculate_mse_across_layers(base_sd, rl2_sd):
    """
    遍历 state_dict，计算指定权重矩阵在每一层的 MSE。
    """
    print("Calculating MSE across layers...")
    
    # 定义我们要追踪的矩阵后缀及其对应的图例标签
    # Qwen2 架构的参数命名规则
    target_matrices = {
        ".self_attn.q_proj.weight": "Language Q",
        ".self_attn.k_proj.weight": "Language K",
        ".self_attn.v_proj.weight": "Language V",
        ".self_attn.o_proj.weight": "Language O",
        ".mlp.gate_proj.weight": "Language MLP Gate",
        ".mlp.up_proj.weight": "Language MLP Up",
        ".mlp.down_proj.weight": "Language MLP Down",
    }
    
    # 用于存储结果: results[label] = {layer_idx: mse_value}
    results = defaultdict(dict)
    max_layer_idx = 0

    # 获取所有键的交集，确保只比较两个模型都有的参数
    common_keys = set(base_sd.keys()) & set(rl2_sd.keys())
    
    for key in tqdm(common_keys, desc="Processing weights"):
        # 仅处理 model.layers 开头的权重
        if not key.startswith("model.layers."):
            continue

        # 解析层号
        try:
            parts = key.split('.')
            layer_idx = int(parts[2])
            max_layer_idx = max(max_layer_idx, layer_idx)
        except (IndexError, ValueError):
            continue

        # 检查这个 key 是否是我们关注的矩阵类型
        matched_label = None
        for suffix, label in target_matrices.items():
            if key.endswith(suffix):
                matched_label = label
                break
        
        if matched_label:
            # 获取张量并确保在同一设备上进行计算
            w_base = base_sd[key].to(DEVICE)
            w_rl2 = rl2_sd[key].to(DEVICE)
            
            # 计算 MSE
            # torch.nn.functional.mse_loss 默认是 mean reduction
            mse_val = torch.nn.functional.mse_loss(w_base, w_rl2).item()
            results[matched_label][layer_idx] = mse_val

    return results, max_layer_idx

# ==========================
# 绘图函数 (复刻提供的代码风格)
# ==========================

def plot_mse_results(results, max_layer_idx, output_path="weight_mse_comparison.png"):
    """
    根据计算结果绘制折线图，风格参照提供的代码片段。
    """
    print(f"Plotting results to {output_path}...")
    
    # 定义要绘制的顺序和颜色，尽量匹配参考图
    plot_order = [
        "Language Q", "Language K", "Language V", "Language O",
        "Language MLP Down", "Language MLP Gate", "Language MLP Up"
    ]
    # 定义颜色映射 (参考 matplotlib 默认 tab10 色盘)
    colors = {
        "Language Q": "#1f77b4",       # Blue
        "Language K": "#ff7f0e",       # Orange
        "Language V": "#2ca02c",       # Green
        "Language O": "#d62728",       # Red
        "Language MLP Down": "#9467bd", # Purple
        "Language MLP Gate": "#8c564b", # Brown
        "Language MLP Up": "#e377c2"    # Pink
    }
    
    layer_range = range(max_layer_idx + 1)
    
    # 创建画布，风格参照提供的 plot_layer_mse_curves 函数
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # 构建标题信息
    base_name = os.path.basename(MODEL_BASE_ID)
    rl2_name = os.path.basename(MODEL_RL2_ID)
    model_info = f"Base: {base_name}\nRL2: {rl2_name}"

    for label in plot_order:
        if label not in results:
            continue
            
        layer_data = results[label]
        mse_list = []
        layer_indices = []
        
        # 确保按层顺序读取数据
        for i in layer_range:
            if i in layer_data:
                mse_list.append(layer_data[i])
                layer_indices.append(i)
        
        if not mse_list:
            continue

        # 绘制线条
        ax.plot(layer_indices, mse_list, 
                marker='o', 
                label=label, 
                linewidth=2, 
                markersize=6,
                color=colors.get(label))

    # 设置图表样式 (参考提供的代码片段)
    ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    # 修改Y轴标签以反映我们计算的是直接的权重 MSE
    ax.set_ylabel('Weight MSE (W_rl2 vs W_base)', fontsize=12, fontweight='bold')
    ax.set_title(f'Weight MSE Across Layers\n{model_info}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both') # both valid for log scale grids
    
    # 关键：设置 Y 轴为对数坐标
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved weight prediction MSE curves to {output_path}")
    # plt.show() # 如果在 notebook 或支持显示的终端运行，可以取消注释

# ==========================
# 主程序
# ==========================
if __name__ == "__main__":
    # 1. 加载权重
    # 为了节省内存，我们依次加载。加载完一个就提取其 state_dict 并释放模型对象。
    base_sd = load_model_weights(MODEL_BASE_ID)
    rl2_sd = load_model_weights(MODEL_RL2_ID)
    
    # 2. 计算 MSE
    mse_results, max_layer = calculate_mse_across_layers(base_sd, rl2_sd)
    
    # 3. 清理内存 (释放 state_dict)
    del base_sd
    del rl2_sd
    gc.collect()
    
    # 4. 画图
    plot_mse_results(mse_results, max_layer, output_path=SAVE_FIGURE_NAME)