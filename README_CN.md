# SVD 模型分析与对比

本仓库实现了基于 SVD 分解的多模型旋转矩阵分析方法，用于研究不同训练阶段模型权重的变化规律。

## 核心思想

当模型经过训练（如强化学习）后，其权重矩阵会发生变化。我们使用 SVD 分解来分析这种变化：

```
W = U @ diag(S) @ Vh
```

其中：
- **U**: 左奇异向量矩阵（输出空间的旋转）
- **S**: 奇异值向量（缩放因子）
- **Vh**: 右奇异向量矩阵（输入空间的旋转）

**关键发现**：通过比较三个模型（base → rl_stage1 → rl_stage2）的旋转矩阵，我们可以验证训练过程中旋转变换的**可组合性**：

```
R_direct = R_stage1 @ R_stage2
```

如果上述等式成立（MSE 很小），说明训练过程在旋转空间中是连续的、可预测的。

## 使用流程

### 第一步：SVD 分解

对每个模型的所有权重层进行 SVD 分解：

```bash
# 方法1: 单个模型分解
python svd_decompose.py --model_id "Qwen/Qwen2.5-VL-3B-Instruct"

# 方法2: 批量并行分解（推荐）
bash run_svd_decompose.sh
```

**脚本说明** (`svd_decompose.py`):
- 加载 HuggingFace 模型到 CPU（避免显存溢出）
- 遍历所有 2D 权重矩阵（跳过标量和向量）
- 将每层 reshape 为 `[out_dim, in_dim]` 然后做 SVD
- 转换为 float32 保证数值稳定性
- 保存结果到 `qwenvl-svd/{model_basename}/{layer_name}.pt`

**输出格式**:
每个 `.pt` 文件包含：
```python
{
    "U": Tensor,        # [m, r]
    "S": Tensor,        # [r]
    "Vh": Tensor,       # [r, n]
    "orig_shape": tuple # 原始权重形状
}
```

**配置** (`run_svd_decompose.sh`):
```bash
MODEL_IDS=(
  "Qwen/Qwen2.5-VL-3B-Instruct"      # base model
  "IffYuan/Embodied-R1-3B-Stage1"    # RL stage 1
  "IffYuan/Embodied-R1-3B-v1"        # RL stage 2
)
```

### 第二步：旋转矩阵对比分析

对比三个模型的旋转变化：

```bash
python svd_compare.py
```

**脚本说明** (`svd_compare.py`):

#### 2.1 旋转矩阵计算

对于 **U 矩阵**（输出空间）：
```python
R1_U = U_base.T @ U_rl1        # base → rl1 的旋转
R2_U = U_rl1.T @ U_rl2         # rl1 → rl2 的旋转
R3_U = U_base.T @ U_rl2        # base → rl2 的直接旋转
R4_U = R1_U @ R2_U             # 组合旋转
```

对于 **Vh 矩阵**（输入空间）：
```python
R1_Vh = Vh_base @ Vh_rl1.T     # base → rl1 的旋转
R2_Vh = Vh_rl1 @ Vh_rl2.T      # rl1 → rl2 的旋转
R3_Vh = Vh_base @ Vh_rl2.T     # base → rl2 的直接旋转
R4_Vh = R1_Vh @ R2_Vh          # 组合旋转
```

#### 2.2 可组合性验证

计算 MSE 来验证旋转的可组合性：
```python
rotation_mse_U = MSE(R4_U, R3_U)     # 理想情况下应该 ≈ 0
rotation_mse_Vh = MSE(R4_Vh, R3_Vh)
```

如果 MSE 很小（例如 < 1e-6），说明：
- ✅ 从 base → rl1 → rl2 的渐进训练等价于 base → rl2 的直接训练
- ✅ 模型在旋转空间中的变化是连续、可预测的
- ✅ 可以用中间阶段的模型来理解整体训练轨迹

#### 2.3 分析模式

脚本支持两种分析模式：

**模式 1: 单层多模块分析**（计算模块内平均 MSE）
```python
vision_layer_list = [
    "visual.blocks.0.attn.qkv.weight",
    "visual.blocks.0.attn.proj.weight",
    "visual.blocks.0.mlp.gate_proj.weight",
    "visual.blocks.0.mlp.up_proj.weight",
    "visual.blocks.0.mlp.down_proj.weight",
]
analyze_module_layers(vision_layer_list, module_name="Vision Module")
```

**模式 2: 跨层曲线分析**（观察不同层的 MSE 变化趋势）
```python
layer_configs = [
    {
        'name': 'model.layers.{}.self_attn.q_proj.weight',
        'range': range(0, 5),
        'label': 'Language Q'
    },
    # ... 更多层配置
]
plot_layer_mse_curves(layer_configs, output_path="rotation_mse_curves.png")
```

**输出**:
- 终端打印：每层的 MSE 值、模块平均统计
- 可视化图表：`rotation_mse_curves.png`（双图：U 矩阵 MSE + Vh 矩阵 MSE）
