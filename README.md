# SVD 模型分析与对比


所有有用的文件

```
# 对一个模型的所有层做 SVD 分解
svd_decompose.py
run_svd_decompose.sh （批量做 SVD 分解）

# 分析 SVD 分解之后 U 和 V 的 rotation MSE
svd_compare.py
```

## 现在的结果

现在用的是这三个models

RL 的顺序是： base → rl1 → rl2

```
MODEL_IDS=(
  "Qwen/Qwen2.5-VL-3B-Instruct" # base
  "IffYuan/Embodied-R1-3B-Stage1" # rl1
  "IffYuan/Embodied-R1-3B-v1" # rl2
)
```

vision part 的结果
[](./rotation_mse_curves_vision.png)

language part 的结果
[](./rotation_mse_curves_language.png)


## 使用流程

### 第一步：SVD 分解

对每个模型的所有权重层进行 SVD 分解：

```bash
# 方法1: 单个模型分解
python svd_decompose.py --model_id "Qwen/Qwen2.5-VL-3B-Instruct"

# 方法2: 批量并行分解（推荐）
bash run_svd_decompose.sh
```

### 第二步：旋转矩阵对比分析

直接跑下边这个指令，所有的模型名称以及参数，在 python 里边写死了

```bash
python svd_compare.py
```


#### 2.1 旋转矩阵计算

对于 **U 矩阵**（输出空间）：
```python
# base → rl1 的旋转: U_rl1 = U_base @ R1_U 
R1_U = U_base.T @ U_rl1        
# rl1 → rl2 的旋转: U_rl2 = U_rl1 @ R2_U
R2_U = U_rl1.T @ U_rl2         
# base → rl2 的直接旋转: U_rl2 = U_base @ R3_U
R3_U = U_base.T @ U_rl2        
R4_U = R1_U @ R2_U             # 组合旋转
```

#### 2.2 可组合性验证

计算 MSE 来验证旋转的可组合性：
```python
rotation_mse_U = MSE(R4_U, R3_U)     # 理想情况下应该 ≈ 0
rotation_mse_Vh = MSE(R4_Vh, R3_Vh)
```

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
