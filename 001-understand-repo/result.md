# Plots 目录图片生成流程分析

## 1. 图片是哪个脚本画出来的？

`/ssd1/zhizhou/workspace/rotation-project/replace/plots/` 下的所有 PNG 图片(共 20 张)
是由 **`hq_plot_scan_all.py`** 生成的。

但这个脚本本身只是绘图脚本，它读取一个 CSV 文件。真正做"重活"(加载模型、SVD
分解、计算指标)的脚本是 **`hq_scan_all.py`**，它会生成
`stiefel_analysis_metrics.csv`。

### 完整数据流

```
hq_scan_all.py (跑 SVD + Stiefel 分析)
     │
     ▼
stiefel_analysis_metrics.csv (中间产物，~大表格)
     │
     ▼
hq_plot_scan_all.py (读 CSV, 用 seaborn 画图)
     │
     ▼
plots/{Lang|Vis}_{Spectrum|Baseline|Inner|Ambient}_{RelErr|MSE|MAE}.png
```

文件时间戳 (CSV 与所有 PNG 都是 `2026-05-01 23:34:25`) 也证实它们是同一次运行
产生的。

### 图片命名规则

文件名格式 `{Domain}_{MetricFamily}_{ErrorType}.png`：

- **Domain**: `Lang` (language layers) 或 `Vis` (vision layers)
- **MetricFamily** (来自 `hq_plot_scan_all.py` 的 `metrics_map`):
  - `Spectrum_RelErr` — 奇异值谱变化 (能量变化)
  - `Baseline_RelErr/MSE/MAE` — 真实权重漂移
  - `Inner_RelErr/MSE/MAE` — Inner Stiefel 假设的残差 (只允许子空间内旋转)
  - `Ambient_RelErr/MSE/MAE` — Ambient Stiefel 假设的残差 (允许子空间漂移，
    但保持原 sigma)
- 注意 `Spectrum` 只有 `RelErr` (因为它本来就是范数比值)，所以共有
  `4 metrics × 3 errors - 2` ≈ 10 张/domain，两个 domain 共 20 张。

每张图内部由 `sns.relplot(col="Transition")` 切成 3 个子图，分别对应：
`Base→Stage1`, `Stage1→Stage2`, `Base→Stage2`。

---

## 2. 重新画这些图片需要做哪些步骤？

### 前置条件

环境用 conda best176

- Python 环境已装好 `torch`, `transformers`, `scipy`, `pandas`,
  `seaborn`, `matplotlib`。

这个你从 huggingface 自己下载

- 能从 HuggingFace 下载/已经缓存好下面 3 个模型 (因为脚本默认这三个):
  - `Qwen/Qwen2.5-VL-3B-Instruct` (base)
  - `IffYuan/Embodied-R1-3B-Stage1` (Stage1)
  - `IffYuan/Embodied-R1-3B-v1` (Stage2)
- 一台**有较多 CPU 内存**的机器: `hq_scan_all.py` 把 3 个模型都用 **float64**
  加载到 **CPU**，3B 参数 × 8 byte × 3 个模型 ≈ 72 GB 内存(粗估)。如果想换
  其它模型，注意这一点。

然后我需要你给我在这个机器上从头开始 reproduce 这些结果，reproduce 给我放到 /ssd1/zhizhou/workspace/rotation-project/replace/plot_reproduce 这个 folder 下边去

### Step-by-step

#### Step 1 — (按需) 修改要分析的模型 / 层范围

打开 `hq_scan_all.py`，文件顶部有这些常量：

```python
model_name_0 = "Qwen/Qwen2.5-VL-3B-Instruct"    # Base
model_name_1 = "IffYuan/Embodied-R1-3B-Stage1"  # Stage 1
model_name_2 = "IffYuan/Embodied-R1-3B-v1"      # Stage 2

total_vis_layers = 32
total_lang_layers = 36
VISION_LAYERS_TO_SCAN = [i for i in range(total_vis_layers)]
LANG_LAYERS_TO_SCAN   = [i for i in range(total_lang_layers)]

VISION_MODULES = ["attn.qkv", "attn.proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
LANG_MODULES   = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                  "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

OUTPUT_CSV = "stiefel_analysis_metrics.csv"
```

如果只是要"复刻原图"，**什么都不用改**，直接跳到 Step 2。
如果要换模型对，需要：
- 改 3 个 `model_name_*`
- 改 `total_vis_layers` / `total_lang_layers` (匹配新模型架构)
- 如果新模型不是 VL 结构，可能还要改 `get_module_weight()` 里的属性访问路径
  (`model.visual.blocks` / `model.language_model.layers` 这些)。

#### Step 2 — 跑分析脚本生成 CSV

```bash
cd /ssd1/zhizhou/workspace/rotation-project/replace
python hq_scan_all.py
```

这一步会：
1. 把 3 个模型用 float64 加载到 CPU；
2. 对每个 `(domain, layer_idx, submodule)` 组合：
   - 取出 W0/W1/W2 三个权重矩阵；
   - 各自 SVD；
   - 对 3 个 transition (`Base→Stage1`, `Stage1→Stage2`, `Base→Stage2`)
     算 4 类指标 (Spectrum / Baseline / Inner Stiefel / Ambient Stiefel)；
3. 写 `stiefel_analysis_metrics.csv` 到当前目录；
4. 在终端打 `groupby("Transition").mean()` 的总结。

⚠️ 这一步**很慢**(模型加载 + 大量 SVD)，建议在后台跑或 nohup。

#### Step 3 — 跑绘图脚本生成 PNG

```bash
python hq_plot_scan_all.py
```

这一步会：
1. 读 `stiefel_analysis_metrics.csv`；
2. 按 `Lang` / `Vis` 两个 domain × 10 个 metric 各画一张大图
   (每张内部 facet 成 3 个 transition 子图)；
3. 保存到 `plots/` 目录(目录不存在会自动创建)，dpi=150。

跑完后 `plots/` 下应该出现 20 张 PNG，文件名跟现有的完全一致。

---

## 3. 一些值得注意的细节

- **CSV 是关键中间产物**: 只要 CSV 在，重画图只需要几秒；改图样式
  (颜色/scale/标题) 只动 `hq_plot_scan_all.py` 即可，不用重跑 `hq_scan_all.py`。
- **数值精度**: 整个分析强制用 `float64` + CPU + `scipy.linalg.svd`，目的是
  让 Inner/Ambient 的残差能小到 ~1e-12 量级而不被精度噪声淹没；图里那条
  红色虚线 `1e-7` 就是 float32 的精度参考线 (代码 line 83)。
- **`hq_plot_scan_all.py` 的 facet 行为**: 用 `sharey=False`，所以每个
  transition 子图 y 轴范围是独立的；横轴是 `Layer Index`，每条线对应一个
  `ModuleType` (q_proj / k_proj / mlp.up_proj …)。
- **目录里其它疑似相关的脚本** (`hq_test*.py`, `svd_compare*.py`,
  `rotation_*.png`) 跟 `plots/` 这批图**没关系** —— 它们是更早阶段或别的
  实验产物，重画 plots/ 里的图只需要 `hq_scan_all.py` + `hq_plot_scan_all.py`
  这两个脚本。

---

## TL;DR

```bash
# 一条命令复刻 plots/ 里所有图
python hq_scan_all.py && python hq_plot_scan_all.py
```

第一步生成 `stiefel_analysis_metrics.csv` (慢，要加载 3 个模型 + 大量 SVD)，
第二步读 CSV 画 20 张图到 `plots/`(快)。
