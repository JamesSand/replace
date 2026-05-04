---
title: 任务：在hq_scan_all.py中新增四模型距离计算
type: task
priority: high
created: 2026-05-03
---

# 任务：新增四模型距离（e_src, e_L, e_R, e_tgt）计算

## 任务概述

在现有的 `hq_scan_all.py` 中扩展 `analyze_triplet()` 函数，新增计算四个模型类的距离指标。这些指标用于论文Section 4.2中的Table 4.1，以验证RLVR使用的是ambient transport还是inner rotation。

---

## 要改动的文件

**路径**：`D:\2026Summer\iso-project\replace\hq_scan_all.py`

**改动函数**：`analyze_triplet()`（当前第104-228行）

---

## 改动详情

### 1. 修改函数签名

**当前**：
```python
def analyze_triplet(name, W0, W1, W2, results_list):
```

**改为**：
```python
def analyze_triplet(name, W0, W1, W2, results_list, domain="Unknown"):
```

在函数内部新增：
```python
row["Domain"] = domain
```

### 2. 在analyze_triplet中新增计算代码

在现有的Ambient Stiefel计算（第154-156行）之后，新增以下代码块：

```python
# --- NEW: Four Model Class Distances (Corollary 3.3) ---
# 计算四个模型类的投影子距离
r = len(S_src)  # full rank

# 建立投影矩阵
P_src = U_src[:, :r] @ U_src[:, :r].T  # 左投影，shape (m, m)
Q_src = V_src[:, :r] @ V_src[:, :r].T  # 右投影，shape (n, n)

# 四个距离计算（都使用Frobenius范数）
# e_src：源拟合（两侧都投影）
e_src = np.linalg.norm(W_tgt - P_src @ W_tgt @ Q_src, 'fro')

# e_L：左混合（左侧投影到源，右侧自由）
e_L = np.linalg.norm(W_tgt - P_src @ W_tgt, 'fro')

# e_R：右混合（右侧投影到源，左侧自由）
e_R = np.linalg.norm(W_tgt - W_tgt @ Q_src, 'fro')

# e_tgt：目标输运（两侧自由，冻结源的谱值）
e_tgt = np.linalg.norm(W_tgt - U_tgt @ np.diag(S_src) @ V_tgt.T, 'fro')

# 添加到CSV行
row["e_src"] = e_src
row["e_L"] = e_L
row["e_R"] = e_R
row["e_tgt"] = e_tgt
```

### 3. 修改调用处

**在main()函数中，修改两处analyze_triplet的调用**：

#### Language Layers（第254行附近）

**当前**：
```python
analyze_triplet(name, W0, W1, W2, results)
```

**改为**：
```python
analyze_triplet(name, W0, W1, W2, results, domain="Lang")
```

#### Vision Layers（第266行附近）

**当前**：
```python
analyze_triplet(name, W0, W1, W2, results)
```

**改为**：
```python
analyze_triplet(name, W0, W1, W2, results, domain="Vis")
```

---

## 新增CSV列

改动后的CSV会新增以下5列：

| 列名 | 数据类型 | 说明 |
|-----|---------|------|
| Domain | str | "Lang" 或 "Vis" |
| e_src | float | 源拟合距离（两侧都投影在源子空间） |
| e_L | float | 左混合距离（左侧投影在源，右侧自由） |
| e_R | float | 右混合距离（右侧投影在源，左侧自由） |
| e_tgt | float | 目标输运距离（两侧自由到目标，冻结源谱值） |

---

## 四个距离的含义

这四个距离对应于Corollary 3.3中的四个模型类，用来验证checkpoint转移是否由ambient transport（两侧都旋转）还是inner rotation（在固定子空间内旋转）主导：

| 距离 | 定义 | 模型类 | 含义 |
|-----|------|--------|------|
| e_src | \|\|W_i - P_j W_i Q_j\|\|_F | I(j) | 如果变化只是内部旋转，e_src应该很小 |
| e_L | \|\|W_i - P_j W_i\|\|_F | L(j→i) | 如果左侧不变，e_L应该很小 |
| e_R | \|\|W_i - W_i Q_j\|\|_F | R(j→i) | 如果右侧不变，e_R应该很小 |
| e_tgt | \|\|W_i - U_i Σ_j V_i^T\|\|_F | A(j→i) | 如果两侧都能自由旋转，e_tgt应该很小 |

**预期结果**：如果RLVR是ambient transport，则 e_src ≈ e_L ≈ e_R >> e_tgt

---

## 测试步骤

### Step 1: 小规模测试

1. 修改hq_scan_all.py顶部的配置：
```python
VISION_LAYERS_TO_SCAN = [0, 1]  # 只扫2层
LANG_LAYERS_TO_SCAN = [0, 1]    # 只扫2层
```

2. 运行：
```bash
cd D:\2026Summer\iso-project\replace
python hq_scan_all.py
```

3. 验证点：
   - ✅ 无报错运行完成
   - ✅ `stiefel_analysis_metrics.csv` 中出现5个新列（Domain, e_src, e_L, e_R, e_tgt）
   - ✅ 数值合理：e_src, e_L, e_R在O(1)量级，e_tgt在O(0.01-0.1)量级
   - ✅ 模式验证：e_src ≈ e_L ≈ e_R > e_tgt（大致相等然后显著小于）

### Step 2: 全量运行

1. 恢复配置为全量扫描：
```python
VISION_LAYERS_TO_SCAN = [i for i in range(total_vis_layers)]
LANG_LAYERS_TO_SCAN = [i for i in range(total_lang_layers)]
```

2. 运行全量计算

3. 验证输出CSV的行数和新列完整性

---

## 关键注意事项

1. **保持原始形式**：所有四个距离都使用原始的Frobenius范数计算，不做归一化。这样与现有的Inner/Ambient MSE保持一致。

2. **e_tgt的计算**：在W矩阵维度上计算，而非在谱维度上。公式是：
   ```python
   e_tgt = np.linalg.norm(W_tgt - U_tgt @ np.diag(S_src) @ V_tgt.T, 'fro')
   ```
   这表示用源模型的谱值但允许两侧自由旋转到目标位置。

3. **Domain标记**：必须正确标记Lang和Vis，这样后续生成Table 4.1时能正确区分。

4. **现有代码保留**：不删除任何现有的Inner/Ambient计算，只是新增这四个距离。

5. **打印输出**（可选但推荐）：可以在现有的打印语句后新增打印这四个距离，便于调试：
   ```python
   print(f"    e_src: {e_src:.1e} | e_L: {e_L:.1e} | e_R: {e_R:.1e} | e_tgt: {e_tgt:.1e}")
   ```

---

## 完成标志

改动完成并通过小规模测试后，应该：
1. ✅ 代码无报错
2. ✅ CSV包含新列
3. ✅ 数值分布符合预期（e_src ≈ e_L ≈ e_R >> e_tgt）
4. ✅ Domain列正确标记（Lang/Vis）

然后可以运行全量计算生成最终数据。

最后的 table 我要一种这个形式的



## 后续：Table 4.1 格式

### 方案 A：按Transition分组展示

**Table 4.1: Checkpoint-level Model Class Distances**

#### Base → Stage1

| Layer | Module | Domain | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-------|-----|-----|-------|
| L0 | self_attn.q_proj | Lang | — | — | — | — |
| L10 | self_attn.q_proj | Lang | — | — | — | — |
| L20 | self_attn.q_proj | Lang | — | — | — | — |
| L35 | self_attn.q_proj | Lang | — | — | — | — |
| L0 | attn.qkv | Vis | — | — | — | — |
| L15 | attn.qkv | Vis | — | — | — | — |
| L31 | attn.qkv | Vis | — | — | — | — |

#### Stage1 → Stage2

| Layer | Module | Domain | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-------|-----|-----|-------|
| L0 | self_attn.q_proj | Lang | — | — | — | — |
| L10 | mlp.gate_proj | Lang | — | — | — | — |
| L20 | mlp.up_proj | Lang | — | — | — | — |
| L35 | mlp.down_proj | Lang | — | — | — | — |
| L0 | attn.proj | Vis | — | — | — | — |
| L15 | mlp.gate_proj | Vis | — | — | — | — |
| L31 | mlp.up_proj | Vis | — | — | — | — |

#### Base → Stage2

| Layer | Module | Domain | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-------|-----|-----|-------|
| L0 | self_attn.q_proj | Lang | — | — | — | — |
| L10 | self_attn.k_proj | Lang | — | — | — | — |
| L20 | self_attn.v_proj | Lang | — | — | — | — |
| L35 | self_attn.o_proj | Lang | — | — | — | — |
| L0 | mlp.gate_proj | Vis | — | — | — | — |
| L15 | mlp.up_proj | Vis | — | — | — | — |
| L31 | mlp.down_proj | Vis | — | — | — | — |

---


你最终给我 deliver 一个上边的 table，只是把其中的数都给我填上

所有的文件和改动你都给我放到 002 这个 folder 下边，




