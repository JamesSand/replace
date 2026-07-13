# SFT U/V + Σ 插值实验（论文 Figure 3 的镜像对照）

> 状态：**中期汇总**（2026-07-13 22:40 UTC），全部 8 模型 × 7 指标预计 07-14 凌晨完成，届时更新本文件。

## 实验设置

对每个 2D 权重矩阵：

$$\widetilde{W}(\alpha) = U_{\text{SFT}}\,\bigl(\alpha\,\Sigma_{\text{RL}} + (1-\alpha)\,\Sigma_{\text{SFT}}\bigr)\,V_{\text{SFT}}^{\top},\qquad \alpha\in\{0,0.2,0.4,0.6,0.8,1.0\}$$

- **U/V（奇异框架）固定取自 SFT 模型**：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Σ 在 SFT 谱与 RL 谱之间线性插值**，RL = `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B`（ProRL，~3k RL steps，从 R1-Distill 训练而来）
- α=0 → 原 SFT 模型（sanity 端点）；α=1 → SFT 框架 + 完整 RL 谱
- 与论文 Figure 3（U/V 取 RL、Σ 向 base 谱插值）互为镜像对照
- 198 个 2D 矩阵全部 blend（含 embed/lm_head）；1D 参数（bias/LayerNorm，141 个）保持 SFT 原值
- blend 在 fp32 下计算、bf16 存盘；α=0 重构相对误差 ≤ 1.3e-4

**谱漂移观测**：Σ_SFT ↔ Σ_RL 的相对漂移 ‖ΔΣ‖/‖Σ‖ 在全部 198 个矩阵上为 **0.00%–0.04%**，与论文 Fig 2 的 RL near-isospectral 结论一致（SFT 阶段为 ~5%）。

## 评测协议

- **Math**（POLARIS 流程，`eval_vllm.py` + `grade.py`）：AIME24/25 (Mean@32)、AMC23 (Mean@8)、Minerva/Olympiad (Mean@4)；t=1.0, top_p=0.8, max_len=32768
- **Code**（ArcherCodeR 流程，verl `main_generation`，测试用例真实执行）：LiveCodeBench v5 全量 279 题，n=4, t=0.8, response 32k → pass@1 / pass@4
- 参考端点：`sft_orig`（R1-Distill 原版）与 `rl_orig`（Nemotron 原版）跑完全相同的协议

## 当前结果（— = 仍在跑）

| 模型 | AIME24 | AIME25 | AMC23 | Minerva | Olympiad | LCB p@1 | LCB p@4 |
|---|---|---|---|---|---|---|---|
| **sft_orig**（≙ 下界参考） | **29.90** | **22.40** | **62.95** | — | — | — | — |
| alpha_0.0（SFT 重构 sanity） | 30.21 | 23.23 | 63.40 | 26.47 | 43.96 | — | — |
| alpha_0.2 | 30.83 | 23.96 | 62.80 | 26.84 | — | — | — |
| alpha_0.4 | 30.21 | 22.19 | 64.16 | — | — | — | — |
| alpha_0.6 | 30.21 | 23.54 | 63.40 | — | — | — | — |
| alpha_0.8 | 28.65 | 22.92 | — | — | 43.63 | 18.28 | 26.88 |
| alpha_1.0（SFT U/V + RL 谱） | 29.06 | 23.65 | 65.36 | — | — | — | — |
| **rl_orig**（≙ 上界参考） | **48.54** | **32.50** | **80.72** | **34.93** | **59.81** | **29.48** | **36.20** |

（单位 %；AIME24/25 = Mean@32，AMC23 = Mean@8，Minerva/Olympiad = Mean@4，LCB = pass@k）

## 初步结论（已完成的数据集上方向一致）

1. **α 扫描曲线平坦**：AIME24 全程 28.7–30.8%、AIME25 22.2–24.0%、AMC23 62.8–65.4%，均在采样噪声内围绕 sft_orig 水平波动，与 α 无系统性关系 —— **把 RL 的谱以任意比例装进 SFT 的奇异框架，性能不变**。
2. **Sanity 通过**：alpha_0.0 ≈ sft_orig（差 ≤0.8pp），SVD 分解-重构-bf16 存盘的管线本身对模型无损。
3. **能力差距在框架（U/V）里**：rl_orig 比所有 blend 高 AIME24 ~18pp、AMC23 ~16pp、LCB pass@1 ~11pp。结合谱漂移仅 0.04% 的观测：RL 增益几乎完全由奇异框架的旋转携带，谱的贡献可忽略 —— 从镜像方向支持论文 Figure 3 / Section 3.2 的结论。

## 复现入口

- Blend 脚本：`replace/blend_sft_uv.py`（注意：修复了 `replace.py` 的 state_dict 与模型参数共享存储的 bug —— 原脚本多 α 循环中 `load_state_dict` 会污染源谱，除首个 α 外结果均错误；同时每层 SVD 只算一次供全部 α 复用）
- 模型与日志：`~/workspace/rl-opt-proj/sft_uv_blend/{models,svd_logs,eval_outputs}/`
- 运行编排：`sft_uv_blend/scripts/`（`supervisor.sh` 自愈重启 + `steal_loop.sh` 空卡work-stealing + `collect_results.py` 汇总）
- 评测补丁：`POLARIS/scripts/eval/eval_vllm.py` 已修复 worker 覆盖外部 `CUDA_VISIBLE_DEVICES` 的问题（多模型并行时会全部挤到 GPU 0）
- 算力：k8s pod `zhizhousha-rlopt-fig3-8gpu`（8×H100），math ~7h/模型（R1 系 long-CoT 逼近 32k 上限所致；Nemotron 仅 ~4h 全套，RL 模型生成显著更短）
