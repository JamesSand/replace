

# before RL: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# after RL: agentica-org/DeepScaleR-1.5B-Preview


# svd_sigma_blend_fp32_multi_alpha_cpu.py
import os
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

# ===================== 可配置区 =====================
# MODEL_BEFORE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"     # before RL
# # MODEL_AFTER  = "agentica-org/DeepScaleR-1.5B-Preview"          # after  RL
# # BASE_OUTPUT_DIR = "DeepScaleR-1.5B-Preview_sigma-blend_fp32"   # 根输出目录

MODEL_BEFORE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
MODEL_AFTER = "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B"
BASE_OUTPUT_DIR = "Nemotron-Research-Reasoning-Qwen-1.5B_sigma-blend_fp32"

# MODEL_BEFORE = "Qwen/Qwen2.5-Math-1.5B"
# MODEL_AFTER = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# BASE_OUTPUT_DIR = "DeepSeek-R1-Distill-Qwen-1.5B_sigma-blend_fp32"

BASE_LOG_DIR    = "svd_logs"                                   # 根日志目录
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]                        # 要遍历的 alpha
DEVICE_MAP   = "cpu"                                           # 全流程 CPU
LOW_CPU_MEM  = True
# ===================================================

# 全局严格 fp32
torch.set_float32_matmul_precision("high")

def safe_key_to_filename(key: str) -> str:
    return key.replace(".", "__").replace("/", "_slash_")

@torch.no_grad()
def svd_full_fp32_cpu(W: torch.Tensor):
    """
    在 CPU 上对 2D 权重做 SVD（严格 fp32）：
    W (m x n) = U (m x k) @ diag(S) (k x k) @ Vh (k x n), k = min(m, n)
    """
    assert W.dim() == 2
    W = W.to(dtype=torch.float32, device="cpu", copy=True)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return U, S, Vh

def blend_and_save_once(model_after, sd_before, sd_after, alpha: float,
                        base_output_dir: str, base_log_dir: str):
    """
    针对单个 alpha：
      - 遍历各 2D 权重做 SVD
      - S_blend = alpha * S_after + (1 - alpha) * S_before
      - 用 U_a @ diag(S_blend) @ Vh_a 重构
      - 保存模型到 BASE_OUTPUT_DIR/alpha_{:.1f}/
      - 每层奇异值记录到 BASE_LOG_DIR/alpha_{:.1f}/
    """
    out_dir = os.path.join(base_output_dir, f"alpha_{alpha:.1f}")
    log_dir = os.path.join(base_log_dir,    f"alpha_{alpha:.1f}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    blended_keys, skipped_keys, per_layer_meta = [], [], {}

    # 复制一份 after 的 sd 作为本次要写回的权重容器
    new_sd = dict(sd_after)

    for k, W_after in sd_after.items():
        if k not in sd_before:
            skipped_keys.append((k, "missing_in_before"))
            continue

        W_before = sd_before[k]
        if W_after.dim() != 2 or W_before.dim() != 2:
            skipped_keys.append((k, f"ndim={W_after.dim()}"))
            continue

        if W_after.shape != W_before.shape:
            skipped_keys.append((k, f"shape_mismatch {tuple(W_before.shape)} vs {tuple(W_after.shape)}"))
            continue

        m, n = W_after.shape
        print(f"[svd][alpha={alpha:.1f}] {k} shape={m}x{n}")

        # after 的 U_a, S_a, Vh_a； before 只取 S_b
        U_a, S_a, Vh_a = svd_full_fp32_cpu(W_after)
        _,  S_b,  _    = svd_full_fp32_cpu(W_before)

        if S_a.shape != S_b.shape:
            # assert False, f"sigma shape mismatch for key {k}: {S_a.shape} vs {S_b.shape}"
            skipped_keys.append((k, f"sigma_mismatch {tuple(S_b.shape)} vs {tuple(S_a.shape)}"))
            continue

        # 线性插值奇异值并重构
        S_blend = alpha * S_a + (1.0 - alpha) * S_b
        W_new = (U_a @ torch.diag(S_blend) @ Vh_a).to(torch.float32)

        # 写回 CPU state_dict
        new_sd[k] = W_new.detach().cpu()
        blended_keys.append(k)

        # 保存奇异值日志（pt）
        log_path = os.path.join(log_dir, f"{safe_key_to_filename(k)}.pt")
        torch.save(
            {
                "sigma_before": S_b.detach().cpu(),
                "sigma_after":  S_a.detach().cpu(),
                "sigma_blend":  S_blend.detach().cpu(),
                "shape":        torch.tensor([m, n]),
                "alpha":        torch.tensor([alpha], dtype=torch.float32),
            },
            log_path,
        )
        per_layer_meta[k] = {
            "shape": [m, n],
            "sigma_len": int(S_a.numel()),
            "log_file": log_path,
            "svd_device": "cpu",
        }

        # 释放中间变量
        del U_a, S_a, Vh_a, S_b, S_blend, W_new

    # 将 new_sd 写回 after 模型对象，并保存到 out_dir
    missing, unexpected = model_after.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[warn][alpha={alpha:.1f}] missing keys when loading: {len(missing)}")
    if unexpected:
        print(f"[warn][alpha={alpha:.1f}] unexpected keys when loading: {len(unexpected)}")

    model_after.to("cpu")
    model_after.save_pretrained(out_dir, safe_serialization=True)

    meta = {
        "model_before": MODEL_BEFORE,
        "model_after": MODEL_AFTER,
        "alpha": alpha,
        "output_dir": out_dir,
        "log_dir": log_dir,
        "blended_keys_count": len(blended_keys),
        "skipped_keys_count": len(skipped_keys),
        "blended_keys": blended_keys,
        "skipped_keys": skipped_keys,
        "per_layer_meta": per_layer_meta,
        "dtype": "float32",
        "cpu_only": True,
    }
    with open(os.path.join(out_dir, "svd_blend_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[save][alpha={alpha:.1f}] done. model -> {out_dir} | logs -> {log_dir}")

def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(BASE_LOG_DIR, exist_ok=True)

    print(f"[config] before={MODEL_BEFORE}")
    print(f"[config] after ={MODEL_AFTER}")
    print(f"[config] alphas={ALPHAS}")
    print(f"[config] out   ={BASE_OUTPUT_DIR}")
    print(f"[config] logs  ={BASE_LOG_DIR}")
    print("[cpu] using CPU only, strict fp32.")

    # 1) 加载两个模型（CPU / fp32）
    print("[load] loading models on CPU (fp32)...")
    model_before = AutoModelForCausalLM.from_pretrained(
        MODEL_BEFORE, torch_dtype=torch.float32, device_map=DEVICE_MAP,
        low_cpu_mem_usage=LOW_CPU_MEM, trust_remote_code=True
    )
    model_after = AutoModelForCausalLM.from_pretrained(
        MODEL_AFTER, torch_dtype=torch.float32, device_map=DEVICE_MAP,
        low_cpu_mem_usage=LOW_CPU_MEM, trust_remote_code=True
    )

    # 2) 提取 fp32 / CPU 的 state_dict（仅做一次，后续复用）
    print("[state_dict] extracting...")
    sd_before = {k: v.detach().to(torch.float32).cpu() for k, v in model_before.state_dict().items()}
    sd_after  = {k: v.detach().to(torch.float32).cpu()  for k, v in model_after.state_dict().items()}

    # 3) 依次处理每个 alpha
    for a in ALPHAS:
        print(f"\n===== Processing alpha = {a:.1f} =====")
        blend_and_save_once(model_after, sd_before, sd_after, a, BASE_OUTPUT_DIR, BASE_LOG_DIR)

    print("\n[all done] Generated models for alphas:", ", ".join(f"{a:.1f}" for a in ALPHAS))

if __name__ == "__main__":
    main()
