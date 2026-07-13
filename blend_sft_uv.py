# blend_sft_uv.py
#
# Mirror of the paper's Figure-3 spectrum-replacement experiment, but with the
# SFT model's singular frames:
#
#   W~(alpha) = U_SFT @ diag(alpha * S_RL + (1 - alpha) * S_SFT) @ Vh_SFT
#
#   SFT (frames + spectrum at alpha=0): deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#   RL  (spectrum at alpha=1):          nvidia/Nemotron-Research-Reasoning-Qwen-1.5B
#
# Differences from replace.py:
#   * state_dicts are cloned (replace.py's sd tensors aliased the live model
#     params, so load_state_dict corrupted the source spectra for every alpha
#     after the first one)
#   * SVD is computed once per layer and reused for all alphas (6x less work)
#   * alpha=0.0 reconstruction is checked against the original SFT weights
#   * outputs are saved in bf16 (same dtype as the source checkpoints),
#     with tokenizer + generation config copied in so vllm can load the dir

import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_SFT = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"   # U, V, and S at alpha=0
MODEL_RL = "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B"  # S at alpha=1

BASE_DIR = "/home/zhizhousha/workspace/rl-opt-proj/sft_uv_blend"
OUT_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "svd_logs")
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def safe_key(key: str) -> str:
    return key.replace(".", "__").replace("/", "_slash_")


@torch.no_grad()
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    print(f"[load] SFT: {MODEL_SFT}")
    model_sft = AutoModelForCausalLM.from_pretrained(MODEL_SFT, torch_dtype=torch.float32)
    print(f"[load] RL : {MODEL_RL}")
    model_rl = AutoModelForCausalLM.from_pretrained(MODEL_RL, torch_dtype=torch.float32)

    # clone() so nothing here aliases live parameter storage
    sd_sft = {k: v.detach().clone().float().cpu() for k, v in model_sft.state_dict().items()}
    sd_rl = {k: v.detach().clone().float().cpu() for k, v in model_rl.state_dict().items()}
    del model_rl

    # one blended state_dict per alpha; non-2D / unmatched keys stay SFT
    new_sds = {a: {k: v.clone() for k, v in sd_sft.items()} for a in ALPHAS}

    blended, skipped = [], []
    recon_err_max = 0.0
    for k, w_sft in sd_sft.items():
        if w_sft.dim() != 2:
            skipped.append((k, f"ndim={w_sft.dim()}"))
            continue
        if k not in sd_rl:
            skipped.append((k, "missing_in_rl"))
            continue
        w_rl = sd_rl[k]
        if w_sft.shape != w_rl.shape:
            skipped.append((k, f"shape {tuple(w_sft.shape)} vs {tuple(w_rl.shape)}"))
            continue

        u, s_sft, vh = torch.linalg.svd(w_sft, full_matrices=False)
        s_rl = torch.linalg.svdvals(w_rl)
        assert s_sft.shape == s_rl.shape, f"sigma shape mismatch at {k}"

        for a in ALPHAS:
            s_blend = a * s_rl + (1.0 - a) * s_sft
            new_sds[a][k] = u @ torch.diag(s_blend) @ vh

        # sanity: alpha=0 must reproduce the SFT weight up to SVD roundoff
        err = (new_sds[0.0][k] - w_sft).abs().max().item()
        scale = w_sft.abs().max().item()
        rel = err / max(scale, 1e-12)
        recon_err_max = max(recon_err_max, rel)
        assert rel < 1e-3, f"alpha=0 reconstruction off at {k}: rel={rel:.3e}"

        drift = ((s_rl - s_sft).norm() / s_sft.norm()).item()
        torch.save(
            {"sigma_sft": s_sft, "sigma_rl": s_rl, "shape": list(w_sft.shape), "drift": drift},
            os.path.join(LOG_DIR, f"{safe_key(k)}.pt"),
        )
        blended.append(k)
        print(f"[svd] {k} {tuple(w_sft.shape)} drift={drift:.4f} a0_rel_err={rel:.2e}", flush=True)

    print(f"[svd done] blended={len(blended)} skipped={len(skipped)} max_a0_rel_err={recon_err_max:.2e}")

    tok = AutoTokenizer.from_pretrained(MODEL_SFT)
    for a in ALPHAS:
        out = os.path.join(OUT_DIR, f"alpha_{a:.1f}")
        missing, unexpected = model_sft.load_state_dict(new_sds[a], strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        if missing:
            print(f"[warn][alpha={a:.1f}] missing keys: {missing[:5]}")
        model_sft.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
        model_sft.to(torch.float32)  # back to fp32 for the next load_state_dict
        tok.save_pretrained(out)
        with open(os.path.join(out, "svd_blend_meta.json"), "w") as f:
            json.dump(
                {
                    "formula": "U_SFT @ diag(alpha*S_RL + (1-alpha)*S_SFT) @ Vh_SFT",
                    "model_sft": MODEL_SFT,
                    "model_rl": MODEL_RL,
                    "alpha": a,
                    "blended_keys_count": len(blended),
                    "skipped_keys": skipped,
                    "save_dtype": "bfloat16",
                },
                f,
                indent=2,
            )
        print(f"[save] alpha={a:.1f} -> {out}", flush=True)

    print("[all done]", ", ".join(f"{a:.1f}" for a in ALPHAS))


if __name__ == "__main__":
    main()
