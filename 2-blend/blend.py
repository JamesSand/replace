# svd_sigma_blend_cache_merge.py
# Two-stage pipeline:
#   1) cache:  compute SVD for (before, after), write cache to disk (per-layer)
#   2) merge:  load cache, for each alpha do sigma blend, reconstruct weights,
#              save as HuggingFace model folder (model + tokenizer)
#
# Default cache format stores:
#   U_after, Vh_after, S_after, S_before  (=> enough for "sigma blending in after basis")
#
# NOTE on your logic:
#   - Storing full SVD (U,Vh) for BOTH models is extremely disk-heavy.
#     Economy SVD stores ~ (U + Vh) sizes comparable to original weight, so full caching for both ≈ ~4x weight size.
#   - If you only blend singular values (sigma), you only need:
#        (U_after, Vh_after, S_after) and S_before
#     That still respects "do SVD for both models" but avoids storing before's U/Vh.

import os
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


torch.set_float32_matmul_precision("high")


def safe_key_to_filename(key: str) -> str:
    return key.replace(".", "__").replace("/", "_slash_")


@torch.no_grad()
def svd_full_fp32_cpu(W: torch.Tensor):
    assert W.dim() == 2
    W = W.to(dtype=torch.float32, device="cpu", copy=True)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return U, S, Vh


def iter_2d_keys(sd_before, sd_after):
    for k, Wa in sd_after.items():
        if k not in sd_before:
            yield k, None, None, "missing_in_before"
            continue
        Wb = sd_before[k]
        if Wa.dim() != 2 or Wb.dim() != 2:
            yield k, Wa, Wb, f"ndim_after={Wa.dim()}, ndim_before={Wb.dim()}"
            continue
        if Wa.shape != Wb.shape:
            yield k, Wa, Wb, f"shape_mismatch {tuple(Wb.shape)} vs {tuple(Wa.shape)}"
            continue
        
        if "embed" in k.lower():
            yield k, Wa, Wb, "skip_embed"
            continue
        
        yield k, Wa, Wb, None


def stage_cache(
    model_before_id: str,
    model_after_id: str,
    cache_dir: str,
    device_map: str = "cpu",
    low_cpu_mem: bool = True,
    store_full_before: bool = False,
):
    cache_dir = Path(cache_dir)
    layers_dir = cache_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cache] before={model_before_id}")
    print(f"[cache] after ={model_after_id}")
    print(f"[cache] dir   ={cache_dir}")
    print(f"[cache] store_full_before={store_full_before}")

    print("[load] loading models on CPU (fp32)...")
    m_before = AutoModelForCausalLM.from_pretrained(
        model_before_id,
        torch_dtype=torch.float32,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem,
        trust_remote_code=True,
    )
    m_after = AutoModelForCausalLM.from_pretrained(
        model_after_id,
        torch_dtype=torch.float32,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem,
        trust_remote_code=True,
    )

    print("[state_dict] extracting...")
    sd_before = {k: v.detach().to(torch.float32).cpu() for k, v in m_before.state_dict().items()}
    sd_after  = {k: v.detach().to(torch.float32).cpu() for k, v in m_after.state_dict().items()}

    blended_keys = []
    skipped = []

    for k, Wa, Wb, reason in iter_2d_keys(sd_before, sd_after):
        if reason is not None:
            skipped.append((k, reason))
            continue

        m, n = Wa.shape
        print(f"[svd-cache] {k} shape={m}x{n}")

        # Do SVD on BOTH models:
        U_a, S_a, Vh_a = svd_full_fp32_cpu(Wa)
        # For "sigma blending in after basis", we only need S_b (but we still *do* before SVD here).
        if store_full_before:
            U_b, S_b, Vh_b = svd_full_fp32_cpu(Wb)
        else:
            _, S_b, _ = svd_full_fp32_cpu(Wb)
            U_b = Vh_b = None

        # Save cache for this layer
        pt_path = layers_dir / f"{safe_key_to_filename(k)}.pt"
        payload = {
            "key": k,
            "shape": torch.tensor([m, n], dtype=torch.int64),

            # always store AFTER basis
            "U_after": U_a.cpu(),
            "Vh_after": Vh_a.cpu(),
            "S_after": S_a.cpu(),

            # store BEFORE sigma
            "S_before": S_b.cpu(),
        }
        if store_full_before:
            payload["U_before"] = U_b.cpu()
            payload["Vh_before"] = Vh_b.cpu()

        torch.save(payload, pt_path)

        blended_keys.append(k)

        del U_a, S_a, Vh_a, S_b, U_b, Vh_b

    meta = {
        "model_before": model_before_id,
        "model_after": model_after_id,
        "cache_dir": str(cache_dir),
        "layers_dir": str(layers_dir),
        "store_full_before": store_full_before,
        "dtype": "float32",
        "cpu_only": True,
        "cached_2d_keys_count": len(blended_keys),
        "skipped_count": len(skipped),
        "cached_2d_keys": blended_keys,
        "skipped": skipped,
    }
    with open(cache_dir / "svd_cache_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[cache] done. cached={len(blended_keys)}, skipped={len(skipped)}")
    print(f"[cache] meta -> {cache_dir / 'svd_cache_meta.json'}")


@torch.no_grad()
def reconstruct_in_after_basis(U_a, Vh_a, S_blend):
    """
    W = U * diag(S) * Vh
    Avoid explicit diag(S): (U * S[None, :]) @ Vh
    """
    return (U_a * S_blend.unsqueeze(0)) @ Vh_a


def stage_merge(
    model_after_id: str,
    cache_dir: str,
    output_dir: str,
    alphas,
    device_map: str = "cpu",
    low_cpu_mem: bool = True,
):
    cache_dir = Path(cache_dir)
    layers_dir = cache_dir / "layers"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[merge] after={model_after_id}")
    print(f"[merge] cache={cache_dir}")
    print(f"[merge] out  ={output_dir}")
    print(f"[merge] alphas={alphas}")

    # Load tokenizer once, and save into each alpha folder.
    # Usually before/after share tokenizer; safest: use after's tokenizer for AutoModel load.
    print("[tokenizer] loading...")
    tok = AutoTokenizer.from_pretrained(model_after_id, trust_remote_code=True)

    layer_files = sorted(layers_dir.glob("*.pt"))
    if not layer_files:
        raise RuntimeError(f"No cache files found in {layers_dir}")

    for alpha in alphas:
        print(f"\n===== alpha = {alpha:.3f} =====")

        # Load a fresh "after" model each alpha to avoid any chance of leftover weights.
        print("[load] loading base after model (fp32/cpu)...")
        m_after = AutoModelForCausalLM.from_pretrained(
            model_after_id,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem,
            trust_remote_code=True,
        )
        m_after.to("cpu")

        sd = m_after.state_dict()
        updated = 0
        skipped = []

        for fp in layer_files:
            pack = torch.load(fp, map_location="cpu")
            k = pack["key"]

            if k not in sd:
                skipped.append((k, "missing_in_model_after_runtime"))
                continue

            U_a = pack["U_after"].to(torch.float32)
            Vh_a = pack["Vh_after"].to(torch.float32)
            S_a = pack["S_after"].to(torch.float32)
            S_b = pack["S_before"].to(torch.float32)

            if S_a.shape != S_b.shape:
                skipped.append((k, f"sigma_mismatch {tuple(S_b.shape)} vs {tuple(S_a.shape)}"))
                continue

            # sigma blend
            S_blend = alpha * S_a + (1.0 - alpha) * S_b

            # reconstruct
            W_new = reconstruct_in_after_basis(U_a, Vh_a, S_blend).to(torch.float32)

            # write in-place (avoids building a huge new state_dict)
            if sd[k].shape != W_new.shape:
                skipped.append((k, f"shape_mismatch_runtime {tuple(sd[k].shape)} vs {tuple(W_new.shape)}"))
                continue
            sd[k].copy_(W_new)

            updated += 1
            del U_a, Vh_a, S_a, S_b, S_blend, W_new

        out_alpha = output_dir / f"alpha_{alpha:.3f}"
        out_alpha.mkdir(parents=True, exist_ok=True)

        print(f"[save] model -> {out_alpha}")
        m_after.save_pretrained(out_alpha, safe_serialization=True)
        tok.save_pretrained(out_alpha)

        meta = {
            "model_after_base": model_after_id,
            "cache_dir": str(cache_dir),
            "alpha": float(alpha),
            "updated_keys": updated,
            "skipped_count": len(skipped),
            "skipped": skipped,
            "dtype": "float32",
            "cpu_only": True,
            "note": (
                "Weights reconstructed as U_after @ diag(alpha*S_after+(1-alpha)*S_before) @ Vh_after. "
                "Tokenizer saved from model_after_base."
            ),
        }
        with open(out_alpha / "svd_blend_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[done] alpha={alpha:.3f} updated={updated} skipped={len(skipped)}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_before", type=str, required=True)
    ap.add_argument("--model_after", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, default="svd_cache")
    ap.add_argument("--output_dir", type=str, default="svd_sigma_blend_outputs")
    ap.add_argument("--stage", type=str, choices=["cache", "merge", "both"], default="both")
    ap.add_argument("--alphas", type=str, default="0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--store_full_before", action="store_true")  # extremely disk-heavy
    ap.add_argument("--device_map", type=str, default="cpu")
    ap.add_argument("--low_cpu_mem", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    if args.stage in ["cache", "both"]:
        stage_cache(
            model_before_id=args.model_before,
            model_after_id=args.model_after,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            low_cpu_mem=args.low_cpu_mem,
            store_full_before=args.store_full_before,
        )

    if args.stage in ["merge", "both"]:
        stage_merge(
            model_after_id=args.model_after,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            alphas=alphas,
            device_map=args.device_map,
            low_cpu_mem=args.low_cpu_mem,
        )


if __name__ == "__main__":
    main()







