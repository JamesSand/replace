import torch
import numpy as np
import scipy.linalg
import pandas as pd
import os
from transformers import AutoModelForVision2Seq

# --- Configuration ---
model_name_0 = "Qwen/Qwen2.5-VL-3B-Instruct"    # Base
model_name_1 = "IffYuan/Embodied-R1-3B-Stage1" # Stage 1
model_name_2 = "IffYuan/Embodied-R1-3B-v1"     # Stage 2 (Final)

# Define what to scan
total_vis_layers = 32  
total_lang_layers = 36
VISION_LAYERS_TO_SCAN = [i for i in range(total_vis_layers)] # Add more indices as needed
# LANG_LAYERS_TO_SCAN = [0, 10, 20]   # Add more indices as needed
LANG_LAYERS_TO_SCAN = [i for i in range(total_lang_layers)]   # Add more indices as needed

# Modules to check
VISION_MODULES = [
    "attn.qkv", 
    "attn.proj", 
    "mlp.gate_proj", 
    "mlp.up_proj",
    "mlp.down_proj"
]
# Note: Qwen-VL Vision naming might vary, adjust if needed (e.g. mlp.c_fc, mlp.c_proj)

LANG_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj", 
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj"
]

OUTPUT_CSV = "stiefel_analysis_metrics.csv"

# --- Helper Functions ---

def get_module_weight(model, layer_idx, submodule_name, is_vision=False):
    """Robust extraction for Qwen-VL structure."""
    try:
        if is_vision:
            # Qwen-VL Vision Part
            # Usually: model.visual.blocks[i].submodule
            if hasattr(model, "visual"):
                blocks = model.visual.blocks
            elif hasattr(model.model, "visual"): # Some HF versions
                blocks = model.model.visual.blocks
            else:
                return None
            
            block = blocks[layer_idx]
            module = block
            for token in submodule_name.split('.'):
                module = getattr(module, token)
            return module.weight.detach().to(dtype=torch.float64).cpu().numpy()
            
        else:
            # Qwen-VL Language Part
            if hasattr(model, "language_model"):
                layers = model.language_model.layers
            elif hasattr(model, "model"):
                layers = model.model.layers
            else:
                layers = model.layers
            
            layer = layers[layer_idx]
            module = layer
            for token in submodule_name.split('.'):
                module = getattr(module, token)
            return module.weight.detach().to(dtype=torch.float64).cpu().numpy()

    except Exception as e:
        # print(f"  [Skip] Could not access {submodule_name} in L{layer_idx}: {e}")
        return None

def get_svd(W):
    """Returns U, S, Vh (as float64)."""
    return scipy.linalg.svd(W, full_matrices=False)

def calc_metrics(W_target, W_pred):
    """Returns dict of MSE, MAE, RelErr."""
    diff = W_target - W_pred
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    
    norm_target = np.linalg.norm(W_target, 'fro')
    norm_diff = np.linalg.norm(diff, 'fro')
    rel_err = norm_diff / (norm_target + 1e-10)
    
    return mse, mae, rel_err

def solve_procrustes(U_src, U_tgt):
    """Finds optimal rotation R such that || U_src @ R - U_tgt || is minimized."""
    # scipy returns R such that A @ R = B
    R, _ = scipy.linalg.orthogonal_procrustes(U_src, U_tgt)
    return R

def analyze_triplet(name, W0, W1, W2, results_list):
    """
    Performs the full A/B/C analysis for the triplet W0->W1->W2.
    """
    if W0 is None or W1 is None or W2 is None:
        return

    # 1. SVD
    U0, S0, V0h = get_svd(W0)
    U1, S1, V1h = get_svd(W1)
    U2, S2, V2h = get_svd(W2)
    
    # Transpose Vh to get V (columns)
    V0, V1, V2 = V0h.T, V1h.T, V2h.T

    # Define pairs to analyze: (Source, Target, Label, Source_Label, Target_Label)
    pairs = [
        (W0, W1, U0, S0, V0, U1, S1, V1, "Base->Stage1"),
        (W1, W2, U1, S1, V1, U2, S2, V2, "Stage1->Stage2"),
        (W0, W2, U0, S0, V0, U2, S2, V2, "Base->Stage2")
    ]

    for (W_src, W_tgt, U_src, S_src, V_src, U_tgt, S_tgt, V_tgt, pair_label) in pairs:
        row = {
            "Module": name,
            "Transition": pair_label
        }

        # --- A. Spectrum Stability ---
        # How much did Sigma change? || S_tgt - S_src || / || S_src ||
        s_diff_norm = np.linalg.norm(S_tgt - S_src)
        s_src_norm = np.linalg.norm(S_src)
        row["Spectrum_RelErr"] = s_diff_norm / (s_src_norm + 1e-10)

        # Baseline Drift (Real change in weights)
        row["Baseline_MSE"], row["Baseline_MAE"], row["Baseline_RelErr"] = calc_metrics(W_tgt, W_src)

        # --- B. Inner Stiefel (Fiber Rotation) ---
        # Hypothesis: W_tgt ≈ (U_src @ R_u) @ S_src @ (V_src @ R_v).T
        # We enforce strict subspace conservation (mixing only).
        R_u = solve_procrustes(U_src, U_tgt)
        R_v = solve_procrustes(V_src, V_tgt)
        
        W_inner = (U_src @ R_u) @ np.diag(S_src) @ (V_src @ R_v).T
        
        row["Inner_MSE"], row["Inner_MAE"], row["Inner_RelErr"] = calc_metrics(W_tgt, W_inner)

        # --- C. Ambient Stiefel (Subspace Transport) ---
        # Hypothesis: W_tgt ≈ U_tgt @ S_src @ V_tgt.T
        # We allow subspace drift (U_src -> U_tgt) but force energy conservation (S_src).
        W_ambient = U_tgt @ np.diag(S_src) @ V_tgt.T
        
        row["Ambient_MSE"], row["Ambient_MAE"], row["Ambient_RelErr"] = calc_metrics(W_tgt, W_ambient)
        
        # --- Comparison Metric ---
        # Ratio > 1 means Ambient is better (Drift is real). Ratio ~ 1 means Rotation is sufficient.
        row["Drift_Ratio"] = row["Inner_MSE"] / (row["Ambient_MSE"] + 1e-30)

        results_list.append(row)
        
        # Live print for sanity
        print(f"{name}  {pair_label:15s} | Spec: {row['Spectrum_RelErr']:.1e} | Baseline_RelErr: {row['Baseline_RelErr']:.1e} | InnerRel: {row['Inner_RelErr']:.1e} | AmbRel: {row['Ambient_RelErr']:.1e}")
        print(f"    Baseline MSE: {row['Baseline_MSE']:.1e} | Inner MSE: {row['Inner_MSE']:.1e} | Ambient MSE: {row['Ambient_MSE']:.1e} | Drift Ratio: {row['Drift_Ratio']:.2f}")

# --- Main ---

def main():
    print("Loading Models (Float64 for Precision)...")
    try:
        m0 = AutoModelForVision2Seq.from_pretrained(model_name_0, device_map="cpu", torch_dtype=torch.float64, trust_remote_code=True)
        m1 = AutoModelForVision2Seq.from_pretrained(model_name_1, device_map="cpu", torch_dtype=torch.float64, trust_remote_code=True)
        m2 = AutoModelForVision2Seq.from_pretrained(model_name_2, device_map="cpu", torch_dtype=torch.float64, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    results = []

    print("\n=== Scanning Language Layers ===")
    for idx in LANG_LAYERS_TO_SCAN:
        for sub in LANG_MODULES:
            name = f"Lang_L{idx}.{sub}"
            print(f"Processing {name}...")
            
            W0 = get_module_weight(m0, idx, sub, is_vision=False)
            W1 = get_module_weight(m1, idx, sub, is_vision=False)
            W2 = get_module_weight(m2, idx, sub, is_vision=False)
            
            analyze_triplet(name, W0, W1, W2, results)

    print("\n=== Scanning Vision Layers ===")
    for idx in VISION_LAYERS_TO_SCAN:
        for sub in VISION_MODULES:
            name = f"Vis_L{idx}.{sub}"
            print(f"Processing {name}...")
            
            W0 = get_module_weight(m0, idx, sub, is_vision=True)
            W1 = get_module_weight(m1, idx, sub, is_vision=True)
            W2 = get_module_weight(m2, idx, sub, is_vision=True)
            
            analyze_triplet(name, W0, W1, W2, results)

    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved metrics to {OUTPUT_CSV}")
    
    # Print Summary for meeting
    print("\n=== MEETING SUMMARY (Averages) ===")
    print(df.groupby("Transition")[["Spectrum_RelErr", "Inner_RelErr", "Ambient_RelErr"]].mean())

if __name__ == "__main__":
    main()