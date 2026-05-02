import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForVision2Seq

# --- Configuration ---
model_name_0 = "Qwen/Qwen2.5-VL-3B-Instruct"    # Base
model_name_1 = "IffYuan/Embodied-R1-3B-Stage1" # Stage 1
model_name_2 = "IffYuan/Embodied-R1-3B-v1"     # Stage 2 (Final)

layer_idx = 10

# Check all these modules
target_modules = [
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj"
]

# Max dimension to attempt Left-Side rotation (avoids OOM on large vocabs)
MAX_DIM_FOR_LEFT_ROTATION = 12000 

# --- Helper Functions ---

def get_layer_weight(model, layer_idx, submodule_name):
    """Robustly extracts the weight matrix (W) from various architectures."""
    try:
        if hasattr(model, "language_model"):
            lm = model.language_model
            layers = lm.model.layers if hasattr(lm, "model") else lm.layers
        elif hasattr(model, "model"):
            layers = model.model.layers
        else:
            layers = model.layers
        
        layer = layers[layer_idx]
        module = layer
        for token in submodule_name.split('.'):
            module = getattr(module, token)
        return module.weight.detach().float().cpu().numpy()
    except Exception as e:
        print(f"Error accessing {submodule_name}: {e}")
        return None

def get_singular_bases(W):
    """Decomposes W into U, S, V. Returns U, S, V (columns)."""
    U, S, Vh = scipy.linalg.svd(W, full_matrices=False)
    return U, S, Vh.T

def solve_rotation_right(Source, Target):
    """
    Solves for R in: Source @ R = Target
    (Internal Mixing / Superposition)
    Returns R (k x k)
    """
    R, _ = scipy.linalg.orthogonal_procrustes(Source, Target)
    return R

def solve_rotation_left(Source, Target):
    """
    Solves for R in: R @ Source = Target
    (Ambient Rotation / Coordinate Shift)
    
    Logic: || R S - T ||_F  == || (R S - T).T ||_F == || S.T R.T - T.T ||_F
    We solve Procrustes(S.T, T.T) to get R.T, then transpose back.
    
    Returns R (N x N) - BE CAREFUL with size!
    """
    # Procrustes solves A @ X = B. We want S.T @ R.T = T.T
    R_T, _ = scipy.linalg.orthogonal_procrustes(Source.T, Target.T)
    return R_T.T

def calculate_mse(A, B):
    return np.mean((A - B)**2)

# --- Main Execution ---

def main():
    print("Loading Models...")
    try:
        # Load float16 on CPU to save memory
        m0 = AutoModelForVision2Seq.from_pretrained(model_name_0, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True)
        m1 = AutoModelForVision2Seq.from_pretrained(model_name_1, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True)
        m2 = AutoModelForVision2Seq.from_pretrained(model_name_2, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print(f"\nAnalyzing Layer {layer_idx}...")

    for mod_name in target_modules:
        print(f"\n--- Module: {mod_name} ---")
        
        # 1. Extract
        W0 = get_layer_weight(m0, layer_idx, mod_name)
        W1 = get_layer_weight(m1, layer_idx, mod_name) # Intermediate (Stage 1)
        W2 = get_layer_weight(m2, layer_idx, mod_name) # Final (Stage 2)
        
        if W0 is None: continue

        # 2. Decompose
        # We only need Base (Start) and Stage 2 (End) for this direct check
        # But to match your logic of "Composed", we can check Base->Stage2 directly first
        U0, S0, V0 = get_singular_bases(W0)
        U2, S2, V2 = get_singular_bases(W2)
        
        # 3. Check Right-Side (Internal) Rotation (U @ R)
        # This is the "Stiefel / Superposition" hypothesis
        L_right = solve_rotation_right(U0, U2)
        R_right = solve_rotation_right(V0, V2)
        
        U_pred_right = U0 @ L_right
        V_pred_right = V0 @ R_right
        W_pred_right = U_pred_right @ np.diag(S0) @ V_pred_right.T
        
        mse_right = calculate_mse(W2, W_pred_right)
        print(f"1. Internal Rotation (Right-Side) MSE : {mse_right:.6e}")

        # 4. Check Left-Side (Ambient) Rotation (R @ U)
        # This is the "Global Coordinate Shift" hypothesis
        # WARNING: Matrices can be huge (N x N)
        dim_out, dim_in = W0.shape
        
        # Output Space Left-Rotation
        if dim_out <= MAX_DIM_FOR_LEFT_ROTATION:
            L_left = solve_rotation_left(U0, U2)
            U_pred_left = L_left @ U0
            mse_u_left = calculate_mse(U2, U_pred_left)
            print(f"   [Left-Side U Check] MSE            : {mse_u_left:.6e}")
        else:
            print(f"   [Left-Side U Check] SKIPPED (Dim {dim_out} > {MAX_DIM_FOR_LEFT_ROTATION})")
            U_pred_left = U_pred_right # Fallback to avoid crash in W reconstruction check

        # Input Space Left-Rotation
        if dim_in <= MAX_DIM_FOR_LEFT_ROTATION:
            R_left = solve_rotation_left(V0, V2)
            V_pred_left = R_left @ V0
            mse_v_left = calculate_mse(V2, V_pred_left)
            print(f"   [Left-Side V Check] MSE            : {mse_v_left:.6e}")
        else:
            print(f"   [Left-Side V Check] SKIPPED (Dim {dim_in} > {MAX_DIM_FOR_LEFT_ROTATION})")
            V_pred_left = V_pred_right # Fallback

        # 5. Comparison
        # If Right-Side MSE is low, Left-Side MSE should theoretically also be low 
        # (because Internal is a subset of Ambient). 
        # BUT if Left-Side is high, it means the subspace moved in a way Procrustes couldn't capture globally? 
        # Actually, if Internal works, Left works.
        # The real test is: Does Rigid Body Rotation of W work? (R * W_base)
        
        if dim_out <= MAX_DIM_FOR_LEFT_ROTATION:
             # Try to rotate W directly (Rigid Body) assuming V is fixed
             # W_new = R_left @ W_old
             W_rigid = L_left @ W0
             mse_rigid = calculate_mse(W2, W_rigid)
             print(f"2. Rigid Body Rotation (Left-Only) MSE: {mse_rigid:.6e} (Should be high)")

        # 6. Conclusion for this module
        if mse_right < 1e-9:
             print(">> VERDICT: Internal (Right-Side) Rotation is confirmed.")
        else:
             print(">> VERDICT: Rotation hypothesis failing.")

if __name__ == "__main__":
    main()