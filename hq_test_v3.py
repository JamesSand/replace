import torch
import numpy as np
import scipy.linalg
from transformers import AutoModelForVision2Seq

# --- Configuration ---
model_name_0 = "Qwen/Qwen2.5-VL-3B-Instruct"    # Base
model_name_2 = "IffYuan/Embodied-R1-3B-v1"     # Stage 2 (Final)
layer_idx = 10
submodule_name = "mlp.up_proj"

def get_layer_weight(model, layer_idx, submodule_name):
    try:
        if hasattr(model, "language_model"):
            lm = model.language_model
            layers = lm.model.layers if hasattr(lm, "model") else lm.layers
        else:
            layers = model.model.layers
        
        layer = layers[layer_idx]
        for token in submodule_name.split('.'):
            layer = getattr(layer, token)
            
        # CRITICAL FIX: Convert to float32 (or float64) immediately
        # We want the analysis to be limited only by model weights, not our variables.
        return layer.weight.detach().to(dtype=torch.float32).cpu().numpy()
    except Exception as e:
        print(f"Error: {e}")
        return None

def decompose_error_sources(W_base, W_final):
    """
    Decomposes the weight update into 3 components using Float64 precision:
    1. Sigma Change (Scaling)
    2. Rotation Error (Alignment)
    3. Subspace Drift (Grassmannian)
    """
    # Force Float64 for SVD to avoid any numerical instability
    W_base = W_base.astype(np.float64)
    W_final = W_final.astype(np.float64)

    # 1. SVD
    # Full matrices=False is fine, we just need the active subspace
    U0, S0, V0 = scipy.linalg.svd(W_base, full_matrices=False)
    U2, S2, V2 = scipy.linalg.svd(W_final, full_matrices=False)
    V0, V2 = V0.T, V2.T # Convert to column vectors
    
    # 2. Compute Procrustes Rotations (Best fit inside the subspace)
    # This finds the optimal 'L' such that U0 @ L ≈ U2
    L = scipy.linalg.orthogonal_procrustes(U0, U2)[0]
    R = scipy.linalg.orthogonal_procrustes(V0, V2)[0]
    
    # 3. Construct the predictions
    
    # Prediction A: "Pure Rotation Hypothesis" 
    # (Old Subspace rotated + Old Energy)
    # W = (U0 @ L) @ S0 @ (V0 @ R).T
    W_rot_pred = (U0 @ L) @ np.diag(S0) @ (V0 @ R).T
    
    # Prediction B: "Subspace Drift Hypothesis" 
    # (New Subspace + Old Energy)
    # W = U2 @ S0 @ V2.T
    # This isolates "Did the singular values change?" from "Did the vectors change?"
    W_subspace_pred = U2 @ np.diag(S0) @ V2.T
    
    # 4. Calculate Errors (MSE)
    # Total actual change in weights
    actual_change_mse = np.mean((W_final - W_base)**2)
    
    # Total error of our Rotation Model
    rot_recon_mse = np.mean((W_final - W_rot_pred)**2)
    
    # Component 1: Energy Error (How much did S change?)
    # Difference between "New Vectors+New S" and "New Vectors+Old S"
    energy_error_mse = np.mean((W_final - W_subspace_pred)**2)
    
    # Component 2: Subspace Drift (How much did U/V leave the manifold?)
    # Difference between "New Vectors" and "Rotated Old Vectors"
    drift_error_mse = np.mean((W_subspace_pred - W_rot_pred)**2)
    
    print(f"\n=== High-Precision Decomposition ({submodule_name}) ===")
    print(f"Baseline Drift (Actual Change): {actual_change_mse:.6e}")
    print(f"Rotation Model Error          : {rot_recon_mse:.6e}")
    print("-" * 40)
    print(f"1. Energy Error (Sigma Change): {energy_error_mse:.6e}")
    print(f"2. Drift Error (Subspace)     : {drift_error_mse:.6e}")
    
    # Ratios
    drift_ratio = drift_error_mse / (energy_error_mse + 1e-30)
    print("-" * 40)
    if drift_ratio > 10:
        print(f">> RESULT: Error is dominated by SUBSPACE DRIFT ({drift_ratio:.1f}x larger than Energy Error).")
        print("   The model kept singular values fixed, but moved the feature subspace.")
    elif drift_ratio < 0.1:
        print(f">> RESULT: Error is dominated by ENERGY CHANGE.")
        print("   The model stayed in the subspace, but scaled the singular values.")
    else:
        print(">> RESULT: Mixed behavior (Drift and Energy change are comparable).")

def main():
    print("Loading Models in Float32...")
    try:
        # Load in Float32 to ensure we don't start with quantization noise
        m0 = AutoModelForVision2Seq.from_pretrained(model_name_0, device_map="cpu", torch_dtype=torch.float32, trust_remote_code=True)
        m2 = AutoModelForVision2Seq.from_pretrained(model_name_2, device_map="cpu", torch_dtype=torch.float32, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    W0 = get_layer_weight(m0, layer_idx, submodule_name)
    W2 = get_layer_weight(m2, layer_idx, submodule_name)

    if W0 is not None and W2 is not None:
        decompose_error_sources(W0, W2)

if __name__ == "__main__":
    main()