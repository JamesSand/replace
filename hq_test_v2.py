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
# submodule_name = "self_attn.o_proj"
submodule_name = "mlp.gate_proj"
# submodule_name = "mlp.up_proj"
# submodule_name = "mlp.down_proj"

# --- Helper Functions ---

def align_svd_signs(U_ref, U, V_ref=None, V=None):
    """
    Align the column signs of U (and optionally V) to match a reference basis.
    Uses sign of column-wise dot products: sign(<u_ref_i, u_i>).

    U_ref, U: [m, r]
    V_ref, V: [n, r]  (optional)
    Returns aligned U (and V if provided).
    """
    # column-wise inner product (r,)
    dots = np.sum(U_ref * U, axis=0)
    signs = np.sign(dots)
    signs[signs == 0] = 1.0  # avoid zero sign

    U_aligned = U * signs  # broadcast over rows

    if V_ref is not None and V is not None:
        V_aligned = V * signs
        return U_aligned, V_aligned, signs

    return U_aligned, signs

def get_layer_weight(model, layer_idx, submodule_name):
    """Robustly extracts the weight matrix (W) with the correct Qwen-VL path."""
    try:
        # 1. Check for Qwen-VL specific structure (User Correction)
        if hasattr(model, "language_model"):
            if hasattr(model.language_model, "layers"):
                layers = model.language_model.layers
            elif hasattr(model.language_model, "model"):
                 # Fallback for some specific HF versions
                layers = model.language_model.model.layers
            else:
                raise AttributeError("Could not find 'layers' inside language_model")
                
        # 2. Check for Standard Llama/Qwen structure
        elif hasattr(model, "model"):
            layers = model.model.layers
            
        # 3. Fallback
        else:
            layers = model.layers
        
        layer = layers[layer_idx]
        tokens = submodule_name.split('.')
        module = layer
        for token in tokens:
            module = getattr(module, token)
            
        return module.weight.detach().float().cpu().numpy()
        
    except Exception as e:
        print(f"Error accessing layer {layer_idx}: {e}")
        # Debug print to help identify structure if it fails again
        if hasattr(model, "language_model"):
             print(f"Debug: Keys in language_model: {model.language_model.__dict__.keys()}")
        return None

def get_singular_bases(W):
    W = W.astype(np.float64)
    U, S, Vh = scipy.linalg.svd(W, full_matrices=False)
    return U, S, Vh.T

def solve_rotation(Source_Basis, Target_Basis):
    """
    Finds orthogonal matrix R using Procrustes (Enforces Orthogonality).
    This forces the 'Strong Assumption' that the transformation is a rotation.
    """
    # R, _ = scipy.linalg.orthogonal_procrustes(Source_Basis, Target_Basis)
    # return R
    return Source_Basis.T @ Target_Basis

def compute_rotation(U_src: torch.Tensor, U_tgt: torch.Tensor) -> torch.Tensor:
    """
    给定两个正交基 U_src, U_tgt，计算旋转矩阵 R，使得:
        U_tgt ≈ U_src @ R
    在理想情况下，R = U_src^T @ U_tgt 是一个正交矩阵。
    """
    if U_src.shape != U_tgt.shape:
        raise ValueError(
            f"Shape mismatch between U_src {tuple(U_src.shape)} and U_tgt {tuple(U_tgt.shape)}"
        )
    # [m, r]^T @ [m, r] -> [r, r]
    return U_src.T @ U_tgt

def analyze_rotation_quality(Source_Basis, Target_Basis, label="Stage Analysis"):
    """
    Compares the 'Solved' Rotation (Procrustes) vs the 'Formula' Projection.
    If MSE is low, the subspace is stable (Pure Rotation).
    If MSE is high, the subspace drifted (Grassmannian Motion).
    """
    # 1. Formula (Projection)
    R_formula = Source_Basis.T @ Target_Basis
    
    # 2. Solved (Procrustes)
    R_solved = solve_rotation(Source_Basis, Target_Basis)
    
    # 3. Compare
    diff = R_formula - R_solved
    mse = np.mean(diff ** 2)
    
    # 4. Check Orthogonality of the Projection
    gram = R_formula.T @ R_formula
    identity = np.eye(gram.shape[0])
    orthogonality_gap = np.linalg.norm(gram - identity, 'fro')
    
    print(f"\n--- {label} (Subspace Stability) ---")
    print(f"Formula vs Solved MSE  : {mse:.6e}")
    print(f"Orthogonality Gap      : {orthogonality_gap:.6e}")
    
    if orthogonality_gap < 1e-2:
        print(">> VERDICT: Pure Rotation (Subspace is Stable)")
    else:
        print(">> VERDICT: Subspace Drift Detected")
        
    return R_solved

def calculate_metrics(W_true, W_pred, label="Reconstruction"):
    diff = W_true - W_pred
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    norm_diff = np.linalg.norm(diff, 'fro')
    norm_true = np.linalg.norm(W_true, 'fro')
    rel_error = norm_diff / norm_true
    
    print(f"\n--- {label} Metrics ---")
    print(f"MSE             : {mse:.6e}")
    print(f"MAE             : {mae:.6e}")
    print(f"Relative Error  : {rel_error:.6%}")

# --- Main Execution ---

def main():
    print(f"Loading Models...")
    try:
        # Load with float16 to save memory
        m0 = AutoModelForVision2Seq.from_pretrained(model_name_0, device_map="cpu", torch_dtype=torch.float64, trust_remote_code=True)
        m1 = AutoModelForVision2Seq.from_pretrained(model_name_1, device_map="cpu", torch_dtype=torch.float64, trust_remote_code=True)
        m2 = AutoModelForVision2Seq.from_pretrained(model_name_2, device_map="cpu", torch_dtype=torch.float64, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    # 1. Extract Weights
    W0 = get_layer_weight(m0, layer_idx, submodule_name)
    W1 = get_layer_weight(m1, layer_idx, submodule_name)
    W2 = get_layer_weight(m2, layer_idx, submodule_name)
    print(f"Extracted weights for layer {layer_idx} and submodule {submodule_name} in the shapes:")
    print(f"W0: {W0.shape if W0 is not None else None}")
    print(f"W1: {W1.shape if W1 is not None else None}")
    print(f"W2: {W2.shape if W2 is not None else None}")

    print("\nCalculating Weight Drift Metrics:")
    calculate_metrics(W0, W1, label="W0 vs W1 Drift")
    calculate_metrics(W1, W2, label="W1 vs W2 Drift")
    calculate_metrics(W0, W2, label="W0 vs W2 Drift")
    
    # print("check two weights in W0, W1, W2:")
    # print(W0[-1, -1], W1[-1, -1], W2[-1, -1])

    if W0 is None or W1 is None or W2 is None:
        print("Error: Could not extract weights.")
        return

    print("Computing SVD...")
    U0, S0, V0 = get_singular_bases(W0)
    U1, S1, V1 = get_singular_bases(W1)
    U2, S2, V2 = get_singular_bases(W2)
    # print(U0.dtype, S0.dtype, V0.dtype)

    # Align Stage1 to Base
    U1, V1, _ = align_svd_signs(U0, U1, V0, V1)

    # Align Stage2 to Stage1 (or to Base; see note below)
    U2, V2, _ = align_svd_signs(U1, U2, V1, V2)

    # 2. Check Spectrum Stability
    S_diff = np.linalg.norm(S2 - S0) / np.linalg.norm(S0)
    print(f"\nSpectrum Stability (Rel. Error): {S_diff:.6f}")
    
    S_diff_stage1 = np.linalg.norm(S1 - S0) / np.linalg.norm(S0)
    print(f"Spectrum Stability Base->Stage1 (Rel. Error): {S_diff_stage1:.6f}")
    
    S_diff_stage2 = np.linalg.norm(S2 - S1) / np.linalg.norm(S1)
    print(f"Spectrum Stability Stage1->Stage2 (Rel. Error): {S_diff_stage2:.6f}")

    # 3. Analyze Subspace Stability (Rotation Validity)
    print("\n=== Output Space (U) Analysis ===")
    L_01 = analyze_rotation_quality(U0, U1, label="Base -> Stage 1")
    L_12 = analyze_rotation_quality(U1, U2, label="Stage 1 -> Stage 2")
    
    print("\n=== Input Space (V) Analysis ===")
    R_01 = analyze_rotation_quality(V0, V1, label="Base -> Stage 1")
    R_12 = analyze_rotation_quality(V1, V2, label="Stage 1 -> Stage 2")

    # 4. Final Reconstruction (Strong Proof)
    # We reconstruct W2 using ONLY Base Spectrum (S0) and Composed Rotations
    L_Composed = L_01 @ L_12
    R_Composed = R_01 @ R_12
    
    U_pred = U0 @ L_Composed
    V_pred = V0 @ R_Composed
    W_pred = U_pred @ np.diag(S0) @ V_pred.T
    
    calculate_metrics(W2, W_pred, label="Rotation Reconstruction (Strong Proof)")
    calculate_metrics(W2, W0, label="Baseline Drift (Actual Change)")
    
    
    ### do an extra check for to avoid the issue of svd ambiguity
    L_02 = scipy.linalg.orthogonal_procrustes(U1, U2)[0]
    R_02 = scipy.linalg.orthogonal_procrustes(V1, V2)[0]
    W_pred_02 = (U1 @ L_02) @ np.diag(S1) @ (V1 @ R_02).T

    calculate_metrics(W2, W_pred_02, label="Rotation Reconstruction (Direct Base->Stage2)")
    
    
    
    

if __name__ == "__main__":
    main()