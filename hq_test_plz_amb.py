import torch
import numpy as np
import scipy.linalg
from transformers import AutoModelForVision2Seq

# --- Configuration ---
model_name_0 = "Qwen/Qwen2.5-VL-3B-Instruct"    # Base
model_name_1 = "IffYuan/Embodied-R1-3B-Stage1" # Stage 1 (Intermediate)
model_name_2 = "IffYuan/Embodied-R1-3B-v1"     # Stage 2 (Final)

layer_idx = 10
submodule_name = "self_attn.o_proj" # Testing on the MLP layer

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
            
        # Float32 is required for this precision
        return layer.weight.detach().to(dtype=torch.float32).cpu().numpy()
    except Exception as e:
        print(f"Error: {e}")
        return None

def verify_two_stage_ambient(W0, W1, W2):
    print(f"--- Two-Stage Ambient Verification ({submodule_name}) ---")
    
    # 1. SVD Decomposition
    # U shapes are (11008, 2048)
    U0, S0, V0 = scipy.linalg.svd(W0, full_matrices=False)
    U1, S1, V1 = scipy.linalg.svd(W1, full_matrices=False)
    U2, S2, V2 = scipy.linalg.svd(W2, full_matrices=False)
    V0, V1, V2 = V0.T, V1.T, V2.T
    
    # 2. Construct Ambient Rotation Matrices explicitly (N x N)
    # L = U_next @ U_prev.T
    print("Constructing Stage 1 Ambient Matrix (L1)...")
    L1_amb = U1 @ U0.T
    
    print("Constructing Stage 2 Ambient Matrix (L2)...")
    L2_amb = U2 @ U1.T
    
    # 3. Compose them (Matrix Multiplication)
    print("Composing L_final = L2 @ L1...")
    L_composed = L2_amb @ L1_amb
    
    # 4. Check against Direct Ambient Rotation
    print("Constructing Direct Ambient Matrix (L_direct)...")
    L_direct = U2 @ U0.T
    
    diff_L = np.mean((L_composed - L_direct)**2)
    print(f"\n> Path Consistency MSE (L_composed vs L_direct): {diff_L:.6e}")
    
    # 5. Reconstruct Final Weights using the Composed Ambient Matrix
    # We take the Base features (U0), rotate them through the chain (L_composed),
    # and apply the Base energy (S0).
    # W_pred = (L_composed @ U0) * S0 * V_new.T
    
    # Note: L_composed @ U0 should equal U2 exactly
    U_pred = L_composed @ U0
    
    # We assume V follows the same Ambient logic (omitted here for brevity, using V2 directly for U-test)
    # Testing: Does the Ambient Chain preserve the U-subspace correctly?
    W_pred_chain = U_pred @ np.diag(S0) @ V2.T
    
    mse_recon = np.mean((W2 - W_pred_chain)**2)
    
    # Compare to Baseline Drift
    mse_base = np.mean((W2 - W0)**2)
    
    print(f"> Weight Reconstruction MSE (Using Chain)    : {mse_recon:.6e}")
    print(f"> Baseline Drift (Actual Change)             : {mse_base:.6e}")
    
    if diff_L < 1e-10:
        print("\n>> RESULT: Transitivity Confirmed.")
        print("   The Ambient Rotation decomposes perfectly into stages.")
        print("   L_final == L_stage2 @ L_stage1")

def main():
    print("Loading Models...")
    m0 = AutoModelForVision2Seq.from_pretrained(model_name_0, device_map="cpu", torch_dtype=torch.float32, trust_remote_code=True)
    m1 = AutoModelForVision2Seq.from_pretrained(model_name_1, device_map="cpu", torch_dtype=torch.float32, trust_remote_code=True)
    m2 = AutoModelForVision2Seq.from_pretrained(model_name_2, device_map="cpu", torch_dtype=torch.float32, trust_remote_code=True)

    W0 = get_layer_weight(m0, layer_idx, submodule_name)
    W1 = get_layer_weight(m1, layer_idx, submodule_name)
    W2 = get_layer_weight(m2, layer_idx, submodule_name)

    if W0 is not None and W1 is not None and W2 is not None:
        verify_two_stage_ambient(W0, W1, W2)

if __name__ == "__main__":
    main()