import transformers
from transformers import AutoModelForVision2Seq
import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
model_name_0 = "Qwen/Qwen2.5-VL-3B-Instruct"
model_name_1 = "IffYuan/Embodied-R1-3B-Stage1"
model_name_2 = "IffYuan/Embodied-R1-3B-v1"
layer_idx = 10
submodule_name = "self_attn.q_proj" # Standard for Qwen/Llama architectures

# --- Helper Functions ---
def get_layer_weight(model, layer_idx, submodule_name):
    """Extracts the weight matrix (W) from a specific layer and submodule."""
    try:
        # 1. Navigate to the ModuleList (the list of decoder layers)
        if hasattr(model, "language_model"):
            # Qwen2-VL specific path: model -> language_model -> model -> layers
            # Note: Sometimes it's model.language_model.model.layers, sometimes just model.language_model.layers
            # The structure print shows: (language_model): Qwen2_5_VLTextModel -> (layers): ModuleList
            layers = model.language_model.layers
        elif hasattr(model, "model"):
            # Standard HF path
            layers = model.model.layers
        else:
            # Fallback for some architectures
            layers = model.layers 
            
        # 2. Select the specific layer index
        layer = layers[layer_idx]
        
        # 3. Navigate to submodule (e.g., 'self_attn.q_proj')
        tokens = submodule_name.split('.')
        module = layer
        for token in tokens:
            module = getattr(module, token)
        
        # 4. Return weight
        return module.weight.detach().float().cpu().numpy()
        
    except AttributeError as e:
        print(f"Error accessing layer {layer_idx} {submodule_name}: {e}")
        # Debug helper: print available keys if it fails
        if hasattr(model, "language_model"):
            print("Debug: model.language_model keys:", model.language_model._modules.keys())
        return None

def get_singular_bases(W):
    """Performs SVD and returns U, S, Vh."""
    # W is (out, in). SVD: W = U @ S @ Vh
    U, S, Vh = scipy.linalg.svd(W, full_matrices=False)
    # Return V (columns) instead of Vh (rows) for consistent rotation logic
    return U, S, Vh.T 

def solve_rotation(Source_Basis, Target_Basis):
    """
    Finds rotation matrix R such that Source @ R approx Target.
    Uses Orthogonal Procrustes problem.
    """
    # orthogonal_procrustes minimizes ||A @ R - B||_F
    R, _ = scipy.linalg.orthogonal_procrustes(Source_Basis, Target_Basis)
    return R

# def compare_rotations(Rot_Direct, Rot_Composed):
#     """Computes the Frobenius error between the direct and composed rotations."""
#     diff = Rot_Direct - Rot_Composed
#     error = np.linalg.norm(diff, 'fro')
#     return error

def compare_rotations_detailed(Rot_Direct, Rot_Composed):
    """Computes both Frobenius Norm and MSE to compare with student results."""
    diff = Rot_Direct - Rot_Composed
    
    # 1. Frobenius Norm (Total Energy of Error) -> Expect ~1e-5
    frob_error = np.linalg.norm(diff, 'fro')
    
    # 2. Mean Squared Error (Average Error per Element) -> Expect ~1e-12
    # Note: MSE usually implies squaring the difference first
    mse_error = (diff ** 2).mean()
    
    return frob_error, mse_error

# --- Main Execution ---

print("Loading models... (This may take a moment)")
# Load with low_cpu_mem_usage to avoid RAM OOM if loading 3 models
model0 = AutoModelForVision2Seq.from_pretrained(
    model_name_0, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
print("Loaded Model 0 (Base)")

model1 = AutoModelForVision2Seq.from_pretrained(
    model_name_1, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
print("Loaded Model 1 (Stage 1)")

model2 = AutoModelForVision2Seq.from_pretrained(
    model_name_2, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
print("Loaded Model 2 (Stage 2)")

print(f"\nAnalyzing Layer {layer_idx} | Module: {submodule_name}")

print(model2)

# 1. Extract Weights
W0 = get_layer_weight(model0, layer_idx, submodule_name)
W1 = get_layer_weight(model1, layer_idx, submodule_name)
W2 = get_layer_weight(model2, layer_idx, submodule_name)

if W0 is not None and W1 is not None and W2 is not None:
    
    # 2. Decompose (SVD)
    print("Computing SVD...")
    U0, S0, V0 = get_singular_bases(W0)
    U1, S1, V1 = get_singular_bases(W1)
    U2, S2, V2 = get_singular_bases(W2)
    
    # Sanity Check: Spectrum Stability
    s_diff = np.linalg.norm(S2 - S0) / np.linalg.norm(S0)
    print(f"Spectrum Stability Check (Frobenius Rel. Error): {s_diff:.6f}")
    if s_diff > 1e-3:
        print("WARNING: Spectrum has changed significantly. Rotation hypothesis may be weak.")

    # 3. Solve Rotations for U (Output Space)
    # L_01: Base -> Stage 1
    L_01 = solve_rotation(U0, U1)
    # L_12: Stage 1 -> Stage 2
    L_12 = solve_rotation(U1, U2)
    # L_02: Base -> Stage 2 (Direct)
    L_02_Direct = solve_rotation(U0, U2)
    
    # 4. Solve Rotations for V (Input Space)
    # R_01: Base -> Stage 1
    R_01 = solve_rotation(V0, V1)
    # R_12: Stage 1 -> Stage 2
    R_12 = solve_rotation(V1, V2)
    # R_02: Base -> Stage 2 (Direct)
    R_02_Direct = solve_rotation(V0, V2)

    # 5. Validate Transitivity (The "Math Proof")
    # If valid, L_02 should be approx L_01 @ L_12
    # Note: Since we solved Source @ R = Target, the composition is U0 @ L01 @ L12
    
    L_02_Composed = L_01 @ L_12
    R_02_Composed = R_01 @ R_12
    
    err_L_frob, err_L_mse = compare_rotations_detailed(L_02_Direct, L_02_Composed)
    err_R_frob, err_R_mse = compare_rotations_detailed(R_02_Direct, R_02_Composed)
    
    print("\n--- Results ---")
    print(f"L (Output Rotation) Composition Frobenius Error: {err_L_frob:.6e}")
    print(f"L (Output Rotation) Composition MSE Error: {err_L_mse:.6e}")
    print(f"R (Input Rotation)  Composition Frobenius Error: {err_R_frob:.6e}")
    print(f"R (Input Rotation)  Composition MSE Error: {err_R_mse:.6e}")
    
    # 6. Visualization (Optional Heatmap of L_12)
    # Ideally, if updates are sparse, this might look close to Identity or block diagonal
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    # Show top 50x50 dims to see if rotation is local
    sns.heatmap(L_12[:50, :50], cmap="RdBu", center=0)
    plt.title("L Rotation (S1 -> S2) [Top 50x50]")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(R_12[:50, :50], cmap="RdBu", center=0)
    plt.title("R Rotation (S1 -> S2) [Top 50x50]")
    plt.tight_layout()
    # plt.show()
    plt.savefig("rotation_matrices_heatmap.png")

else:
    print("Failed to extract weights from one or more models.")