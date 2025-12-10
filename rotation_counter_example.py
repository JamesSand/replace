# 这是一个 counter example，如果不是旋转的话，会出现什么情况

import torch
import torch.nn.functional as F

def run_experiment_fixed():
    torch.manual_seed(42)
    
    # ==========================================
    # 设定维度
    # ==========================================
    D_OUT = 100  
    D_IN = 20    
    
    # 创建正交基
    full_basis = torch.linalg.qr(torch.randn(D_OUT, D_OUT))[0]
    subspace_A = full_basis[:, :D_IN]       # 子空间 A
    subspace_B = full_basis[:, D_IN:2*D_IN] # 子空间 B (与 A 垂直)

    # ==========================================
    # Base Model: 位于子空间 A
    # ==========================================
    U_base = subspace_A 
    # 为了简化，假设 Base 就是 A 的基

    # ==========================================
    # Case 1: Rotation Hypothesis (完美旋转)
    # 所有的模型都在子空间 A 里，只是方向不同
    # ==========================================
    # RL1 (Rotation): 在 A 里随机转
    Q_rl1_rot = torch.linalg.qr(torch.randn(D_IN, D_IN))[0]
    U_rl1_rot = subspace_A @ Q_rl1_rot
    
    # RL2 (Rotation): 在 A 里随机转
    Q_rl2_rot = torch.linalg.qr(torch.randn(D_IN, D_IN))[0]
    U_rl2_rot = subspace_A @ Q_rl2_rot

    # ==========================================
    # Case 2: Subspace Mismatch (特征漂移/断桥)
    # 假设 Base 和 RL2 其实是有重叠的 (都在 A)，
    # 但是中间的 RL1 跑偏了 (跑到了 B)，导致“桥”断了。
    # ==========================================
    
    # RL1 (Shift): 跑到了子空间 B !!!
    Q_rl1_shift = torch.linalg.qr(torch.randn(D_IN, D_IN))[0]
    U_rl1_shift = subspace_B @ Q_rl1_shift 
    
    # RL2 (Return): 依然在子空间 A (或者说与 Base 有强相关性)
    # 这样 R3 (Base->RL2) 是存在的，但 R4 (Base->RL1->RL2) 会被断掉
    U_rl2_same = U_rl2_rot # 直接复用上面的，保证它在 A 里

    # ==========================================
    # 计算函数
    # ==========================================
    def calc_metrics(name, u_base, u_rl1, u_rl2):
        # R3: Direct Path
        R3 = u_base.T @ u_rl2
        
        # R4: Composed Path via RL1
        R1 = u_base.T @ u_rl1
        R2 = u_rl1.T @ u_rl2
        R4 = R1 @ R2
        
        mse = F.mse_loss(R3, R4)
        
        # 打印 R3 的范数，证明它不是 0 (说明 Base 和 RL2 有联系)
        norm_r3 = torch.norm(R3)
        # 打印 R4 的范数，看看通过 RL1 后还剩多少信息
        norm_r4 = torch.norm(R4)
        
        print(f"[{name}]")
        print(f"  R3 Norm (Direct):   {norm_r3:.4f}")
        print(f"  R4 Norm (Composed): {norm_r4:.4f}")
        print(f"  MSE Loss:           {mse:.10f}")
        print("-" * 30)

    print("--- 修正后的实验结果 ---\n")
    
    # 运行 Case 1
    calc_metrics("Case 1: Rotation (全在子空间A)", U_base, U_rl1_rot, U_rl2_rot)
    
    # 运行 Case 2
    calc_metrics("Case 2: Shift (RL1 跑偏到了 B)", U_base, U_rl1_shift, U_rl2_rot)

if __name__ == "__main__":
    run_experiment_fixed()