import torch

# # random init n * n matrix 
# n = 100
# m = 200

n = 200
m = 100

matrix_a = torch.randn(n, m)
matrix_b = matrix_a + 1e-3 * torch.randn(n, m)

# do svd for a
U_a, S_a, Vh_a = torch.linalg.svd(matrix_a, full_matrices=False)
# do svd for b
U_b, S_b, Vh_b = torch.linalg.svd(matrix_b, full_matrices=False)

V_a = Vh_a.T
V_b = Vh_b.T

Q_1 = U_a.T @ U_b
Q_2 = V_a.T @ V_b

diff = Q_1 - Q_2

from matplotlib import pyplot as plt

fig, axes = plt.subplots(1, 1, figsize=(8.4, 4.2), dpi=200)

im0 = axes.imshow(diff)
axes.set_title("diff = Q_1 - Q_2")

ticks = list(range(0, min(n, m), 10))
labels = [str(t) for t in ticks]
axes.set_xticks(ticks); axes.set_xticklabels(labels)
axes.set_yticks(ticks); axes.set_yticklabels(labels)

fig.colorbar(im0, ax=axes)
fig.savefig("diff.png")
plt.close()

exit()


# print(f"u shape: {U_a.shape}, {U_b.shape}")
# print(f"v shape: {V_a.shape}, {V_b.shape}")
# print(f"q shape: {Q_1.shape}, {Q_2.shape}")

I_orth = Q_1.T @ Q_2

# printdiagnal elements
diag_elements = torch.diagonal(I_orth)
print("Diagonal elements of I_orth:")
for i, val in enumerate(diag_elements):
    print(f"  {i:4d}: {val:.6f}")
    
print(I_orth[:5, :5])

from matplotlib import pyplot as plt

fig, axes = plt.subplots(1, 1, figsize=(8.4, 4.2), dpi=200)

im0 = axes.imshow(I_orth)
axes.set_title("I_orth")

ticks = list(range(0, min(n, m), 10))
labels = [str(t) for t in ticks]
axes.set_xticks(ticks); axes.set_xticklabels(labels)
axes.set_yticks(ticks); axes.set_yticklabels(labels)

fig.colorbar(im0, ax=axes)
fig.savefig("I_orth_matrix.png")
plt.close()
