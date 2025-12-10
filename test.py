
import torch

file_path = "debug_U_rl1_U_rl1T.pt"

M_tensor = torch.load(file_path, map_location='cpu')

# 接下来，给我写一个脚本，看一下 M tensor 的对角线元素，哪些是大于 0.5，哪些是小于 0.5
diag_elements = torch.diagonal(M_tensor)

# 给我把 diag elements 的取值分布的图画出来

import matplotlib.pyplot as plt

plt.hist(diag_elements.numpy(), bins=50)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Diagonal Elements of M_tensor')
plt.grid(True)

plt.savefig("diag_elements_distribution.png")

# plt.show()

# print("=" * 50)
# print(diag_elements)

# # 统计大于 0.5 和小于等于 0.5 的元素数量
# num_greater_than_0_5 = torch.sum(diag_elements > 0.5).item()
# num_less_equal_0_5 = torch.sum(diag_elements <= 0.5).item()

# print("=" * 50)
# print(f"Number of elements greater than 0.5: {num_greater_than_0_5}")
# print(f"Number of elements less than or equal to 0.5: {num_less_equal_0_5}")

# # 还要统计有哪些坐标
# indices_greater_than_0_5 = torch.nonzero(diag_elements > 0.5, as_tuple=True)[0].tolist()
# indices_less_equal_0_5 = torch.nonzero(diag_elements <= 0.5, as_tuple=True)[0].tolist()

# breakpoint()

# print("=" * 50)
# print(f"Indices of elements greater than 0.5: {indices_greater_than_0_5}")
# # print(f"Indices of elements less than or equal to 0.5: {indices_less_equal_0_5}")



