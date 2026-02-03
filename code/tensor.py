"""
================================================================================
PyTorch 张量基础
================================================================================
功能：演示PyTorch张量的基本创建和属性

PyTorch张量(Tensor)：
    - 类似于NumPy的ndarray，但可以在GPU上加速计算
    - 支持自动微分（autograd），是PyTorch深度学习的基础

张量类型：
    - torch.FloatTensor：32位浮点型
    - torch.DoubleTensor：64位浮点型
    - torch.LongTensor：64位整数型
    - torch.IntTensor：32位整数型
    - torch.BoolTensor：布尔型

注意事项：
    1. torch.Tensor 默认创建 FloatTensor
    2. 张量的形状（shape）很重要，影响后续操作
    3. GPU计算需要使用 .to('cuda') 或 .cuda()
================================================================================
"""

import torch

# ================================================================================
# 创建张量
# ================================================================================

# 方式1：创建未初始化的张量（不推荐，可能包含任意垃圾值）
# torch.Tensor(2, 3) 创建 2×3 的 FloatTensor

# 方式2：创建已初始化的张量
a = torch.Tensor(2, 3)  # 2行3列，未初始化
print("未初始化的张量:")
print(a)
print(f"形状: {a.shape}")
print(f"数据类型: {a.dtype}")

# 方式3：推荐使用的方式
b = torch.ones(2, 3)      # 全1张量
c = torch.zeros(2, 3)     # 全0张量
d = torch.randn(2, 3)     # 标准正态分布随机
e = torch.arange(0, 10, 2)  # 等差数列：[0, 2, 4, 6, 8]

print("\n推荐创建方式:")
print(f"torch.ones: {b}")
print(f"torch.zeros: {c}")
print(f"torch.randn: {d}")
print(f"torch.arange: {e}")

# ================================================================================
# 张量基本操作
# ================================================================================
print("\n基本操作示例:")
print(f"形状: {a.shape}")      # torch.Size([2, 3])
print(f"维度数: {a.dim()}")    # 2
print(f"元素总数: {a.numel()}")  # 6
print(f"重塑形状: {a.reshape(3, 2)}")  # 3×2

# ================================================================================
# 与NumPy互转
# ================================================================================
import numpy as np

print("\n与NumPy互转:")
np_array = np.array([[1, 2], [3, 4]])
torch_tensor = torch.from_numpy(np_array)
back_to_np = torch_tensor.numpy()

print(f"NumPy数组: {np_array}")
print(f"PyTorch张量: {torch_tensor}")
print(f"转回NumPy: {back_to_np}")

# 注意：共享内存，修改一个会影响另一个
np_array[0, 0] = 100
print(f"修改NumPy后，Tensor: {torch_tensor[0, 0]}")