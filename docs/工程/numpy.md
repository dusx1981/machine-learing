# NumPy 库详细介绍

NumPy（Numerical Python）是 Python 科学计算的基础库，提供了高性能的多维数组对象和用于处理这些数组的工具。

## 一、NumPy 的核心特性

### 1. **多维数组对象 (ndarray)**
- 同质数据类型的高效存储
- 支持向量化操作
- 广播功能

### 2. **性能优势**
- 底层用 C 和 Fortran 实现
- 比纯 Python 列表快 10-100 倍
- 内存效率高

### 3. **丰富的数学函数**
- 线性代数
- 傅里叶变换
- 随机数生成
- 统计运算

---

## 二、安装与导入

```bash
# 安装 NumPy
pip install numpy
```

```python
# 导入惯例
import numpy as np
```

---

## 三、核心数据结构：ndarray

### 1. 创建数组

```python
import numpy as np

# 从列表创建
arr1 = np.array([1, 2, 3, 4, 5])
print(f"一维数组: {arr1}")

# 二维数组
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"二维数组:\n{arr2}")

# 特殊数组创建方法
zeros = np.zeros((3, 4))        # 全零数组
ones = np.ones((2, 3))          # 全一数组
empty = np.empty((2, 2))        # 未初始化数组
full = np.full((2, 2), 7)       # 填充指定值
identity = np.eye(3)            # 单位矩阵
range_arr = np.arange(0, 10, 2) # 类似 range
linspace = np.linspace(0, 1, 5) # 等间距数组
```

### 2. 数组属性

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(f"数组维度: {arr.ndim}")        # 2
print(f"数组形状: {arr.shape}")       # (2, 3)
print(f"数组大小: {arr.size}")        # 6
print(f"数据类型: {arr.dtype}")       # int64
print(f"元素大小: {arr.itemsize}")    # 8 bytes
```

---

## 四、数组操作

### 1. 索引和切片

```python
arr = np.array([[1, 2, 3, 4], 
                [5, 6, 7, 8], 
                [9, 10, 11, 12]])

# 基本索引
print(arr[0, 1])        # 2
print(arr[1])           # [5, 6, 7, 8]

# 切片
print(arr[0:2, 1:3])    # [[2, 3], [6, 7]]
print(arr[:, 2])        # [3, 7, 11] - 所有行的第3列

# 布尔索引
bool_idx = arr > 5
print(bool_idx)
print(arr[bool_idx])    # [6, 7, 8, 9, 10, 11, 12]
```

### 2. 形状操作

```python
arr = np.arange(12)

# 改变形状
reshaped = arr.reshape(3, 4)
print(f"重塑后:\n{reshaped}")

# 展平
flattened = reshaped.flatten()
print(f"展平: {flattened}")

# 转置
transposed = reshaped.T
print(f"转置:\n{transposed}")
```

### 3. 数组拼接和分割

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

# 垂直拼接
v_stack = np.vstack((a, b))
print(f"垂直拼接:\n{v_stack}")

# 水平拼接
h_stack = np.hstack((a, b.T))
print(f"水平拼接:\n{h_stack}")

# 分割
arr = np.arange(12).reshape(3, 4)
split_arr = np.split(arr, 2, axis=1)  # 沿列分割
print(f"分割结果:")
for i, part in enumerate(split_arr):
    print(f"部分 {i}:\n{part}")
```

---

## 五、数学运算

### 1. 基本运算

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"加法: {a + b}")           # [5, 7, 9]
print(f"减法: {a - b}")           # [-3, -3, -3]
print(f"乘法: {a * b}")           # [4, 10, 18] (逐元素)
print(f"除法: {b / a}")           # [4., 2.5, 2.]
print(f"平方: {a**2}")            # [1, 4, 9]
```

### 2. 矩阵运算

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵乘法
dot_product = np.dot(A, B)
print(f"矩阵乘法:\n{dot_product}")

# 或者使用 @ 运算符
at_operator = A @ B
print(f"@运算符:\n{at_operator}")

# 其他矩阵运算
print(f"转置:\n{A.T}")
print(f"逆矩阵:\n{np.linalg.inv(A)}")
print(f"行列式: {np.linalg.det(A)}")
```

### 3. 统计运算

```python
arr = np.array([[1, 2, 3], 
                [4, 5, 6]])

print(f"总和: {np.sum(arr)}")              # 21
print(f"每列和: {np.sum(arr, axis=0)}")    # [5, 7, 9]
print(f"每行和: {np.sum(arr, axis=1)}")    # [6, 15]
print(f"均值: {np.mean(arr)}")             # 3.5
print(f"标准差: {np.std(arr)}")            # 1.707...
print(f"最大值: {np.max(arr)}")            # 6
print(f"最小值: {np.min(arr)}")            # 1
```

---

## 六、广播机制

广播是 NumPy 的强大特性，允许不同形状的数组进行数学运算。

```python
# 标量与数组
arr = np.array([1, 2, 3])
result = arr + 5
print(f"数组 + 标量: {result}")  # [6, 7, 8]

# 不同形状数组
A = np.array([[1, 2, 3], 
              [4, 5, 6]])
B = np.array([10, 20, 30])

result = A + B  # B 被广播到与 A 相同的形状
print(f"广播加法:\n{result}")
# [[11, 22, 33],
#  [14, 25, 36]]
```

---

## 七、实际应用示例

### 1. 机器学习中的数据预处理

```python
# 假设我们有房价数据
# 面积(平方米), 卧室数, 房龄, 价格(万元)
data = np.array([
    [100, 3, 5, 300],
    [150, 4, 10, 450],
    [200, 5, 15, 600],
    [120, 3, 8, 360]
])

# 分离特征和目标
X = data[:, :-1]  # 所有行，除了最后一列
y = data[:, -1]   # 所有行，最后一列

print(f"特征矩阵 X:\n{X}")
print(f"目标向量 y: {y}")

# 特征标准化
X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
print(f"标准化后的 X:\n{X_normalized}")
```

### 2. 简单的线性回归实现

```python
def linear_regression(X, y, learning_rate=0.01, iterations=1000):
    """使用梯度下降实现线性回归"""
    m, n = X.shape
    theta = np.zeros(n)  # 参数初始化
    cost_history = []
    
    # 添加偏置项
    X_b = np.c_[np.ones(m), X]
    
    for i in range(iterations):
        predictions = X_b.dot(theta)
        errors = predictions - y
        
        # 计算梯度
        gradients = (1/m) * X_b.T.dot(errors)
        
        # 更新参数
        theta = theta - learning_rate * gradients
        
        # 计算损失
        cost = (1/(2*m)) * np.sum(errors**2)
        cost_history.append(cost)
        
        if i % 100 == 0:
            print(f"迭代 {i}: 损失 = {cost:.4f}")
    
    return theta, cost_history

# 使用示例
X = np.array([[100], [150], [200], [120]])  # 房屋面积
y = np.array([300, 450, 600, 360])          # 价格

theta, history = linear_regression(X, y)
print(f"\n最终参数: {theta}")

# 预测新数据
new_house = np.array([180])
prediction = theta[0] + theta[1] * new_house
print(f"180平方米房屋预测价格: {prediction[0]:.1f}万元")
```

---

## 八、NumPy 的优势总结

1. **性能**：比纯 Python 快几个数量级
2. **简洁**：向量化操作代替循环
3. **功能丰富**：内置大量数学函数
4. **互操作性**：与其他科学计算库完美集成
5. **内存效率**：连续内存存储
6. **社区支持**：广泛使用，文档完善

---

## 九、常用函数速查

```python
# 创建数组
np.array(), np.zeros(), np.ones(), np.arange(), np.linspace()

# 数学运算
np.add(), np.subtract(), np.multiply(), np.divide()
np.sqrt(), np.exp(), np.log(), np.sin(), np.cos()

# 线性代数
np.dot(), np.linalg.inv(), np.linalg.det(), np.linalg.eig()

# 统计
np.mean(), np.median(), np.std(), np.var(), np.min(), np.max()

# 随机数
np.random.rand(), np.random.randn(), np.random.randint()
```

NumPy 是 Python 数据科学生态系统的基石，掌握 NumPy 是学习 Pandas、Scikit-learn、TensorFlow 等高级库的前提条件。