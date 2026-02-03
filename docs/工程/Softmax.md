Softmax 是深度学习中最常用的激活函数之一，特别是在多分类问题中。它将任意实数值向量转换为概率分布。

---

### 一、基本定义

对于输入向量 $\mathbf{z} = [z_1, z_2, \ldots, z_K]^T$，Softmax 函数定义为：

$$
\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad i = 1, 2, \ldots, K
$$

其中 $K$ 是类别的数量，$\sigma(\mathbf{z})_i$ 表示第 $i$ 个类别的概率。

---

### 二、核心性质

**1. 概率归一化**：
$$
\sum_{i=1}^{K} \sigma(\mathbf{z})_i = \sum_{i=1}^{K} \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} = \frac{\sum_{i=1}^{K} e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} = 1
$$

**2. 输出范围**：
$$
0 < \sigma(\mathbf{z})_i < 1, \quad \forall i
$$

**3. 单调性**：Softmax 保持输入的相对顺序。如果 $z_i > z_j$，则 $\sigma(\mathbf{z})_i > \sigma(\mathbf{z})_j$。

**4. 指数放大差异**：当输入差异较大时，指数函数会放大这种差异，使得最大值对应的概率趋近于 1，其他趋近于 0。

---

### 三、直观理解

考虑三分类问题，输入向量 $\mathbf{z} = [2.0, 1.0, 0.1]^T$：

**步骤 1：计算指数**
$$
e^{z_1} = e^{2.0} \approx 7.389, \quad e^{z_2} = e^{1.0} \approx 2.718, \quad e^{z_3} = e^{0.1} \approx 1.105
$$

**步骤 2：计算归一化因子**
$$
\sum_{j=1}^{3} e^{z_j} = 7.389 + 2.718 + 1.105 = 11.212
$$

**步骤 3：计算概率**
$$
\sigma(\mathbf{z})_1 = \frac{7.389}{11.212} \approx 0.659, \quad \sigma(\mathbf{z})_2 = \frac{2.718}{11.212} \approx 0.242, \quad \sigma(\mathbf{z})_3 = \frac{1.105}{11.212} \approx 0.099
$$

验证：$0.659 + 0.242 + 0.099 = 1$ ✓

---

### 四、梯度计算（反向传播）

Softmax 常与交叉熵损失函数结合使用。记：
- $\mathbf{z}$ 为 logits（未归一化的输出）
- $\mathbf{y}$ 为 one-hot 编码的真实标签
- $\mathbf{p} = \sigma(\mathbf{z})$ 为预测概率

交叉熵损失：
$$
L = -\sum_{i=1}^{K} y_i \log(p_i)
$$

对 $z_j$ 的梯度：
$$
\frac{\partial L}{\partial z_j} = p_j - y_j
$$

**证明**：
$$
\frac{\partial L}{\partial z_j} = \sum_{i=1}^{K} \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_j}
$$

其中：
$$
\frac{\partial L}{\partial p_i} = -\frac{y_i}{p_i}
$$

对于 Softmax 的 Jacobian 矩阵（详细推导）：

首先，Softmax 的输出是一个向量 $\mathbf{p} = [p_1, p_2, \ldots, p_K]^T$，其中：
$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} = \frac{e^{z_i}}{S}
$$
其中 $S = \sum_{j=1}^{K} e^{z_j}$ 是归一化因子。

**情况 1：当 $i = j$ 时（对角线元素）**

使用商法则求导：
$$
\frac{\partial p_i}{\partial z_i} = \frac{\partial}{\partial z_i}\left(\frac{e^{z_i}}{S}\right)
$$

由于 $S$ 也依赖于 $z_i$，展开：
$$
\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} \cdot S - e^{z_i} \cdot \frac{\partial S}{\partial z_i}}{S^2}
$$

计算 $\frac{\partial S}{\partial z_i}$：
$$
\frac{\partial S}{\partial z_i} = \frac{\partial}{\partial z_i}\sum_{j=1}^{K} e^{z_j} = e^{z_i}
$$

代回：
$$
\begin{aligned}
\frac{\partial p_i}{\partial z_i} &= \frac{e^{z_i} \cdot S - e^{z_i} \cdot e^{z_i}}{S^2} \\
&= \frac{e^{z_i}}{S} - \frac{e^{z_i} \cdot e^{z_i}}{S^2} \\
&= p_i - p_i^2 \\
&= p_i(1 - p_i)
\end{aligned}
$$

**情况 2：当 $i \neq j$ 时（非对角线元素）**

此时 $p_i$ 的分子不包含 $z_j$，但分母 $S$ 包含 $z_j$：
$$
\frac{\partial p_i}{\partial z_j} = \frac{\partial}{\partial z_j}\left(\frac{e^{z_i}}{S}\right) = e^{z_i} \cdot \left(-\frac{1}{S^2}\right) \cdot \frac{\partial S}{\partial z_j}
$$

计算 $\frac{\partial S}{\partial z_j}$：
$$
\frac{\partial S}{\partial z_j} = e^{z_j}
$$

代回：
$$
\begin{aligned}
\frac{\partial p_i}{\partial z_j} &= e^{z_i} \cdot \left(-\frac{1}{S^2}\right) \cdot e^{z_j} \\
&= -\frac{e^{z_i} e^{z_j}}{S^2} \\
&= -\left(\frac{e^{z_i}}{S}\right) \left(\frac{e^{z_j}}{S}\right) \\
&= -p_i p_j
\end{aligned}
$$

**完整的 Jacobian 矩阵**：
$$
J = \frac{\partial \mathbf{p}}{\partial \mathbf{z}} = \begin{bmatrix}
p_1(1-p_1) & -p_1 p_2 & \cdots & -p_1 p_K \\
-p_2 p_1 & p_2(1-p_2) & \cdots & -p_2 p_K \\
\vdots & \vdots & \ddots & \vdots \\
-p_K p_1 & -p_K p_2 & \cdots & p_K(1-p_K)
\end{bmatrix}
$$

**性质验证**：

1. **行和为零**：每行的和为 0
   $$
   \sum_{j=1}^{K} \frac{\partial p_i}{\partial z_j} = p_i(1-p_i) + \sum_{j \neq i} (-p_i p_j) = p_i - p_i^2 - p_i \sum_{j \neq i} p_j = p_i - p_i(p_i + \sum_{j \neq i} p_j) = p_i - p_i \cdot 1 = 0
   $$
   这反映了概率约束 $\sum_i p_i = 1$。

2. **对称性**：$J_{ij} = -p_i p_j$，$J_{ji} = -p_j p_i$，满足 $J_{ij} = J_{ji}$。

**数值示例**（三分类，$\mathbf{z} = [2.0, 1.0, 0.1]^T$）：
- 预测概率：$\mathbf{p} = [0.659, 0.242, 0.099]^T$
- Jacobian 矩阵：
$$
J = \begin{bmatrix}
0.659(1-0.659) & -0.659 \times 0.242 & -0.659 \times 0.099 \\
-0.242 \times 0.659 & 0.242(1-0.242) & -0.242 \times 0.099 \\
-0.099 \times 0.659 & -0.099 \times 0.242 & 0.099(1-0.099)
\end{bmatrix} = \begin{bmatrix}
0.225 & -0.159 & -0.065 \\
-0.159 & 0.183 & -0.024 \\
-0.065 & -0.024 & 0.089
\end{bmatrix}
$$
- 验证行和：$0.225 - 0.159 - 0.065 = 0$，其他行同理。

**向量化的矩阵形式**：
$$
J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T
$$
其中：
- $\text{diag}(\mathbf{p})$ 是对角矩阵，$\text{diag}(\mathbf{p})_{ii} = p_i$
- $\mathbf{p}\mathbf{p}^T$ 是外积矩阵，$(\mathbf{p}\mathbf{p}^T)_{ij} = p_i p_j$

**在反向传播中的应用**：

设损失函数 $L$ 对 Softmax 输出 $\mathbf{p}$ 的梯度为 $\nabla_{\mathbf{p}}L = [\frac{\partial L}{\partial p_1}, \ldots, \frac{\partial L}{\partial p_K}]^T$，则对 logits $\mathbf{z}$ 的梯度为：
$$
\nabla_{\mathbf{z}}L = J^T \cdot \nabla_{\mathbf{p}}L
$$

展开计算：
$$
\frac{\partial L}{\partial z_j} = \sum_{i=1}^{K} \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_j}
$$

当 $L$ 为交叉熵损失且 $\mathbf{y}$ 为 one-hot 标签时，$\frac{\partial L}{\partial p_i} = -\frac{y_i}{p_i}$，代入得：
$$
\frac{\partial L}{\partial z_j} = -\frac{y_j}{p_j} \cdot p_j(1-p_j) + \sum_{i \neq j} \left(-\frac{y_i}{p_i}\right) \cdot (-p_i p_j) = p_j - y_j
$$

这正是交叉熵 + Softmax 组合的简洁梯度形式。

**Hessian 矩阵（二阶导数）**：

对于二阶优化（如牛顿法），需要计算 $\frac{\partial^2 p_k}{\partial z_i \partial z_j}$。

对 Jacobian 求导：
$$
\frac{\partial^2 p_k}{\partial z_i \partial z_j} = \frac{\partial}{\partial z_j}\left(\frac{\partial p_k}{\partial z_i}\right)
$$

分三种情况讨论：

1. **$k = i = j$**：
$$
\frac{\partial^2 p_i}{\partial z_i^2} = \frac{\partial}{\partial z_i}[p_i(1-p_i)] = p_i'(1-p_i) + p_i(-p_i') = p_i(1-p_i) - p_i^2(1-p_i) = p_i(1-p_i)(1-2p_i)
$$

2. **$k = i \neq j$**：
$$
\frac{\partial^2 p_i}{\partial z_i \partial z_j} = \frac{\partial}{\partial z_j}[-p_i p_j] = -p_i' p_j - p_i p_j' = -p_i(1-p_i)p_j - p_i p_j(1-p_j) = -p_i p_j(2 - p_i - p_j)
$$

3. **$k \neq i$**：
$$
\frac{\partial^2 p_k}{\partial z_i \partial z_j} = \frac{\partial}{\partial z_j}[-p_k p_i] = -p_k' p_i - p_k p_i' = -p_k(1-p_k)p_i - p_k p_k p_i = -p_k p_i(1 - p_k)
$$

Hessian 矩阵为 $K \times K \times K$ 的三阶张量，计算复杂度为 $O(K^3)$，在深度学习中通常不直接使用二阶方法。

---

**补充：Jacobian 矩阵的另一种视角——从链式法则出发**

使用全微分形式推导更直观。设 $S = \sum_{k=1}^{K} e^{z_k}$，则 $p_i = e^{z_i} / S$。

对 $p_i$ 取全微分：
$$
dp_i = d\left(\frac{e^{z_i}}{S}\right) = \frac{e^{z_i} d z_i \cdot S - e^{z_i} d S}{S^2}
$$

计算 $dS$：
$$
dS = \sum_{k=1}^{K} e^{z_k} d z_k
$$

代入并整理：
$$
\begin{aligned}
dp_i &= \frac{e^{z_i} S d z_i - e^{z_i} \sum_{k} e^{z_k} d z_k}{S^2} \\
&= \frac{e^{z_i}}{S} d z_i - \frac{e^{z_i}}{S} \sum_{k} \frac{e^{z_k}}{S} d z_k \\
&= p_i d z_i - p_i \sum_{k} p_k d z_k
\end{aligned}
$$

分离 $dz_j$ 的系数：
$$
dp_i = p_i dz_i - p_i \sum_{k \neq i} p_k dz_k - p_i^2 dz_i = p_i(1-p_i) dz_i - p_i \sum_{k \neq i} p_k dz_k
$$

因此 $dz_j$ 的系数为：
$$
\frac{\partial p_i}{\partial z_j} = \begin{cases}
p_i(1-p_i), & j = i \\
-p_i p_j, & j \neq i
\end{cases}
$$

此推导利用全微分，避免了显式使用商法则，更直观地展示了各 $z_j$ 对 $p_i$ 的影响。

---

**补充：Jacobian 矩阵与 KL 散度的关系**

考虑真实分布 $\mathbf{y}$（one-hot）和预测分布 $\mathbf{p}$ 之间的 KL 散度：
$$
D_{KL}(\mathbf{y} \parallel \mathbf{p}) = \sum_{i=1}^{K} y_i \log\frac{y_i}{p_i}
$$

对 $z_j$ 求导：
$$
\frac{\partial}{\partial z_j} D_{KL}(\mathbf{y} \parallel \mathbf{p}) = \sum_{i=1}^{K} y_i \left(-\frac{1}{p_i}\right) \frac{\partial p_i}{\partial z_j} = -\sum_{i=1}^{K} \frac{y_i}{p_i} \frac{\partial p_i}{\partial z_j}
$$

代入 Jacobian 元素：
- 当 $i = j$：$-\frac{y_j}{p_j} \cdot p_j(1-p_j) = -y_j(1-p_j)$
- 当 $i \neq j$：$-\frac{y_i}{p_i} \cdot (-p_i p_j) = y_i p_j$

求和：
$$
\frac{\partial D_{KL}}{\partial z_j} = -y_j(1-p_j) + \sum_{i \neq j} y_i p_j = -y_j + y_j p_j + p_j \sum_{i \neq j} y_i = p_j - y_j
$$

与交叉熵梯度一致，因为 $D_{KL}(\mathbf{y} \parallel \mathbf{p}) = H(\mathbf{y}, \mathbf{p}) - H(\mathbf{y})$，而 $H(\mathbf{y})$ 为常数。

---

**补充：批量处理的 Jacobian**

设批量输入 $\mathbf{Z} \in \mathbb{R}^{N \times K}$（$N$ 为样本数），Softmax 输出 $\mathbf{P} = \text{Softmax}(\mathbf{Z})$。

对于单个样本 $n$，其 Jacobian 为 $K \times K$ 矩阵，整个批量的 Jacobian 为 $(N \times K) \times (N \times K)$ 的块对角矩阵：
$$
J_{\text{batch}} = \text{diag}(J_1, J_2, \ldots, J_N)
$$

其中每个块 $J_n = \text{diag}(\mathbf{p}_n) - \mathbf{p}_n \mathbf{p}_n^T$。

这意味着批量反向传播时，各样本的梯度计算相互独立，可并行处理。

---

**Jacobian 矩阵总结与实现要点**

| 性质 | 描述 |
|------|------|
| 维度 | $K \times K$ |
| 对角线 | $p_i(1-p_i)$ |
| 非对角线 | $-p_i p_j$ |
| 行和 | 0 |
| 对称性 | $J_{ij} = J_{ji}$ |
| 向量化形式 | $\text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$ |

**实现注意事项**：

1. **数值稳定**：计算 Jacobian 时使用对数空间或避免直接计算 $p_i p_j$（可能导致下溢）。

2. **稀疏性**：当使用 argmax 标签时，梯度 $\nabla_{\mathbf{z}}L = \mathbf{p} - \mathbf{y}$ 是稀疏的（仅在真实类别处非零）。

3. **置信度影响**：当 $p_i$ 接近 0 或 1 时，对角线元素 $p_i(1-p_i)$ 较小，梯度信号较弱，可能导致梯度消失。

4. **温度参数**：带温度 $\tau$ 的 Softmax：$p_i = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}$，此时 Jacobian 对角线变为 $\frac{1}{\tau} p_i(1-p_i)$，可通过调整 $\tau$ 控制梯度大小。

---

**完整的 Jacobian 矩阵简洁表达式**：
$$
\frac{\partial p_i}{\partial z_j} = \begin{cases}
p_i(1 - p_j), & i = j \\
-p_i p_j, & i \neq j
\end{cases}
$$

可用矩阵形式表示为：
$$
J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T
$$
其中 $\text{diag}(\mathbf{p})$ 是以 $\mathbf{p}$ 为对角线的对角矩阵，$\mathbf{p}\mathbf{p}^T$ 是外积矩阵。

综合：
$$
\begin{aligned}
\frac{\partial L}{\partial z_j} &= -\frac{y_j}{p_j} \cdot p_j(1 - p_j) + \sum_{i \neq j} \left(-\frac{y_i}{p_i}\right) \cdot (-p_i p_j) \\
&= -y_j(1 - p_j) + \sum_{i \neq j} y_i p_j \\
&= -y_j + y_j p_j + \sum_{i \neq j} y_i p_j \\
&= -y_j + p_j(y_j + \sum_{i \neq j} y_i) \\
&= -y_j + p_j \cdot 1 \\
&= p_j - y_j
\end{aligned}
$$

**关键结论**：梯度非常简洁——预测概率与真实标签的差值。

---

### 五、数值稳定性问题

直接计算 Softmax 可能存在数值溢出问题。例如，当 $z_i$ 很大时，$e^{z_i}$ 可能溢出。

**解决方案：使用最大值偏移**

Softmax 具有平移不变性：
$$
\frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{e^{z_i - C} \cdot e^C}{\sum_j e^{z_j - C} \cdot e^C} = \frac{e^{z_i - C}}{\sum_j e^{z_j - C}}
$$

实际实现中，取 $C = \max(\mathbf{z})$：

$$
\sigma(\mathbf{z})_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{K} e^{z_j - \max(\mathbf{z})}}
$$

这确保了所有指数的输入 $\leq 0$，避免了数值溢出。

---

### 六、二分类特例：Sigmoid

当 $K = 2$ 时，Softmax 退化为 Sigmoid 函数。

设 $\mathbf{z} = [z, 0]^T$（二分类通常只需一个输出）：

$$
\sigma(\mathbf{z})_1 = \frac{e^z}{e^z + e^0} = \frac{e^z}{e^z + 1} = \frac{1}{1 + e^{-z}} = \text{sigmoid}(z)
$$

Sigmoid 是 Softmax 在二分类情况下的特殊形式。

---

### 七、应用场景

**1. 多分类问题**：神经网络输出层，如图像分类、文本分类。

**2. 注意力机制**：Transformer 中的注意力权重计算。

**3. 强化学习**：策略网络输出动作概率分布。

**4. 序列标注**：如命名实体识别（NER）、词性标注。

---

### 八、总结

Softmax 的核心价值：
- 将任意实数值映射为有效的概率分布
- 梯度计算简单高效（与交叉熵结合时）
- 通过指数函数放大最大值，提高分类置信度
- 具有平移不变性，数值稳定易实现

代码实现示例（Python）：

```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  # 数值稳定
    return e_x / e_x.sum(axis=-1, keepdims=True)

# 示例
z = np.array([2.0, 1.0, 0.1])
p = softmax(z)  # [0.659, 0.242, 0.099]
```