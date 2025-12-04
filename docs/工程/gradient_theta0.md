## 线性回归代价函数偏导数推导

对于线性回归模型，假设函数为：

$$
h_\theta(x) = \theta_0 + \theta_1 x
$$

代价函数（均方误差）为：

$$
J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2
$$

其中 $ m $ 是样本数量。

---

### 1. 对 $\theta_0$ 的偏导数推导

将假设函数代入代价函数：

$$
J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^m (\theta_0 + \theta_1 x^{(i)} - y^{(i)})^2
$$

对 $\theta_0$ 求偏导（应用链式法则）：

$$
\frac{\partial J}{\partial \theta_0} = \frac{1}{2m} \sum_{i=1}^m 2(\theta_0 + \theta_1 x^{(i)} - y^{(i)}) \cdot \frac{\partial}{\partial \theta_0}(\theta_0 + \theta_1 x^{(i)} - y^{(i)})
$$

因为：
$$
\frac{\partial}{\partial \theta_0}(\theta_0 + \theta_1 x^{(i)} - y^{(i)}) = 1
$$

所以：

$$
\frac{\partial J}{\partial \theta_0} = \frac{1}{2m} \sum_{i=1}^m 2(\theta_0 + \theta_1 x^{(i)} - y^{(i)}) \cdot 1
$$

化简：

$$
\frac{\partial J}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^m (\theta_0 + \theta_1 x^{(i)} - y^{(i)})
$$

这就是代码中计算的对 $\theta_0$ 的偏导数。

---

### 2. 对 $\theta_1$ 的偏导数推导（补充）

类似地，对 $\theta_1$ 求偏导：

$$
\frac{\partial J}{\partial \theta_1} = \frac{1}{2m} \sum_{i=1}^m 2(\theta_0 + \theta_1 x^{(i)} - y^{(i)}) \cdot \frac{\partial}{\partial \theta_1}(\theta_0 + \theta_1 x^{(i)} - y^{(i)})
$$

因为：
$$
\frac{\partial}{\partial \theta_1}(\theta_0 + \theta_1 x^{(i)} - y^{(i)}) = x^{(i)}
$$

所以：

$$
\frac{\partial J}{\partial \theta_1} = \frac{1}{m} \sum_{i=1}^m (\theta_0 + \theta_1 x^{(i)} - y^{(i)}) \cdot x^{(i)}
$$

---

### 3. 代码中的实现

代码中实现的是对 $\theta_0$ 的偏导数计算：
```python
def gradient_theta0(x, y, theta0, theta1):
    m = len(x)
    h = theta0 + theta1 * x  # 预测值
    return np.sum(h - y) / m  # ∂J/∂θ0
```

注意：原始代价函数中的系数 $ \frac{1}{2} $ 在求导后与平方项的导数 2 抵消，因此最终偏导数公式中没有 $ \frac{1}{2} $。