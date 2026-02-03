# Machine Learning Code Documentation

本文档对 `code/` 目录下的所有代码文件进行梳理和说明，涵盖梯度下降、优化算法、线性回归、逻辑回归、Softmax 等核心机器学习概念。

---

## 一、代码文件总览

| 文件名 | 功能 | 核心API/函数 | 关键参数 |
|--------|------|--------------|----------|
| `gradient.py` | 基础梯度下降 | 手动实现 | x, y, alpha, n |
| `large_alpha.py` | 学习率过大导致梯度爆炸 | 演示发散 | alpha=100 |
| `tensor.py` | PyTorch张量基础 | torch.Tensor, torch.ones/zeros/randn | shape, dtype |
| `tensor_gradient.py` | PyTorch自动微分梯度下降 | torch.autograd | requires_grad |
| `pytorch_gradient.py` | PyTorch自动微分求导 | backward(), detach() | linspace, requires_grad |
| `print_daoshu.py` | 手动计算函数导数 | 手动df/dx | x**2+3*x+2 |
| `gradient_theta0.py` | 线性回归梯度下降 | gradient_des, costJ | x, y, alpha, n |
| `gradient_theta0_house.py` | 房价预测-线性回归 | gradient_des | LotArea -> SalePrice |
| `er_cheng_fa.py` | 正规方程（解析解） | normal_equation_array/matrix | X, y |
| `logic_gradient.py` | 逻辑回归梯度下降 | gradient_descent, sigmoid | X, y, alpha, it |
| `sgd.py` | SGD优化器 | torch.optim.SGD | lr=0.001 |
| `adam.py` | Adam优化器 | torch.optim.Adam | lr=0.001 |
| `lbfgs.py` | LBFGS优化器 | torch.optim.LBFGS | closure函数 |
| `linear_regression_data.py` | 数据预处理 | pandas, train_test_split | 数据标准化 |
| `linear_regression.py` | PyTorch线性回归 | nn.Linear, MSELoss | 12个特征 |
| `softmax.py` | Softmax多分类 | SoftmaxRegression, CrossEntropyLoss | 3分类 |
| `最小二乘法_梯度下降.py` | 最小二乘法与梯度下降对比 | np.linalg.inv, gradient_descent | 代价函数可视化 |

---

## 二、核心函数详细说明

### 2.1 梯度下降相关

#### `gradient_descent`（基础版）
```python
def gradient_descent(x, y, alpha, iterations):
    """
    功能：使用梯度下降算法求解线性回归参数
    
    参数:
        x: 特征数组，shape = (m,)
        y: 目标数组，shape = (m,)
        alpha: 学习率 (建议范围: 1e-4 ~ 0.1)
        iterations: 迭代次数
    
    返回:
        theta0, theta1: 最优参数
    
    原理:
        θ = θ - α * ∇J(θ)
        
    注意事项:
        1. 学习率过小会导致收敛慢
        2. 学习率过大会导致发散
        3. 需要特征标准化
        4. 建议添加收敛判断（如梯度<1e-6）
    """
```

#### `gradient_des`（含代价函数）
```python
def gradient_des(x, y, a, n):
    """
    功能：带代价函数计算的梯度下降
    
    参数:
        x: 特征数组
        y: 目标数组  
        a: 学习率 alpha
        n: 迭代次数
    
    包含函数:
        - gradient_theta0(): 计算 ∂J/∂θ₀
        - gradient_theta1(): 计算 ∂J/∂θ₁
        - costJ(): 计算均方误差 J(θ)
    
    代价函数公式:
        J(θ) = 1/(2m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
    
    梯度公式:
        ∂J/∂θ₀ = 1/m * Σ(h(x) - y)
        ∂J/∂θ₁ = 1/m * Σ((h(x) - y) * x)
    """
```

### 2.2 优化器相关

#### SGD优化器
```python
# PyTorch标准使用流程
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(epochs):
    optimizer.zero_grad()           # 1. 清空梯度
    loss = criterion(output, target)  # 2. 前向传播
    loss.backward()                 # 3. 反向传播
    optimizer.step()                # 4. 更新参数

# 注意事项:
# 1. 每次迭代必须清空梯度
# 2. 学习率需要根据问题调整
# 3. 可以添加动量(momentum)加速收敛
```

#### Adam优化器
```python
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=0.001,           # 学习率（默认0.001）
    betas=(0.9, 0.999), # 一阶/二阶矩估计的指数衰减率
    eps=1e-8,           # 数值稳定性项
    weight_decay=0      # L2正则化系数
)

# 特点:
# 1. 自适应学习率
# 2. 对超参数选择鲁棒
# 3. 推荐作为默认优化器
```

#### LBFGS优化器
```python
optimizer = torch.optim.LBFGS(model.parameters(), lr=1)

def closure():
    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()
    return loss

for epoch in range(epochs):
    optimizer.step(closure)  # 需要传入closure函数

# 特点:
# 1. 拟牛顿法，二阶优化
# 2. 内存占用大
# 3. 需要closure函数重新计算前向传播
# 4. 适合小数据集
```

### 2.3 线性回归相关

#### 正规方程（解析解）
```python
def normal_equation_array(X, y):
    """
    功能：通过正规方程直接求解线性回归参数
    
    公式: θ = (X^T X)^(-1) X^T y
    
    参数:
        X: 设计矩阵，shape = (m, n+1)，第一列为全1
        y: 目标向量，shape = (m,)
    
    返回:
        θ: 参数向量
    
    注意事项:
        1. X^T X 必须可逆（需检查行列式或使用伪逆）
        2. 计算复杂度 O(n³)，不适合高维数据
        3. 不需要学习率
        4. 不能进行在线学习
    """
    
def normal_equation_matrix(X, y):
    """
    使用NumPy矩阵实现
    
    与array版的区别:
        - 使用np.asmatrix转为矩阵类型
        - 使用*.I求逆，*表示矩阵乘法
    """
```

#### PyTorch线性回归
```python
class LinearRegression(nn.Module):
    """
    功能：PyTorch实现的线性回归模型
    
    组成:
        - nn.Linear(12, 1): 12个输入特征，1个输出
    
    训练流程:
        1. 前向传播: model(x)
        2. 计算损失: criterion(h, y)
        3. 反向传播: loss.backward()
        4. 更新参数: optimizer.step()
        5. 清空梯度: optimizer.zero_grad()
    
    注意事项:
        1. model.train() / model.eval() 切换训练/评估模式
        2. 评估时使用 with torch.no_grad(): 节省内存
        3. detach() 用于从计算图分离，转为numpy
    """
```

### 2.4 逻辑回归相关

#### Sigmoid函数
```python
def sigmod(z):
    """
    Sigmoid激活函数
    
    公式: σ(z) = 1 / (1 + e^(-z))
    
    参数:
        z: logit值（任意实数）
    
    返回:
        概率值 (0, 1)
    
    特点:
        1. 将任意实数映射到(0,1)
        2. 中心对称点在0.5
        3. 导数: σ'(z) = σ(z)(1-σ(z))
    """

def hypothesis(theta, x, n):
    """
    逻辑回归假设函数
    
    公式: h_θ(x) = σ(θ^T x)
    
    参数:
        theta: 参数向量（包括偏置）
        x: 特征向量（包括偏置特征1）
        n: 特征数量（不包括偏置）
    
    返回:
        预测为正类的概率
    """
```

#### 逻辑回归代价函数
```python
def costJ(x, y, theta, m, n):
    """
    逻辑回归的交叉熵代价函数
    
    公式: J(θ) = -1/m * Σ[y*log(h) + (1-y)*log(1-h)]
    
    参数:
        x: 特征矩阵（包括偏置列）
        y: 标签（0或1）
        theta: 参数向量
        m: 样本数
        n: 特征数
    
    注意事项:
        1. 内部使用log，需防止h=0或h=1导致的数值问题
        2. 代价越低，模型越好
        3. 是凸函数，保证全局最优
    """
```

#### 逻辑回归梯度下降
```python
def gradient_descent(x, y, alpha, m, n, it):
    """
    逻辑回归梯度下降
    
    梯度公式: ∂J/∂θ_j = 1/m * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x⁽ⁱ⁾_j
    
    参数:
        x: 特征矩阵，shape = (m, n+1)
        y: 标签向量，shape = (m,)
        alpha: 学习率
        m: 样本数
        n: 特征数（不包括偏置）
        it: 迭代次数
    
    注意事项:
        1. 需要在x第一列添加全1列（偏置特征）
        2. 初始theta通常设为0
        3. 学习率通常比线性回归更小
    """
```

### 2.5 Softmax相关

#### Softmax回归模型
```python
class SoftmaxRegression(nn.Module):
    """
    功能：Softmax多分类回归
    
    模型: z = X * W + b, p = softmax(z)
    
    参数:
        features: 输入特征数
        classes: 类别数
    
    输出:
        logits (未归一化的类别分数)
    
    注意事项:
        1. 输出层不需要激活函数（CrossEntropyLoss包含softmax）
        2. 标签需要是LongTensor（类别索引）
    """
```

#### 训练函数
```python
def train_model(model, train_data, label, epochs=10000, rate=0.01):
    """
    功能：训练Softmax回归模型
    
    参数:
        model: PyTorch模型
        train_data: 训练特征，shape = (m, features)
        label: 类别标签，shape = (m,)，类型为LongTensor
        epochs: 迭代次数
        rate: 学习率
    
    损失函数: nn.CrossEntropyLoss()
        - 内部包含LogSoftmax + NLLLoss
        - 等价于 交叉熵 + Softmax
    
    优化器: SGD
        - 学习率通常0.01~0.1
        - 可使用Adam替代
    
    注意事项:
        1. model.train() 设为训练模式
        2. optimizer.zero_grad() 清空梯度
        3. loss.backward() 反向传播
        4. optimizer.step() 更新参数
    """
```

#### 决策边界可视化
```python
def draw_decision_boundary(minx1, maxx1, minx2, maxx2, model, device='cpu'):
    """
    功能：绘制分类决策边界
    
    参数:
        minx1, maxx1: 第一个特征的显示范围
        minx2, maxx2: 第二个特征的显示范围
        model: 训练好的模型
        device: 计算设备
    
    原理:
        1. 生成网格点
        2. 对每个点预测类别
        3. 使用等高线绘制决策区域
    
    注意事项:
        1. model.eval() 设为评估模式
        2. with torch.no_grad(): 禁用梯度计算
        3. 使用contourf()绘制决策区域
        4. 使用contour()绘制决策边界线
    """
```

---

## 三、PyTorch自动微分系统

### 3.1 核心概念

```python
# requires_grad: 标记需要计算梯度的张量
x = torch.tensor([1.0], requires_grad=True)

# backward(): 自动计算梯度
loss.backward()

# grad: 存储梯度值
print(x.grad)

# zero_(): 清空梯度
optimizer.zero_grad()  # 推荐方式
x.grad.zero_()         # 等价
```

### 3.2 重要方法

| 方法 | 功能 | 示例 |
|------|------|------|
| `requires_grad` | 标记是否追踪梯度 | `x.requires_grad=True` |
| `backward()` | 反向传播计算梯度 | `loss.backward()` |
| `grad` | 访问梯度值 | `x.grad` |
| `detach()` | 从计算图分离 | `x.detach()` |
| `data` | 访问内部数据 | `x.data` |
| `zero_()` | 清空梯度 | `x.grad.zero_()` |

### 3.3 注意事项

```python
# 1. backward()只能对标量调用
y.sum().backward()  # 正确
# y.backward()      # 错误

# 2. 转换numpy前必须detach
y_np = y.detach().numpy()

# 3. 梯度会累积，需要清零
optimizer.zero_grad()

# 4. 使用data而非直接赋值避免追踪
x.data -= alpha * x.grad  # 推荐
# x -= alpha * x.grad     # 会报错（涉及计算图操作）
```

---

## 四、数据预处理

### 4.1 数据标准化
```python
# Z-score标准化
features = data.columns.difference(['price'])
data[features] = (data[features] - data[features].mean()) / data[features].std()

# 作用:
# 1. 消除特征量纲影响
# 2. 加速梯度下降收敛
# 3. 提高模型稳定性

# Min-Max标准化（另一种方式）
# data[features] = (data[features] - min) / (max - min)
```

### 4.2 数据分割
```python
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(
    data, 
    test_size=0.15,      # 测试集比例
    random_state=42      # 随机种子（可复现）
)

# 注意事项:
# 1. test_size 通常 0.1 ~ 0.3
# 2. random_state 保证结果可复现
# 3. 可以添加 stratify 参数保持类别比例
```

### 4.3 分类变量处理
```python
# 类别编码
data['city'] = data['city'].astype('category').cat.codes

# 删除无关特征
data = data.drop(columns=['date', 'waterfront', 'view', 'street', 'country'])

# 单位转换
data['price'] = data['price'] / 10000  # 万元
```

---

## 五、常见问题与解决方案

### 5.1 梯度消失/爆炸

**问题表现：**
- 梯度接近0，参数不更新
- 梯度非常大，参数跳跃

**解决方案：**
```python
# 1. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. 使用自适应优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. 权重初始化
torch.nn.init.xavier_uniform_(layer.weight)
torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

# 4. Batch Normalization
nn.BatchNorm1d(num_features)
```

### 5.2 数值稳定性

**问题：**
- log(0) 导致 -inf
- exp(大数) 导致 inf
- 除以0导致 nan

**解决方案：**
```python
# 1. 防止log(0)
epsilon = 1e-15
p = torch.clamp(pred, epsilon, 1 - epsilon)
log_p = torch.log(p)

# 2. Softmax数值稳定
def stable_softmax(z):
    z_shifted = z - torch.max(z, dim=-1, keepdim=True)[0]
    exp_z = torch.exp(z_shifted)
    return exp_z / torch.sum(exp_z, dim=-1, keepdim=True)

# 3. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 5.3 学习率选择

| 方法 | 建议范围 | 场景 |
|------|----------|------|
| SGD | 0.01 ~ 0.1 | 大数据集，简单模型 |
| Adam | 0.0001 ~ 0.001 | 通用，推荐默认值 |
| LBFGS | 0.5 ~ 2 | 小数据集，二阶信息 |

**学习率调度：**
```python
# 1. 阶梯衰减
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=1000, gamma=0.1
)

# 2. 余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=0
)

# 3. ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10
)
```

---

## 六、代码执行顺序建议

### 入门学习路径
1. `gradient.py` → 理解梯度下降基本概念
2. `tensor.py` → 掌握PyTorch张量操作
3. `tensor_gradient.py` → 学习自动微分机制
4. `gradient_theta0.py` → 线性回归梯度下降实现
5. `最小二乘法_梯度下降.py` → 解析解与数值解对比

### 进阶学习路径
1. `logic_gradient.py` → 逻辑回归（分类问题）
2. `softmax.py` → 多分类问题
3. `linear_regression.py` → PyTorch完整训练流程
4. `sgd.py` / `adam.py` / `lbfgs.py` → 优化器对比

### 实践项目
1. `linear_regression_data.py` → 数据预处理
2. `linear_regression.py` → 房价预测模型

---

## 七、参考公式速查

### 线性回归
- 假设函数: $h_θ(x) = θ^T x$
- 代价函数: $J(θ) = \frac{1}{2m}\sum_{i=1}^m(h_θ(x^{(i)}) - y^{(i)})²$
- 梯度: $\frac{\partial J}{\partial θ_j} = \frac{1}{m}\sum_{i=1}^m(h_θ(x^{(i)}) - y^{(i)})x_j^{(i)}$

### 逻辑回归
- Sigmoid: $σ(z) = \frac{1}{1+e^{-z}}$
- 代价函数: $J(θ) = -\frac{1}{m}\sum_{i=1}^m[y^{(i)}\log(h_θ(x^{(i)})) + (1-y^{(i)})\log(1-h_θ(x^{(i)}))]$
- 梯度: $\frac{\partial J}{\partial θ_j} = \frac{1}{m}\sum_{i=1}^m(h_θ(x^{(i)}) - y^{(i)})x_j^{(i)}$

### Softmax
- Softmax: $p_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$
- 交叉熵: $L = -\sum_{i=1}^K y_i \log p_i$
- 梯度: $\frac{\partial L}{\partial z_j} = p_j - y_j$

### 优化器
- SGD: $θ = θ - α \cdot ∇J(θ)$
- Adam: $m_t = β_1 m_{t-1} + (1-β_1)g_t$, $v_t = β_2 v_{t-1} + (1-β_2)g_t²$
- LBFGS: 拟牛顿法，二阶优化