import numpy as np
import matplotlib.pyplot as plt

# 数据
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 3.5, 4.5])

# 最小二乘法解
X = np.vstack([np.ones(3), x]).T
theta_ols = np.linalg.inv(X.T @ X) @ X.T @ y
theta0_ols, theta1_ols = theta_ols[0], theta_ols[1]

# 梯度下降
def gradient_descent(x, y, alpha=0.1, iterations=50):
    theta0, theta1 = 0, 0  # 初始值
    m = len(x)
    history = []  # 记录参数变化
    
    for i in range(iterations):
        # 预测值
        y_pred = theta0 + theta1 * x
        
        # 梯度
        grad0 = np.sum(y_pred - y) / m
        grad1 = np.sum((y_pred - y) * x) / m
        
        # 更新参数
        theta0 -= alpha * grad0
        theta1 -= alpha * grad1
        
        # 记录历史
        history.append((theta0, theta1, i))
        
        # 检查收敛
        if i > 0 and abs(grad0) < 1e-6 and abs(grad1) < 1e-6:
            break
    
    return theta0, theta1, history

# 运行梯度下降
theta0_gd, theta1_gd, history = gradient_descent(x, y, alpha=0.2, iterations=20)

# 可视化
plt.figure(figsize=(15, 5))

# 1. 数据点和拟合线
plt.subplot(1, 3, 1)
x_line = np.linspace(0, 4, 100)

# 最小二乘拟合线
y_ols = theta0_ols + theta1_ols * x_line
# 梯度下降拟合线（最终）
y_gd = theta0_gd + theta1_gd * x_line

plt.scatter(x, y, color='red', s=100, label='数据点', zorder=5)
plt.plot(x_line, y_ols, 'b-', linewidth=2, label=f'最小二乘: y={theta0_ols:.3f}+{theta1_ols:.3f}x')
plt.plot(x_line, y_gd, 'g--', linewidth=2, label=f'梯度下降: y={theta0_gd:.3f}+{theta1_gd:.3f}x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('拟合结果对比')
plt.legend()
plt.grid(True)

# 2. 代价函数等高线
plt.subplot(1, 3, 2)
# 生成网格
theta0_range = np.linspace(-1, 2, 100)
theta1_range = np.linspace(-1, 3, 100)
Theta0, Theta1 = np.meshgrid(theta0_range, theta1_range)

# 计算每个点的代价
J = np.zeros_like(Theta0)
for i in range(len(theta0_range)):
    for j in range(len(theta1_range)):
        y_pred = Theta0[j, i] + Theta1[j, i] * x
        J[j, i] = np.sum((y_pred - y)**2) / (2*len(x))

# 绘制等高线
contour = plt.contour(Theta0, Theta1, J, levels=20, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)

# 标记最小二乘解
plt.scatter(theta0_ols, theta1_ols, color='blue', s=100, label='最小二乘解', zorder=5)

# 标记梯度下降路径
theta0_history = [h[0] for h in history]
theta1_history = [h[1] for h in history]
plt.plot(theta0_history, theta1_history, 'r-', linewidth=2, label='梯度下降路径')
plt.scatter(theta0_history, theta1_history, color='red', s=20, zorder=4)

plt.xlabel('θ₀ (截距)')
plt.ylabel('θ₁ (斜率)')
plt.title('代价函数等高线与优化路径')
plt.legend()
plt.grid(True)

# 3. 迭代过程中代价函数的变化
plt.subplot(1, 3, 3)
iterations = [h[2] for h in history]
costs = []
for theta0, theta1, _ in history:
    y_pred = theta0 + theta1 * x
    cost = np.sum((y_pred - y)**2) / (2*len(x))
    costs.append(cost)

plt.plot(iterations, costs, 'b-o', linewidth=2, markersize=6)
plt.axhline(y=0.02083, color='r', linestyle='--', label='最小二乘代价')
plt.xlabel('迭代次数')
plt.ylabel('代价函数 J(θ)')
plt.title('梯度下降过程中代价函数的变化')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 输出结果对比
print("=== 结果对比 ===")
print(f"最小二乘法: θ₀ = {theta0_ols:.6f}, θ₁ = {theta1_ols:.6f}")
print(f"梯度下降:   θ₀ = {theta0_gd:.6f}, θ₁ = {theta1_gd:.6f}")
print(f"\n绝对误差: Δθ₀ = {abs(theta0_ols-theta0_gd):.6f}, Δθ₁ = {abs(theta1_ols-theta1_gd):.6f}")