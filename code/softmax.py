# import numpy as np
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch

# def make_data(num):
#     green = np.random.randn(num, 2) + np.array([0, -2])
#     blue = np.random.randn(num, 2) + np.array([-2, 2])
#     red = np.random.randn(num, 2) + np.array([2, 2])
#     return green, blue, red

# def draw_decision_boundary(minx1, maxx1, minx2, maxx2, model):
#     # 调用mesh-grid生成网格数据点
#     # 每个点的距离是0.02，这样生成的点可以覆盖平面的全部范围
#     xx1, xx2 = np.meshgrid(np.arange(minx1, maxx1, 0.02),
#                            np.arange(minx2, maxx2, 0.02))
#     # 设置x1s、x2s和z分别表示数据点的横坐标、纵坐标和类别的预测结果
#     x1s = xx1.ravel()
#     x2s = xx2.ravel()
#     z = list()
    
#     for x1, x2 in zip(x1s, x2s):  # 遍历全部样本
#         # 将样本转为张量
#         test_point = torch.FloatTensor([[x1, x2]])
#         output = model(test_point)  # 使用model预测结果
#         # 选择概率最大的类别
#         _, predicted = torch.max(output, 1)
#         z.append(predicted.item())  # 添加到高度z中
    
#     # 将z重新设置为和xx1相同的形式
#     z = np.array(z).reshape(xx1.shape)
#     return xx1, xx2, z  # 返回xx1、xx2和z

# class SoftmaxRegression(nn.Module):
#     def __init__(self, features, classes):
#         super(SoftmaxRegression, self).__init__()
#         self.linear = nn.Linear(features, classes)
        
#     def forward(self, x):
#         return self.linear(x)

# if __name__ == "__main__":
#     green, blue, red = make_data(30)

#     features = 2
#     classes = 3
#     epochs = 10000
#     rate = 0.01

#     green = torch.FloatTensor(green)
#     blue = torch.FloatTensor(blue)
#     red = torch.FloatTensor(red)

#     train_data = torch.cat((green, blue, red), dim=0)
#     label = torch.LongTensor([0]*len(green) + [1]*len(blue) + [2]*len(red))
#     model = SoftmaxRegression(features, classes)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=rate)

#     for epoch in range(epochs):
#         output = model(train_data)
#         loss = criterion(output, label)

#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
        
#         if epoch % 1000 == 0:
#             # 每1000次迭代，打印一次当前的损失
#             print(f'{epoch} iterations : loss = {loss.item():.31f}')
            
#     # 调用决策边界函数
#     xx1, xx2, z = draw_decision_boundary(-4, 4, -4, 4, model)

#     # 绘制决策边界
#     plt.contourf(xx1, xx2, z, colors=['orange'])
#     # plt.show()

#     board = plt.figure()
#     axis = board.add_subplot(1, 1, 1)
#     axis.set(
#         xlim=[-4, 4],
#         ylim=[-4, 4],
#         title="Softmax regression",
#         xlabel='x1',
#         ylabel='x2'
#     )

#     plt.scatter(green[:, 0], green[:, 1], color='green')
#     plt.scatter(blue[:, 0], blue[:, 1], color='blue')
#     plt.scatter(red[:, 0], red[:, 1], color='red')
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

def make_data(num):
    """生成三类数据"""
    green = np.random.randn(num, 2) + np.array([0, -2])
    blue = np.random.randn(num, 2) + np.array([-2, 2])
    red = np.random.randn(num, 2) + np.array([2, 2])
    return green, blue, red

def draw_decision_boundary(minx1, maxx1, minx2, maxx2, model, device='cpu'):
    """
    绘制决策边界的高效版本
    Args:
        model: 训练好的模型
        device: 使用的设备（cpu或cuda）
    """
    # 生成网格点
    xx1, xx2 = np.meshgrid(np.linspace(minx1, maxx1, 300),
                           np.linspace(minx2, maxx2, 300))
    
    # 将所有网格点转换为张量
    grid_points = np.column_stack([xx1.ravel(), xx2.ravel()])
    grid_tensor = torch.FloatTensor(grid_points).to(device)
    
    # 批量预测
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        outputs = model(grid_tensor)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu().numpy()
    
    # 重塑预测结果
    z = predictions.reshape(xx1.shape)
    
    return xx1, xx2, z

class SoftmaxRegression(nn.Module):
    def __init__(self, features, classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(features, classes)
        
    def forward(self, x):
        return self.linear(x)

def train_model(model, train_data, label, epochs=10000, rate=0.01):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    
    losses = []
    for epoch in range(epochs):
        model.train()  # 设置为训练模式
        output = model(train_data)
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
            losses.append(loss.item())
    
    return losses

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 生成数据
    green, blue, red = make_data(30)
    
    # 转换数据为PyTorch张量
    green_tensor = torch.FloatTensor(green)
    blue_tensor = torch.FloatTensor(blue)
    red_tensor = torch.FloatTensor(red)
    
    # 合并训练数据和标签
    train_data = torch.cat((green_tensor, blue_tensor, red_tensor), dim=0)
    label = torch.LongTensor([0]*len(green) + [1]*len(blue) + [2]*len(red))
    
    # 创建模型
    features = 2
    classes = 3
    model = SoftmaxRegression(features, classes)
    
    # 训练模型
    print("开始训练模型...")
    losses = train_model(model, train_data, label, epochs=10000, rate=0.01)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制训练损失
    ax1.plot(range(0, 10000, 1000), losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # 绘制决策边界和数据点
    ax2.set_xlim([-4, 4])
    ax2.set_ylim([-4, 4])
    ax2.set_title("Softmax Regression Decision Boundary")
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    
    # 绘制决策边界（使用等高线填充）
    xx1, xx2, z = draw_decision_boundary(-4, 4, -4, 4, model)
    
    # 使用三个颜色分别代表三个类别
    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    ax2.contourf(xx1, xx2, z, cmap=cmap_light, alpha=0.3)
    
    # 绘制数据点
    ax2.scatter(green[:, 0], green[:, 1], color='green', edgecolors='k', 
                s=50, label='Class 0 (Green)', alpha=0.8)
    ax2.scatter(blue[:, 0], blue[:, 1], color='blue', edgecolors='k', 
                s=50, label='Class 1 (Blue)', alpha=0.8)
    ax2.scatter(red[:, 0], red[:, 1], color='red', edgecolors='k', 
                s=50, label='Class 2 (Red)', alpha=0.8)
    
    # 绘制决策边界线
    ax2.contour(xx1, xx2, z, colors='k', linewidths=0.5, alpha=0.5)
    
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 计算准确率
    model.eval()
    with torch.no_grad():
        outputs = model(train_data)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == label).sum().item() / len(label)
        print(f"训练准确率: {accuracy*100:.2f}%")
    
    # 可选：绘制更详细的决策边界图
    fig2, ax3 = plt.subplots(figsize=(8, 8))
    ax3.set_xlim([-4, 4])
    ax3.set_ylim([-4, 4])
    ax3.set_title("Softmax Regression Decision Regions")
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    
    # 使用更细的网格绘制决策边界
    xx1_detailed, xx2_detailed, z_detailed = draw_decision_boundary(-4, 4, -4, 4, model)
    
    # 绘制决策区域
    cmap_detailed = ListedColormap(['#FFCCCC', '#CCFFCC', '#CCCCFF'])
    ax3.contourf(xx1_detailed, xx2_detailed, z_detailed, 
                 cmap=cmap_detailed, alpha=0.3)
    
    # 绘制数据点
    ax3.scatter(green[:, 0], green[:, 1], color='green', edgecolors='k', 
                s=80, label='Class 0 (Green)')
    ax3.scatter(blue[:, 0], blue[:, 1], color='blue', edgecolors='k', 
                s=80, label='Class 1 (Blue)')
    ax3.scatter(red[:, 0], red[:, 1], color='red', edgecolors='k', 
                s=80, label='Class 2 (Red)')
    
    # 绘制决策边界线
    ax3.contour(xx1_detailed, xx2_detailed, z_detailed, 
                colors='k', linewidths=1, alpha=0.7)
    
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()