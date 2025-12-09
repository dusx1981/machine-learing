import torch
from torch import nn
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(12, 1)

    def forward(self, x):
        return self.layer(x)

if __name__ == '__main__':
    # 1. 准备数据和模型
    df = pd.read_excel('E:/projects/machine-learing/data/train.xlsx')

    feature_names = [col for col in df.columns if col not in ['price']]
    x = torch.Tensor(df[feature_names].values)
    y = torch.Tensor(df['price'].values).unsqueeze(1)

    model = LinearRegression()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(20000):
        # 2. 前向传播：计算预测和损失
        h = model(x)    #预测值
        loss = criterion(h, y)    #均方误差和

        # 3. 反向传播：计算梯度
        gradient = loss.backward()

        # 4. 梯度下降：优化器使用梯度更新参数
        optimizer.step()

        # 5. 重要！清空梯度，为下一次迭代准备
        optimizer.zero_grad()

        # 当迭代次数是1000的整数倍时，执行评估
        if (epoch + 1) % 1000 == 0:
            
            # 1. 打印当前的训练轮次和损失值
            print(f'After {epoch + 1} iterations, Train Loss: {loss.item():.3f}')
            
            # 2. 将模型的预测输出(h)和真实标签(y)从计算图中分离，并转换为NumPy数组
            h_np = h.detach().numpy()
            y_np = y.detach().numpy()
            
            # 3. 计算并打印三个回归评估指标
            mse = mean_squared_error(y_np, h_np)  # 均方误差，惩罚大误差
            mae = mean_absolute_error(y_np, h_np) # 平均绝对误差，直观解释
            r2 = r2_score(y_np, h_np)             # R²分数，表示模型解释的方差比例
            
            print(f'\tMean Squared Error: {mse:.3f}')
            print(f'\tMean Absolute Error: {mae:.3f}')
            print(f'\tR2 Score: {r2:.3f}')

    print(model.state_dict())
    torch.save(model.state_dict(), 'E:/projects/machine-learing/data/model.pth')

