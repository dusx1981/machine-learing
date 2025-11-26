import torch
import matplotlib.pyplot as plt

def f(x):
    return x^2 + 2*x + 3

if __name__ == '__main__':
    # 生成自变量序列x，张量x，需要自动微分功能
    x = torch.linspace(-6.5, end=3.5, steps=1000, requires_grad=True)
    y_f = f(x)

    # 使用backward函数，计算f(x)关于x的梯度
    # 这样，所有的梯度值，就都会保存在x.grad中
    y_f.sum().backward()
    # 因为backward()函数只能对标量进行操作
    # 需要先使用y_f.sum，将y_f中的元素求和，将其转换为一个标量
    # 再在这个标量上调用backward，计算梯度
    
    # 将梯度值x.grad、函数值y_f、自变量x，从pytorch张量转换为numpy数组
    # 注意，转换前需要调用detach函数
    y_df =x.grad.detach().numpy()
    y_f = y_f.detach().numpy()
    x = x.detach().numpy()
    # detach方法会创建一个原张量的副本，该副本不会跟踪张量的梯度
    # 使用detach后，才能正常的将张量转换为numpy数组
    # 而不影响自动梯度的计算

    # 使用plot绘制图
    plt.plot(*args:x,y_f,label='f(x)= x*x + 3 * x + 2')
    plt.plot(*args:x,y_df,label="f'(x)= 2 * x + 3")
    plt.legend() # 对图像进行标记
    plt.grid(True)
    plt.show()