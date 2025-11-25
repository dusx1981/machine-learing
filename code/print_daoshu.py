import numpy as np
import matplotlib.pyplot as plt

# 定义函数 f(x) = x² + 3x + 2
def f(x):
    return x**2 + 3*x + 2

# 定义导数 df(x) = 2x + 3
def df(x):
    return 2*x + 3

if __name__ == '__main__':
    x = np.linspace(-6.5, 3.5, 1000)
    y_f = f(x)
    y_df = df(x)

    plt.plot(x, y_f, label='x*x + 3*x +2')
    plt.plot(x, y_df, label='2*x + 3')

    plt.legend()
    plt.grid(True)
    plt.show()