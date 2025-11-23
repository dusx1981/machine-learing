# 梯度下降算法求 （x^2 + y^2）的极值

x = 2.1
y = 1.1
n = 100
alpha = 0.05 #学习速率，迭代速率

for i in range (1, n+1):
    gx = 2 * x
    gy = 2 * y

    x = x - alpha * gx
    y = y - alpha * gy

    print(
            f'After {i}, '
            f'x = {x: .3f}, '
            f'y = {y: .3f}, '
            f'f(x, y) = {x**2 + y**2: .3f}'
        )