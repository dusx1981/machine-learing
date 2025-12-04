# 均方差代价函数求偏导数，求最小值

import numpy as np
import matplotlib.pyplot as plt

def gradient_theta0(x, y, theta0, theta1):
    m = len(x)
    h = theta0 + x * theta1

    return np.sum(h-y)/m

def gradient_theta1(x, y, theta0, theta1):
    m = len(x)
    h = theta0 + x * theta1

    return np.sum((h-y) * x)/m

def costJ(x, y, theta0, theta1):
    m = len(x)
    h = theta0 + x * theta1

    return np.sum((h-y)**2)/(2*m)

def gradient_des(x, y, a, n):
    theta0 = 0
    theta1 = 0
    for i in range (1, n+1):
        g0 = gradient_theta0(x, y, theta0, theta1)
        g1 = gradient_theta1(x, y, theta0, theta1)

        theta0 -= a * g0
        theta1 -= a * g1

    return theta0, theta1

if __name__ == '__main__':
    x = np.array([50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0])
    y = np.array([280.0, 305.0, 350.0, 425.0, 480.0, 500.0, 560.0, 630.0])
    
    # 调整学习率和迭代次数
    a = 0.0001  # 更小的学习率
    n = 100    # 更多迭代次数
    theta0, theta1 = gradient_des(x, y, a, n)
    cost = costJ(x, y, theta0, theta1)
    
    print("After %d iterations, the cost is %lf" % (n, cost))
    print("theta0 = %lf, theta1 = %lf" % (theta0, theta1))
    print("predict(112) = %lf" % (theta0 + theta1 * 112))
    print("predict(110) = %lf" % (theta0 + theta1 * 110))