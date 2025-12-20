import math
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy

def sigmod(z):
    return 1.0 / (1 + math.exp(-z))

def hypethesis(theta, x, n):
    h = 0.0
    for i in range(0, n+1):
        h += theta[i]*x[i]

    return sigmod(h)

def gradient_thetaj(x, y, theta, m, n, j):
    sum = 0
    for i in range (0, m):
        h = hypethesis(theta, x[i], n)
        sum += (h - y[i]) * x[i][j]

    return sum / m

def gradient_descent(x, y, alpha, m, n, it):
    theta = [0] * (int(n) + 1)

    for i in range (0, it):
        temp = [0] * (int(n) + 1)
        for j in range (0, n+1):
            temp[j] = theta[j] - alpha * gradient_thetaj(x, y, theta, m, n, j)
        for j in range (0, n+1):
            theta[j] = temp[j]

    return theta

def costJ(x, y, theta, m, n):
    sum = 0
    for i in range (0, m):
        h = hypethesis(theta, x[i], n)
        sum += -y[i]*math.log(h) - (1-y[i]) * math.log(1-h)
    return sum / m

if __name__ == '__main__': 
    # 使用make_blobs随机的生成正例和负例，其中n_samples代表样本数量，设置为50
    # centers代表聚类中心点的个数，可以理解为类别标签的数量，设置为2
    # random_state是随机种子，将其固定为0，这样每次运行就生成相同的数据
    # cluster_std是每个类别中样本的方差，方差越大说明样本越离散，这里设置为0.5
    X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.5)
    posx1, posx2 = X[y == 1][:, 0], X[y == 1][:, 1]
    negx1, negx2 = X[y == 0][:, 0], X[y == 0][:, 1]

    # 创建画板对象，并设置坐标轴
    board = plt.figure()
    axis = board.add_subplot(1, 1, 1)
    # 横轴和纵轴分别对应x1和x2两个特征，长度从-1到6
    axis.set(xlim=[-1, 6],
        ylim=[-1, 6],
        title='Logistic Regression',
        xlabel='x1',
        ylabel='x2')

    # 画出正例和负例
    # 其中正例使用蓝色圆圈表示，负例使用红色叉子表示
    plt.scatter(posx1, posx2, color='blue', marker='o')
    plt.scatter(negx1, negx2, color='red', marker='x')
    # 完成样本的绘制后，进行模型迭代

    m = len(X)  # 保存样本个数  
    n = 2  # 保存特征个数  
    alpha = 0.001  # 迭代速率  
    iterate = 20000  # 迭代次数  

    # 将生成的特征向量x的添加一列1，作为偏移特征  
    X = numpy.insert(X, 0, values=[1] * m, axis=1).tolist()  
    y = y.tolist()

    # 调用梯度下降算法，迭代出决策边界，并计算代价值  
    theta = gradient_descent(X, y, alpha, m, n, iterate)  
    costJ = costJ(X, y, theta, m, n)  
    for i in range(0, len(theta)):  
        print("theta[%d] = %lf" % (i, theta[i]))  
    print("Cost J is %lf" % (costJ))

    # 根据迭代出的模型参数θ，绘制分类的决策边界
    # 其中θ1、θ2、θ0，分别对应直线中的w1、w2和b参数

    w1 = theta[1]
    w2 = theta[2]
    b = theta[0]

    # 使用linspace在-1到5之间构建间隔相同的100个点
    x = numpy.linspace(-1, 6, 100)
    # 将这100个点，代入到决策边界，计算纵坐标
    d = - (w1 * x + b) * 1.0 / w2
    # 绘制分类的决策边界
    plt.plot(x, d)
    plt.show()
