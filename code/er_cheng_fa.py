import numpy

def normal_equation_array(X, y):
    XT = numpy.transpose(X)
    XTX = numpy.dot(XT, X)
    det = numpy.linalg.det(XTX)

    if det == 0:
        return "error"

    XTX_inv = numpy.linalg.inv(XTX)
    XTy = numpy.dot(XT, y)
    
    theta = numpy.dot(XTX_inv, XTy)
    return theta

def normal_equation_matrix(X, y):
    X = numpy.asmatrix(X)
    y = numpy.asmatrix(y)

    XTX = X.T * X
    det = numpy.linalg.det(XTX)
    if det == 0:
        return "error"

    theta = XTX.I * X.T * y
    return theta

# --------------------------------------------------------------

if __name__ == '__main__':
    # 数据准备
    X = numpy.array([
        [1, 96.79, 2, 1, 2],
        [1, 110.39, 3, 1, 0],
        [1, 70.25, 1, 0, 2],
        [1, 99.96, 2, 1, 1],
        [1, 118.15, 3, 1, 0],
        [1, 115.08, 3, 1, 21]
    ])

    y = numpy.array([[287], [343], [199], [298], [340], [350]])

    # 求解参数
    theta_array = normal_equation_array(X, y)
    theta_matrix = normal_equation_matrix(X, y)

    print("Array method theta:")
    print(theta_array)
    print("\nMatrix method theta:")
    print(theta_matrix)

    # 计算损失
    m = len(y)
    y_pred = numpy.dot(X, theta_array)
    error = y - y_pred
    costJ = numpy.dot(error.T, error) / (2 * m)
    print(f"\nCost J is {costJ[0][0]:.6f}")

    # 预测新样本
    test1 = numpy.array([1, 112, 3, 1, 0])
    test2 = numpy.array([1, 110, 3, 1, 1])

    pred1 = numpy.dot(test1, theta_array)
    pred2 = numpy.dot(test2, theta_array)

    print(f"\ntest1 = {pred1[0]:.3f}")
    print(f"test2 = {pred2[0]:.3f}")

