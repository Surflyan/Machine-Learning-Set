# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
def getData(rank):
    X = np.arange(0, 1.0, 0.1)
    Y = [np.sin(x * 2 *np.pi) for x in X]
    trainY = np.array([np.random.normal(0,0.15) + y for y in Y])
    theta = np.array(np.random.normal(size=rank + 1))
    return X, trainY, theta

# 共轭梯度法
def CGradient(arrX, arrY, arrW, regular):
    W = np.mat(arrW.copy()).T
    Y = np.mat(arrY.copy()).T
    X = np.mat(np.zeros((arrX.size, arrW.size)))
    a = arrX.size
    b = arrY.size
    for i in range(a):
         for j in range(b):
            X[i, j] = np.power(arrX[i], j)
    E = np.eye(arrW.size)

    if (regular == False) :
        A = X.T* X
    else:
        A = X.T * X + E * regular
    #计算梯度
    g =  (np.dot(np.dot(X.T, X), W) - np.dot(X.T, Y))
    if (CalNorm(g) != 0):
        d = -g
        while (CalNorm(g) > 0.00002):
            step = (np.dot(-g.T, d) / np.dot(np.dot(d.T, A), d))[0, 0]
            W += step * d
            g = (np.dot(np.dot(X.T, X), W) - np.dot(X.T, Y))
            alpha = (np.dot(np.dot(d.T, A), g) / np.dot(np.dot(d.T, A), d))[0, 0]
            d = -g + alpha * d

    ret = np.array([W[i, 0] for i in range(arrW.size)])
    return ret

# 计算二范数**2
def CalNorm(X):
    norm = 0.0
    for i in range(X.size):
        norm += X[i, 0] ** 2
    return norm


def predict(testX, theta, rank):
    testVanderX = (np.array([x ** r for x in testX
                             for r in range(0, rank + 1, 1)])).reshape(len(testX), rank + 1)
    resultY = np.dot(testVanderX, theta)
    return resultY


if __name__ == "__main__":
    rank = 9

    X, trainY, theta = getData(rank)
    theta2 = theta
    theta = CGradient(X, trainY, theta,False)

    testX = np.linspace(0, 1)
    resultY = predict(testX, theta, rank)
    theta2 = CGradient(X, trainY, theta2, True)
    regularResultY = predict(testX,theta2,rank)
    plt.title("CG")
    plt.plot(X, trainY, 'ro')
    plt.plot(testX, np.sin(2 * np.pi * testX), 'r', label="TrueLine")
    plt.plot(testX, resultY, 'g', label="CG")
    plt.plot(testX, regularResultY, 'b', label="RegularCG")
    plt.legend()
    plt.show()

