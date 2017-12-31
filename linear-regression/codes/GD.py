# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt

# 生成训练数据
def getData(rank):
    X = np.arange(0,1,0.1)
    Y = [np.sin(2* np.pi*x) for x in X]
    trainX = (np.array([x**r for x in X
                       for r in range(0,rank+1,1)])).reshape(len(X),rank+1)
    trainY = [np.random.normal(0,0.15) + y for y in Y]

    theta = np.random.normal(size=rank+1)
    return trainX,trainY,theta,X

# 迭代跟新theta
def batchGradientDescent(x,y,theta,alpha,maxIterations):
    xt = x.transpose()
    for i in range(maxIterations):
        hypothesis = np.dot(x,theta)
        loss = hypothesis - y
        gradient = np.dot(xt,loss)
        theta = theta - alpha * gradient
    return theta

# 带有惩罚项的theta 更新
def RegularzedGradientDescent(x,y,theta,alpha,maxIterations,lamb):
    xt = x.transpose()
    for i in range(maxIterations):
        hypothesis = np.dot(x,theta)
        loss = hypothesis - y
        gradient = np.dot(xt,loss)
        theta = theta * (1 - alpha * lamb) - alpha * gradient
    return theta

# 预测测试数据
def predict(testX,theta,rank):
    testVanderX= (np.array([x**r for x in testX
                       for r in range(0,rank+1,1)])).reshape(len(testX),rank+1)

    resultY = np.dot(testVanderX,theta)
    return resultY

def pipLine(rank):
    alpha = 0.003
    maxIterations = 200000
    lamb = 0.002
    trainX, trainY, theta, X = getData(rank)
    theta2 = theta
    theta = batchGradientDescent(trainX, trainY, theta, alpha, maxIterations)

    testX = np.linspace(0, 0.95)
    resultY = predict(testX, theta, rank)

    theta2 = RegularzedGradientDescent(trainX, trainY, theta2, alpha, maxIterations, lamb)
    regularResultY = predict(testX, theta2, rank)

    plt.plot(X, trainY, 'ro')
    plt.plot(testX, np.sin(2 * np.pi * testX), 'r', label="TrueLine")

    plt.plot(testX, resultY, 'g', label="GradientDescent")
    plt.plot(testX, regularResultY, 'b', label="RegularGradientDescent")
    plt.legend()
    plt.show()
    return

if __name__=="__main__":
    for rank in range(0,10):
        pipLine(rank)
