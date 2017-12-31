# -*- coding: utf-8 -*-
"""
@author: Surflyan
@date: 2017-11-21
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from numpy.linalg import cholesky


def multiPDF(x, mu, sigma):
    """
    多元高斯模型概率密度计算
    """
    size = len(x)
    if size ==len(mu) and (size, size) == sigma.shape:
        detSigma = np.linalg.det(sigma)
        try :
            div = 1.0 / (np.math.pow(2 * np.pi, len(x) / 2.0) * np.math.pow(detSigma, 1.0 / 2))
        except:
            print ("except",detSigma)

        x_mu = np.matrix(x - mu)
        invSigma = np.linalg.inv(sigma)
        numerator = np.math.pow(np.math.e, -0.5 * (x_mu.T) * invSigma * x_mu)
        result = numerator * div
        return result
    else:
        return -1

def logLikelihood(dataSet, Alpha, Mu, Sigma):
    K = len(Alpha)
    N, M = np.shape(dataSet)
    P = np.zeros([N, K])
    for k in range(K):
        for i in range(N):
            P[i, k] = multiPDF(dataSet[i, :][None].T, Mu[:, k][None].T, Sigma[:, :, k])
    result = np.sum(np.log(np.dot(P,Alpha)))
    return result


def initEM(dataSet, K):
    """
    N: 样本数
    M: 样本维度
    K: 混合模型个数，或者聚类个数
    计算每个样本x 由个混合成分生成的后验概率
    W[i,j] 代表i 由第j个模型生成的概率
    初始化各项参数，按照原始数据的下标等分到不同的类别中，或者指定到某一成分生成的，利用Mximization 产生参数
    """

    N, M = dataSet.shape
    W = np.zeros([N, K])
    numPerK = N / K
    for k in range(K):
        start = int(np.floor(k * numPerK))
        end = int(np.floor((k + 1) * numPerK))
        W[start: end, k] = 1

    Alpha, Mu, Sigma = Mstep(dataSet, W)
    return W, Alpha, Mu, Sigma


def Mstep(dataSet, W):

    # 每一列代表一个聚类/成分，
    N, M = dataSet.shape
    K = W.shape[1]

    # 更新Alpha 混合系数
    N_k = np.sum(W, 0)           # 每个成分所产生的样本概率和
    Alpha = N_k / np.sum(N_k)    # /总样本

    # 更新mu (M * K) 新均值向量
    Mu = np.dot(np.dot(dataSet.T,W),np.diag(np.reciprocal(N_k)))

    # 更新协方差矩阵（M*M*K ）（包含k个协方差矩阵)
    Sigma = np.zeros([M, M, K])
    for k in range(K):
        dataMeanSub = dataSet.T - np.dot(Mu[:,k][None].T,np.ones([1,N]))
        Sigma[:, :, k] = (dataMeanSub.dot(np.diag(W[:, k])).dot(dataMeanSub.T)) / N_k[k]

    return Alpha, Mu, Sigma


def Estep(dataSet, Alpha, Mu, Sigma):
    """
    计算每个样本x 由个混合成分生成的后验概率
    W[i,j] 代表i 由第j个模型生成的概率
    """

    N = dataSet.shape[0]
    K = len(Alpha)
    W = np.zeros([N,K])
    for k in range(K):
        for i in range(N):
            W[i, k] = Alpha[k] * multiPDF(dataSet[i, :][None].T, Mu[:, k][None].T, Sigma[:, :, k])
    # 将每一行也就是一个样本出现的概率归一
    W = W * np.reciprocal(np.sum(W, 1)[None].T)
    return W


def EM(data,K):
    W, Alpha, Mu, Sigma = initEM(data, K)
    iter = 0
    prevLL = -999999
    valLogLikelihood = []
    while (True):

        W = Estep(data, Alpha, Mu, Sigma)
        Alpha, Mu, Sigma = Mstep(data, W)
        valLL = logLikelihood(data, Alpha, Mu, Sigma)
        iter = iter + 1
        if iter % 1 == 0 :
            valLogLikelihood.append(valLL)

        if (iter > 200 or abs(valLL - prevLL) < 0.000000000001):
            break
        print valLL
        prevLL = valLL
    drawLikeliHood(valLogLikelihood)
    return Alpha,Mu,Sigma

def getData() :

    iris = datasets.load_iris()

    # 这里没有采用交叉验证，只是将训练集和测试集按3：1 分开
    skf = StratifiedKFold(iris.target, n_folds=4)
    # Only take the first fold.
    train_index, test_index = next(iter(skf))

    X_train = iris.data[train_index]
    Y_train = iris.target[train_index]
    X_test = iris.data[test_index]
    Y_test = iris.target[test_index]

    return X_train,Y_train,X_test,Y_test


def getManData(m) :
    """
    获取二维数据
    :param m: 样本个数
    :return:
    """
    mu1 = np.array([1,5])
    mu2 = np.array([5,3])
    sigma1 = np.array([[1,0.5],[1.5,3]])
    sigma2 = np.array([[1,0.9],[1.5,5]])
    R = cholesky(sigma1)
    data1 = np.dot(np.random.randn(int(0.75*m), 2),R) + mu1

    R2 = cholesky(sigma2)
    data2 = np.dot(np.random.randn(int(0.25*m), 2),R2) + mu2
    data = np.concatenate((data2,data1))

    return data


def drawPic(data):
    """
    二维数据分布图
    """
    plt.plot(data[:,0],data[:,1],'+')
    plt.title('Sample distribution')
    plt.show()

def drawLikeliHood(data):
    iterNum = len(data)
    X = [ i for i in range(iterNum)]
    plt.plot(X,data,'-')
    plt.show()

def clusterPredict(data,Alpha,Mu,Sigma):
    #计算每个样本所属类别概率
    # W[N*K]
    W = Estep(data,Alpha,Mu,Sigma)
    predictFlag = []
    N = W.shape[0]
    for i in range(N):
        predictFlag.append(np.argmax(W[i]))
    return np.array(predictFlag)

def drawIris(trainAccuracy,testAccuracy,X_test,Y_test):
    iris = datasets.load_iris()
    h = plt.subplot(111)
    for n, color in enumerate('rgb'):
        data = iris.data[iris.target == n]
        plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
                    label=iris.target_names[n])
        # Plot the test data with crosses
    for n, color in enumerate('rgb'):
        data = X_test[Y_test == n]
        plt.plot(data[:, 0], data[:, 1], 'x', color=color)

    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % trainAccuracy,
            transform=h.transAxes)
    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % testAccuracy,
            transform=h.transAxes)
    plt.show()

def pipLine():

    X_train ,Y_train,X_test,Y_test = getData()
    #trainSet = getManData(400)
    #trainSet = getOneFeacData(400)
    #drawPic(trainSet)
    K = 3
    Alpha,Mu,Sigma = EM(X_train,K)
    predict = clusterPredict(X_train,Alpha,Mu,Sigma)
    train_accuracy = np.mean(predict.ravel() == Y_train.ravel()) * 100
    print ("train_accuracy : ",train_accuracy)

    predictTest = clusterPredict(X_test,Alpha,Mu,Sigma)
    test_accuracy = np.mean(predictTest.ravel() == Y_test.ravel()) * 100
    print ("test_accuracy : ", test_accuracy)

    drawIris(train_accuracy,test_accuracy,X_test,Y_test)



if __name__ == '__main__':
    pipLine()
