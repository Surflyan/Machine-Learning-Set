# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt

# test
def manualData():
    sampleSize = 20
    x1 = [i/2.0 for i in range(sampleSize)]
    #x2 = [1,3,2,4,7,6,9,12,10,10,3,5,14,6,8,7,11,8,9,12,10] # 改变4 -> 14,过拟合
    # 不满足条件独立
    # x2 = [i**3 -1 for i in x1]
    #x2 = [10* np.sin(i) for i in x1]
    x2 = [2*i+1 for i in x1]
    dataMat = [[1.0,x1[i],x2[i]] for i in range(sampleSize) ]
    labelMat = []
    [labelMat.append([1]) for i in range(sampleSize/2)]
    [labelMat.append([0]) for i in range(sampleSize/2)]
    dataMat = np.matrix(dataMat)
    labelMat = np.matrix(labelMat)

    theta = np.zeros([dataMat.shape[1],1])
    theta =np.matrix(theta)

    return dataMat, labelMat, theta


# 加载数据，返回以矩阵形式特征表、标签表，初始theta 矩阵
def loadDataSet(dataFileName,labelFileName):
    dataMat = []
    labelMat = []
    fr = open(dataFileName)
    labelFile = open(labelFileName)
    for line in fr:
        lineArr = line.strip().split(",")
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    for line in labelFile:
        lineArr = line.strip()
        labelMat.append([int(lineArr)])

    # 转为 matrix 类型返回
    dataMat = np.matrix(dataMat)
    labelMat = np.matrix(labelMat)

    theta = np.zeros([dataMat.shape[1],1])
    theta =np.matrix(theta)
    return dataMat, labelMat, theta

def sigmoid(x):
    g = 1.0 / (1 + np.exp(-x))
    return g

def gradientAscend(theta,X,Y,maxIteration,alpha) :
    """
    梯度下降法
    :param theta矩阵
    :param X特征矩阵
    :param Y矩阵
    :param 最大迭代次数
    :param 惩罚系数
    :return 优化解theta
    """
    m = X.shape[0]
    cost = []  # 记录每次迭代所有样本cost值之和
    while(maxIteration):
        grad = np.matrix(np.zeros(3)).T
        costValue = 0
        for i in range(m) :
            thetaTX = np.dot(theta.T,X[i].T)
            thetaTX = thetaTX[0,0]           # 实数
            hypothesis = sigmoid(thetaTX)
            costValue += -int(Y[i]) * np.log(hypothesis) - (1 - int(Y[i])) * np.log(1 - hypothesis)  # cost function
            error = hypothesis - int(Y[i])
            grad = grad + (error/m * X[i].T)

        # 计算|| Wk+1 - Wk||，通过阈值停止
        norm = np.dot(grad.T,grad)
        if (norm < 0.0005) :
            break
        theta = theta - alpha * grad
        maxIteration -= 1
        cost.append(costValue)
    return theta, cost

# 梯度法，使用向量形式（
# 相比手动计算梯度叠加，直接使用向量形式，运算速度明显更快
def gradientAscendMatrix(theta,X,Y,maxIteration,alpha,lamd = 0) :
    """
    梯度下降法
    :param theta矩阵
    :param X特征矩阵
    :param Y矩阵
    :param 最大迭代次数
    :param 下降步长
    :return 优化解theta
    """
    m = X.shape[0]
    normWList = []    # 存放每次迭代后theta 的二范数
    while(maxIteration):
        thetaTX = np.dot(X,theta)
        hypothesis = sigmoid(thetaTX)
        error = hypothesis - Y
        grad = np.dot(X.T,error/m)

        # 计算|| Wk+1 - Wk||，根据阈值判断是否停止迭代
        updateW = np.dot(grad.T, grad)
        if (updateW < 0.0005):
            break

        # 正则项添加
        if (lamd == 0) :
            theta = theta - alpha * grad
        else :
            theta = theta - alpha * (grad + lamd * theta)

        # 计算||W||二范数，考虑是否过拟合
        normW = np.dot(theta.T, theta)[0,0]
        normWList.append(normW)
        maxIteration -= 1
    return theta, normWList

def newtonMethod(theta,X, Y, maxIteration,lamd = 0) :
    """
    牛顿法进行参数估计
    :param theta矩阵
    :param X特征矩阵
    :param Y矩阵
    :param 最大迭代次数
    :lamd 惩罚系数
    :return theta优化解
    """
    m = X.shape[0]
    cost = []            # 存放每次迭代所有样本数据的cost值
    normWList = []        # 存放每次迭代后theta 的二范数
    while(maxIteration) :
        grad = np.matrix(np.zeros(3)).T
        Hessian = np.matrix(np.zeros((3,3)))
        costValue = 0          # 计算error
        for i in range(m):
            thetaTX = np.dot(theta.T,X[i].T)
            thetaTX = thetaTX[0,0]           # 实数
            hypothesis = sigmoid(thetaTX)
            costValue += -int(Y[i])* np.log(hypothesis) -(1 - int(Y[i]))* np.log(1 - hypothesis)   # cost function
            error = hypothesis - int(Y[i])
            grad += error/m * X[i].T           #delta J
            Hessian += 1.0/m * hypothesis * (1 - hypothesis) * np.dot(X[i].T, X[i])     # ! 1/m  1.0/m
        cost.append(costValue)             # 记录cost

        # 通过阈值停止迭代
        updateW = np.dot(grad.T, grad)
        if (updateW < 0.0005):
            break

        # 正则项添加
        try:
            if(lamd == 0) :             # 不加正则项
                theta = theta - (np.linalg.inv(Hessian) * grad )
            else:                      #  加正则项
                regHessian = Hessian + lamd * np.eye(len(theta))
                regGrad = grad + lamd * theta
                theta = theta - (np.linalg.inv(regHessian) * regGrad)
        except:
                maxIteration -=1
                continue

        # 计算||W||二范数，考虑是否过拟合
        normW = np.dot(theta.T, theta)[0,0]
        normWList.append(normW)


        maxIteration -= 1
    return theta, cost, normWList

# 图形化展示
def plotFigure(dataMat,labelMat,resTheta,cost=[],normWList = [], method = "gradient-descent"):

    dataNum = dataMat.shape[0]
    # 转换为np.array 形式
    dataArray = np.array(dataMat)
    #labelArray = np.array(labelMat)

    thetaArray = np.array(resTheta)

    #xClass1 yClass1 存放类别为 1 的点
    xClass1 = []
    xClass2 = []
    yClass1 = []
    yClass2 = []
    for i in range(dataNum):
        if(labelMat[i] == 1):
            xClass1.append(dataArray[i][1])
            yClass1.append(dataArray[i][2])
        else:
            xClass2.append(dataArray[i][1])
            yClass2.append(dataArray[i][2])

    x1 = np.max(xClass1)
    x2 = np.max(xClass2)
    maxX = (x1 if x1 > x2 else x2)

    # 激活第一个subplot, 划分图
    plt.subplot(1,2,1)
    plt.plot(xClass1,yClass1,'ro')
    plt.plot(xClass2,yClass2,'b^')

    xline = np.arange(0, maxX, 1)
    yline = (-thetaArray[0] - thetaArray[1] * xline) / thetaArray[2]
    plt.title(method)
    plt.plot(xline,yline,'g')

    # 激活第二个子图， cost 调整图
    if(len(cost) != 0):
        plt.subplot(1,2,2)
        x = [i for i in range(len(cost))]
        #plt.plot(x,cost,'gx')
        plt.plot(x,cost, 'g--')
        plt.title("cost")

    # 展示||w|| 变化
    if(len(normWList) != 0):
        plt.figure()
        x = [i for i in range(len(normWList))]

        plt.plot(x,normWList,'c--')
        plt.title("2-norm-W")
    plt.show()

# 采用留一交叉验证
def calAccuracy(dataMat,labelMat,theta):
    count = 0
    newtonIteration = 10
    lamd = 0.0005
    for i in range(len(dataMat)):
        trainDataMat = np.concatenate((dataMat[:i],dataMat[i+1:]),axis = 0)
        trainLabelMat = np.concatenate((labelMat[:i], labelMat[i + 1:]), axis=0)
        resTheta,cost,normWList  = newtonMethod(theta,trainDataMat,trainLabelMat,newtonIteration, lamd )
        wx = np.dot(dataMat[i],resTheta)[0,0]
        probability = np.exp(wx) / (1.0 + np.exp(wx))
        if(int(probability + 0.5) == int(labelMat[i])):
            count += 1

    return float(count)/len(dataMat)


if __name__=="__main__":
    # 加载数据
    dataMat, labelMat, theta = loadDataSet("ex4x.dat","ex4y.dat")
    # 手工数据
    #dataMat, labelMat, theta = manualData()
    maxIteration = 300000          # 控制梯度法最大迭代次数
    alpha = 0.001                  # 下降步长
    lamd = 0                   # 惩罚项系数，默认为零，及不加惩罚项
    cost = []                      # cost 每次迭代所有样本cost 值之和
    normWList = []

    # 梯度法
    #resTheta, cost= gradientAscend(theta,dataMat,labelMat,maxIteration,alpha)
    # 向量形式
    #resTheta, normWList = gradientAscendMatrix(theta,dataMat,labelMat,maxIteration,alpha,lamd)
    # 牛顿法
    newtonIteration = 10
    resTheta, cost, normWList = newtonMethod(theta,dataMat,labelMat,newtonIteration,lamd)
    plotFigure(dataMat,labelMat,resTheta,cost,normWList, method = "newton-method")


    accuracy = calAccuracy(dataMat,labelMat,theta)
    print "The accuracy is ",accuracy
