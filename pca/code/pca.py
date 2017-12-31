# --coding: utf-8 --

"""
@author: Surflyan
@date: 2017-11-15
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
import copy
from sklearn.decomposition import PCA
import seaborn as sns

def getManualData(dataSize):

    # 二维数据生成
    if(dataSize[1] == 2):
        MySample = np.zeros(dataShape)
        MySample[:,0] = np.random.random((100))*10
        for i in range(100):
            MySample[i,1] = MySample[i,0] + np.random.random()*3

    #三维数据生成
    else:
        MySample = np.random.random((100,3))*10
        for i in range(100):
            MySample[i,1] = MySample[i,1] * np.random.random()

    return MySample


def procHomogenization(data):
    """
    数据中心化和方差归一化
    :param 样本数据，以维度为列，样本为行
    """
    dimension ,  = data[0].shape;
    meanData = np.zeros(data.shape)
    for i in range (dimension):
        mean = np.mean(data[:,i]) 
        std = np.std(data[:,i])
        meanData[:,i] = (data[:,i] - mean) / std
    return meanData


def calCovarianceMatrix(data):
    m = data.shape[0];
    sigma = np.dot(data.transpose(), data)

    return sigma


def getEigenVector(sigma):
    """
    计算特征向量阵
    :return 按特征值降序排列的特征向量矩阵
    """
    # U, S, V = np.linalg.svd(sigma)
    dimeFeatures = sigma.shape[0]
    eigVal, eigVec = np.linalg.eig(sigma)
    eigPairs = [(np.abs(eigVal[i]), eigVec[:,i]) for i in range(dimeFeatures)]
    eigPairs.sort(reverse=True)
    eigenVector =np.array([eigItem[1] for eigItem in eigPairs])

    pca = PCA(n_components = 100).fit(imgSet)
    var = pca.explained_variance_
    cmap = sns.color_palette()
    plt.bar(np.arange(1, len(var) + 1), var / np.sum(var), align="center", color=cmap[0])
    plt.step(np.arange(1, len(var) + 1), np.cumsum(var) / np.sum(var), where="mid", color=cmap[1])
    plt.show()
    return eigenVector



def getUReduce(U, k):
    """
    取得k维度得投影矩阵
    :param U 所有维度的降序特征向量
    """
    UReduece = U[:k].transpose()
    return UReduece



def MyPCA(data, k) :
    """
    定义自己PCA
    :param data: 样本数据集，（n*m) n 个样本，m 维特征
    :param k:  降维后的维度
    :return res: 降维后的矩阵
    """

    meanSample = procHomogenization(data)
    sigma = calCovarianceMatrix(meanSample)
    eigenVector = getEigenVector(sigma)
    UReduce = getUReduce(eigenVector, k)
    res = np.dot(meanSample, UReduce)

    return res, UReduce

def regainData(transData,UReduce) :

    newData = np.dot(transData,UReduce.transpose())
    return newData

def drawPic(data3d, data2d) :
    """
     3D 散点图
     :param data3d(x，3)
     :param data2d(x,2)
    """
    X3d = data3d[:,0]
    Y3d = data3d[:,1]
    Z3d = data3d[:,2]

    X2d = data2d[:,0]
    Y2d = data2d[:,1]
    fig3D= plt.figure()
    ax = fig3D.add_subplot(121 ,projection='3d')
    ax.plot(X3d,Y3d,Z3d,'r+')

    ax2d = fig3D.add_subplot(122)
    Y2d[:1] = 2
    ax2d.plot(X2d,Y2d,'g+')

    plt.show()



def getFaceData(PIC_PATH) :
    dataSet = []
    imgArray = []
    for i in range(1,41):
        for j in range(1,11):
            path = PIC_PATH + "\\s" + str(i) + "\\" + str(j) + ".pgm"
            img = cv2.imread(path)
            imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            imgArray.append(imgGray)
            x, y = imgGray.shape
            imgRow = imgGray.reshape(x * y)
            dataSet.append(imgRow)
    # plt.gray()
    #     # 以下选前面7个特征脸，按顺序分别显示到其余7个格子
    # for i in range(12):
    #     plt.subplot(3, 4, i + 1)
    #     plt.imshow(imgArray[i*10])
    # plt.title("origin face")
    # plt.show()

    return np.array(dataSet)





def pipLine(dataShape,k):

    MySample = getManualData(dataShape)
    meanData = procHomogenization(MySample)
    res ,UReduce = MyPCA(MySample,k)
    print ("res",res)
    drawPic(meanData, res)

def calcPSNR(img1,img2) :
    err = np.sum((img1.astype("float") - img2.astype("float"))**2)
    err /= float(img1.shape[0] * img1.shape[1])

    psnr = 10 * np.log((255**2)/err)
    return psnr

def facePipLine(dataSet,k):
    res, UReduce = MyPCA(dataSet,k)

    newData = regainData(res,UReduce)

    img = []
    for i in range(400):
        img.append(newData[:,i].reshape((112,92)))
    plt.gray()
        # 以下选前面7个特征脸，按顺序分别显示到其余7个格子

    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.55)
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.imshow(img[i*10])
        psnr = calcPSNR(img[i*10],dataSet[:,i*10].reshape(112,92))
        plt.title("psnr: %.2f" %psnr)

    plt.show()




if __name__=="__main__":
    #dataShape = (100,2)
    #pipLine(dataShape,1)
    #imgGray = getLenna()

    dataSet = getFaceData("C:\\Users\\Surflyan\\Desktop\\ML\\Lab3-PCA\\dataSet")
    print dataSet.shape

    print("Total dataset size:")
    print("n_samples: %d" % dataSet.shape[0])
    print("n_features: %d" % dataSet.shape[1])
    facePipLine(dataSet.T,80)







