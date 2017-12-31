# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
rank = 10
X = np.arange(0,1,0.025)
Y = [np.sin(2 * np.pi * x) for x in X]

# 添加高斯分布的噪声
YNoise = [np.random.normal(0,0.2)+y for y in Y]
matY = np.matrix(YNoise)

# 生成X的范德蒙德矩阵
matX = (np.array([x**r for x in X  for r in range(0,rank,1)])).reshape(len(X),rank)
matX = np.matrix(matX)

# 计算解析解
thetaMat = (matX.T*matX).I*matX.T*matY.T


lineX = np.linspace(0,0.9,50)
fitMatX = (np.array([x**r for x in lineX for r in range(0,rank,1)])).reshape(len(lineX),rank)
fitMatX = np.matrix(fitMatX)
result = fitMatX*thetaMat

resultLi = result.tolist()
resultList = [i[0] for i in resultLi]
print type(resultList)


plt.title("least square ")
plt.plot(X,YNoise,'ro')

a = np.linspace(0,0.9)
plt.plot(a,np.sin(2 * np.pi * a),'b',label = "true_line")
plt.plot(lineX, resultList,'g',label = "LS_fitting")
plt.legend()

#plt.savefig("least.png",dpi=100,figsize = (3,2))
plt.show()
