#coding:--utf-8--

#Author: Surflyan
#Date: 2017-12-17
import cv2
from skimage import data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn.decomposition import PCA
import seaborn as sns

def getData(path):
    train = pd.read_csv(path+'train.csv')
    test = pd.read_csv(path+'test.csv')

    # x, y = train.shape
    # for i in range(y-20):
    #     count = 0
    #     for j in range(x):
    #         if train.iloc[j,i] == 0:
    #             count+=1
    #     if count > 0.75*x:
    #         train.drop(train.columns[[i]],axis = 1,inplace = True)
    #         test.drop(test.columns[[i]],axis = 1,inplace = True)

    train_id = train.iloc[:,0]
    test_id = test.iloc[:,0]
    train_species = train.iloc[:,1]
    train = train.drop('species', 1)

    return train,test,train_id,test_id,train_species

def getFeatureFromTrainPic(path,train_id,train):
    areaList = []
    perimeterList = []       #周长
    ratioList = []           #比率
    extentList = []          #轮廓面积与边界矩形面积的比
    diameterList = []        #相等圆面积的直径
    directionList = []       #方向，竖直还是水平
    cxList = []              # 重心
    cyList = []
    xyList = []              #乘积
    xdyList = []             #差
    xayList = []             #和
    xiyList = []

    print path
    #从图片中生成Area，perimeter，diameter
    for val in train_id:
        num = val
        num = str(num)
        val = num + '.jpg'
        completePath = path + 'images\\' + val
        #从文件读图片
        img = cv2.imread(completePath,0)
        ret,thresh = cv2.threshold(img,150,255,0)
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cnt = contours[0]

        # 计算重心
        M = cv2.moments(img)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        cxList.append(cx)
        cyList.append(cy)
        xyList.append(cx*cy)
        xdyList.append(cx-cy)
        xayList.append(cx+cy)
        xiyList.append(float(cx)/cy)



        #获得 Area
        area = cv2.contourArea(cnt)
        area = round(area,2)

        areaList.append(area)

        perimeter = cv2.arcLength(cnt,True)
        perimeter = round(perimeter, 2)
        perimeterList.append(perimeter)

        x,y,w,h = cv2.boundingRect(cnt)
        ratio = float(w)/h
        ratioList.append(ratio)

        x,y,w,h = cv2.boundingRect(cnt)
        rect_area = w * h
        extent = float(area)/rect_area
        extent = round(extent, 2)
        extentList.append(extent)

        if(w > h):
            direction = 0
        else :
            direction = 1
        directionList.append(direction)

        equi_diameter = np.sqrt(4*area/np.pi)
        diameterList.append(equi_diameter)

    #将新特征添加到训练集中
    Cx_train = pd.DataFrame(cxList)
    Cx_train.columns = ['Cx']
    Cy_train = pd.DataFrame(cyList)
    Cy_train.columns = ['Cy']
    xy_train = pd.DataFrame(xyList)
    xy_train.columns = ['XY']
    xdy_train = pd.DataFrame(xdyList)
    xdy_train.columns = ['XDY']
    xay_train = pd.DataFrame(xayList)
    xay_train.columns = ['XAY']
    xiy_train = pd.DataFrame(xiyList)
    xiy_train.columns = ['XIY']

    Perimeter_train  = pd.DataFrame(perimeterList)
    Perimeter_train.columns = ['Perimeter']
    Area_train  = pd.DataFrame(areaList)
    Area_train.columns = ['Area']
    Ratio_train  = pd.DataFrame(ratioList)
    Ratio_train.columns = ['Ratio']
    Extent_train  = pd.DataFrame(extentList)
    Extent_train.columns = ['Extent']
    Dia_train  = pd.DataFrame(diameterList)
    Dia_train.columns = ['Diameter']
    Direction_train = pd.DataFrame(directionList)
    Direction_train.columns = ['Direction']

    #去掉species
    newTrain = train
    newTrain['Cx'] = Cx_train
    newTrain['Cy'] = Cy_train
    newTrain['XY'] = xy_train
    newTrain['XDY'] = xdy_train
    newTrain['XAY'] = xay_train
    newTrain['XIY'] = xiy_train

    newTrain['Perimeter'] =  Perimeter_train
    newTrain['Area'] =  Area_train
    newTrain['Ratio'] =  Ratio_train
    newTrain['Extent'] =  Extent_train
    newTrain['Diameter'] =  Dia_train
    newTrain['Direction'] = Direction_train


    return  newTrain

def getFeatureFromTestPic(path,test_id,test):
    areaList = []
    perimeterList = []       #周长
    ratioList = []           #宽高比率
    extentList = []          #轮廓面积与边界矩形面积的比
    diameterList = []        #相等圆面积的直径
    directionList = []       # 方向, 水平or竖直
    cxList = []              # 重心
    cyList = []
    xyList = []
    xdyList = []             #差
    xayList = []             #和
    xiyList = []

    #从图片中生成Area，perimeter，diameter
    for val in test_id:
        num = val
        num = str(num)
        val = num + '.jpg'
        completePath = path + 'images\\' + val

        #从文件读图片
        img = cv2.imread(completePath,0)

        ret,thresh = cv2.threshold(img,150,255,0)
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cnt = contours[0]

        # 计算重心
        M = cv2.moments(img)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # print "no"
            # cv2.imshow("img",im2)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            cx, cy = 0, 0
        cxList.append(cx)
        cyList.append(cy)
        xyList.append(cx*cy)
        xdyList.append(cx-cy)
        xayList.append(cx+cy)
        xiyList.append(float(cx)/cy)


        #获得 Area

        area = cv2.contourArea(cnt)
        area = round(area,2)

        areaList.append(area)

        perimeter = cv2.arcLength(cnt,True)
        perimeter = round(perimeter, 2)
        perimeterList.append(perimeter)

        x,y,w,h = cv2.boundingRect(cnt)
        ratio = float(w)/h
        ratioList.append(ratio)

        x,y,w,h = cv2.boundingRect(cnt)
        rect_area = w*h
        extent = float(area)/rect_area
        extent = round(extent, 2)
        extentList.append(extent)


        if(w > h):
            direction = 0
        else :
            direction = 1
        directionList.append(direction)

        equi_diameter = np.sqrt(4*area/np.pi)
        diameterList.append(equi_diameter)
    Cx_test = pd.DataFrame(cxList)
    Cx_test.columns = ['Cx']
    Cy_test = pd.DataFrame(cyList)
    Cy_test.columns = ['Cy']
    xy_test = pd.DataFrame(xyList)
    xy_test.columns = ['XY']
    xdy_test = pd.DataFrame(xdyList)
    xdy_test.columns = ['XDY']
    xay_test = pd.DataFrame(xayList)
    xay_test.columns = ['XAY']
    xiy_test = pd.DataFrame(xiyList)
    xiy_test.columns = ['XIY']

    Perimeter_test  = pd.DataFrame(perimeterList)
    Perimeter_test.columns = ['Perimeter']
    Area_test  = pd.DataFrame(areaList)
    Area_test.columns = ['Area']
    Ratio_test  = pd.DataFrame(ratioList)
    Ratio_test.columns = ['Ratio']
    Extent_test  = pd.DataFrame(extentList)
    Extent_test.columns = ['Extent']
    Dia_test  = pd.DataFrame(diameterList)
    Dia_test.columns = ['Diameter']
    Direction_test = pd.DataFrame(directionList)
    Direction_test.columns = ['Direction']

    #添加新特征到测试集中
    test['Cx'] = Cx_test
    test['Cy'] = Cy_test
    test['XY'] = xy_test
    test['XDY'] = xdy_test
    test['XAY'] = xay_test
    test['XIY'] = xiy_test

    test['Perimeter'] =  Perimeter_test
    test['Area'] =  Area_test
    test['Ratio'] =  Ratio_test
    test['Solidity'] =  Extent_test
    test['Diameter'] =  Dia_test
    test['Direction'] = Direction_test

    return test

# 按照长宽比重新生成等大图片
def getMaxWidthHeight(path):
    image_count = 1584
    images = []
    for i in range(1,image_count+1):
        img_path = path + str(i) + '.jpg'
        img = cv2.imread(img_path,0)
        images.append(img)

    widths = [img.shape[0] for img in images]
    heights = [img.shape[1] for img in images]
    max_width = max(widths)
    max_height = max(heights)
    return max_width,max_height

def resizeImage(fin,fout,size):
    image = Image.open(fin)
    w_ratio = float(size[0]) / image.size[0]
    h_ratio = float(size[1]) / image.size[1]

    if w_ratio <= h_ratio:
        ratio = w_ratio
    else:
        ratio = h_ratio

    new_size = int(ratio * image.size[0]), int(ratio * image.size[1])
    resized_image = image.resize(new_size, Image.ANTIALIAS)

    offset_x = int(max((size[0] - new_size[0]) / 2, 0))
    offset_y = int(max((size[1] - new_size[1]) / 2, 0))

    out_image = Image.new('L', size)
    out_image.paste(resized_image, (offset_x, offset_y))
    out_image.save(fout)

def resizeImageMain(path):

    image_count = 1584
    for i in range(1,image_count+1):
        fin = path + 'images\\'+ str(i) + '.jpg'
        fout = path + 'rev\\'+ str(i) + '.jpg'
        max_size = getMaxWidthHeight(path + 'images\\')
        resizeImage(fin,fout,max_size)


def reduceImage(fin,fout,scale = 0.05):
    image = Image.open(fin)
    new_size = int (image.size[0] * scale),int (image.size[1] * scale)
    resize_img = image.resize(new_size,Image.ANTIALIAS)
    resize_img.save(fout)

def reduceImageMain(Path):
    image_count = 1584
    for img_index in range(1,image_count+1):
        fin = Path +'rev\\'+ str(img_index) + '.jpg'
        fout = Path + 'imges_0.05\\' + str(img_index) + '.jpg'
        reduceImage(fin,fout)

def getPCAFeature(path,train_id,train,test_id,test):

    imgSet = []
    for i in train_id:
        completePath = path + 'images_0.05\\' + str(i) + '.jpg'
        img = cv2.imread(completePath,0)
        x,y = img.shape
        imgRow = img.reshape(x*y)
        imgSet.append(imgRow)
    for i in test_id:
        completePath = path + 'images_0.05\\' + str(i) + '.jpg'
        img = cv2.imread(completePath,0)
        x,y = img.shape
        imgRow = img.reshape(x*y)
        imgSet.append(imgRow)


    # pca = PCA(n_components = 100).fit(imgSet)
    # var = pca.explained_variance_
    # cmap = sns.color_palette()
    # plt.bar(np.arange(1, len(var) + 1), var / np.sum(var), align="center", color=cmap[0])
    # plt.step(np.arange(1, len(var) + 1), np.cumsum(var) / np.sum(var), where="mid", color=cmap[1])
    # plt.show()

    pca = PCA(n_components = 35)
    pca.fit(imgSet)
    imgFeature = pca.transform(imgSet)
    # var = pca.explained_variance_
    # cmap = sns.color_palette()
    # plt.bar(np.arange(1, len(var) + 1), var / np.sum(var), align="center", color=cmap[0])
    # plt.show()

    train_number = train_id.shape[0]


    x,y = imgFeature.shape
    for i in range(y):
        columns = 'feature'+ str(i+1)
        feature = pd.DataFrame(imgFeature[:train_number,i])
        feature.columns = [columns]
        train[columns] = feature
    for i in range(y):
        columns = 'feature' + str(i + 1)
        feature = pd.DataFrame(imgFeature[train_number:, i])
        feature.columns = [columns]
        test[columns] = feature

    return train,test

def encodingCategoricalSpecies2NumValues(train_species):
    label = LabelEncoder().fit(train_species)
    y_train = label.transform(train_species)

    return y_train,label



def preprocessData(train_new,test_new):
    #对数据进行中心化和方差归一化
    scaler = StandardScaler().fit(train_new)
    x_train = scaler.transform(train_new)
    scaler = StandardScaler().fit(test_new)
    x_test = scaler.transform(test_new)

    return x_train,x_test

def MyModel(x_train,y_train,x_test):
    # C 正则化系数lamda的倒数
    # tol迭代终止判据的误差范围
    clf = LogisticRegression(solver = 'lbfgs',multi_class = 'multinomial',
                             C=1000, tol=0.00008)#0.00005
    clf.fit(x_train,y_train)
    scores = cross_validation.cross_val_score(
    clf, x_train,y_train, cv = 5)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    y_test = clf.predict_proba(x_test)

    return y_test



def generateSubmissionFile(y_test,test_id,label):
    submission = pd.DataFrame(y_test,index = test_id,columns = label.classes_)
    submission.to_csv("MySubmission.csv")


def pipLine(path):
    train,test,train_id,test_id,train_species = getData(path)
    print train.shape
    newTrain = getFeatureFromTrainPic(path,train_id,train)
    newTest = getFeatureFromTestPic(path,test_id,test)
    newTrain,newTest = getPCAFeature(path,train_id,newTrain,test_id,newTest)


    x_train, x_test = preprocessData(newTrain,newTest)
    y_train,label = encodingCategoricalSpecies2NumValues(train_species)

    y_test = MyModel(x_train,y_train,x_test)

    generateSubmissionFile(y_test,test_id,label)
    #resizeImageMain(path)
    #reduceImageMain(path)



if __name__=="__main__":
    path = "C:\\Users\\Surflyan\\Desktop\\ML\\Kaggle\\"
    pipLine(path)


