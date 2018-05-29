# -*- coding: utf-8 -*-
from numpy import *
from time import sleep


# SMO算法的辅助函数
def loadDataSet(fileName):  #加载并预处理数据集
    dataMat = []; labelMat = []
    fr = open(fileName,'r')
    for line in fr.readlines():
        lineArr = line.strip().split('\t') # 以制表符分割
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  #提取前两个元素存入data.Mat中
        labelMat.append(float(lineArr[2]))  # [].append(),最终的形式是矩阵
    return dataMat,labelMat

def selectJrand(i,m):  # 该辅助函数用于在某个区间范围内随机选择一个整数
    j=i                 # m是所有alpha的数目，i是第一个alpha的下标
    while (j==i):
        j = int(random.uniform(0,m))  # random.uniform(0,m)用于生成指定范围内的随机浮点数
    return j

def clipAlpha(aj,H,L): # 该辅助函数用于在数值太大时对其进行调整
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

# 简化版SMO算法
def smoSimple(dataMatIn, classLabels, C, toler, maxIter): # 参数：数据集，类别标签，常数c，容错率，循环次数
    dataMatrix = mat(dataMatIn)  # mat()转换成矩阵类型
    labelMat = mat(classLabels).transpose()  #转置之前是列表，转置后是一个列向量
    b = 0; m,n = shape(dataMatrix)  # 得到行，列数，m行，n列
    alphas = mat(zeros((m,1)))  # zeros(shape, dtype=float, order='C'),所以也可以写作zeros((10,1),)
    iter = 0  # 该变量存储的是在没有任何alpha改变时遍历数据集的次数
    while (iter < maxIter):  # 限制循环迭代次数，也就是在数据集上遍历maxIter次，且不再发生任何alpha修改，则循环停止
        alphaPairsChanged = 0  # 每次循环时先设为0，然后再对整个集合顺序遍历，该变量用于记录alpha是否已经进行优化
        for i in range(m): # 遍历每行数据向量，m行
            # 该公式是分离超平面，我们预测值
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            #print 'fxi:',fxi
            Ei = fXi - float(labelMat[i]) # 预测值和真实输出之差
            # 如果误差很大就对该数据对应的alpha进行优化，正负间隔都会被测试，同时检查alpha值
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)  # 随机选择不等于i的0-m的第二个alpha值
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):   # 这里是对SMO最优化问题的子问题的约束条件的分析
                    L = max(0, alphas[j] - alphas[i]) # L和H分别是alpha所在的对角线端点的界
                    H = min(C, C + alphas[j] - alphas[i])  # 调整alphas[j]位于0到c之间
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print "L==H"; continue   # L=H停止本次循环
                # 是一个中间变量：eta=2xi*xi-xixi-xjxj，是alphas[j]的最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T \
                                      - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print "eta>=0"; continue  # eta>=0停止本次循环，这里是简化计算
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta  # 沿着约束方向未考虑不等式约束时的alpha[j]的解
                alphas[j] = clipAlpha(alphas[j],H,L)    # 此处是考虑不等式约束的alpha[j]解
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print "j not moving enough"; continue  # 如果该alpha值不再变化，就停止该alpha的优化
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j]) # 更新alpha[i]
                # 完成两个alpha变量的更新后，都要重新计算阈值b
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T \
                             - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]* dataMatrix[j,:].T #李航统计学习7.115式
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T \
                              - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T  #李航统计学习7.116式
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0  # alpha[i]和alpha[j]是0或者c,就取中点作为b
                alphaPairsChanged += 1   # 到此的话说明已经成功改变了一对alpha
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1 # 如果alpha不再改变迭代次数就加1
        else: iter = 0
        print "iteration number: %d" % iter
    return b,alphas


# 主函数
dataArr,labelArr=loadDataSet('testSet.txt')  # 因为在同一文件夹下，就不用写绝对路径
#print dataArr
b,alphas=smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
print 'b:',b
print 'alphas[alphas>0]:',alphas[alphas>0]  # 数组过滤
print shape(alphas[alphas>0])  # 得到支持向量的个数
for i in range(100):  # 得到是支持向量的数据点
    if alphas[i]>0.0: print dataArr[i],labelArr[i]