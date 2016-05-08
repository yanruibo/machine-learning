# coding=utf-8
'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
import numpy as np
import operator
from os import listdir
from kdtree import *
'''
书上的例子
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()     
    classCount = {}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # classCount.get(voteIlabel,0)是一种技巧　直接用下标会导致KeyError
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    returnMat = np.zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector
    
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals
   
def datingClassTest():
    hoRatio = 0.50  # hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))
    print errorCount
'''
将图像转化为一维向量
''' 
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect
'''
设置k值，测试书上的例子
'''
def handwritingClassTest(k):
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    #到这里训练数据集加载完成保存在trainMat中，1934*1024，训练集标签存在hwLabels中
    '''
    我看统计学学方法这本书上讲，precision,recall,F1是计算两类分类问题的，这里要计算的话得计算10个，
    TPs[i]表示测试完毕之后值为i时的的测试样本的数目 i取0-9
    FNs TNs FPs 类似
    precisions[i]标示分类为i的准确率，i取0-9
    recalls和f1s类似
    '''
    
    TPs = np.zeros([10])
    FNs = np.zeros([10])
    TNs = np.zeros([10])
    FPs = np.zeros([10])
    precisions = np.zeros([10])
    recalls =  np.zeros([10])
    f1s =  np.zeros([10]) 
    
    errorCount = 0.0
    testFileList = listdir('testDigits')  # iterate through the test set
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, k)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): 
            errorCount += 1.0
        #对于每一个测试结果进行计算TP FN TN FP的个数
        for i in range(10):
            '''
            属于类C的样本被正确分类到类C，记这一类样本数为 TP
            不属于类C的样本被错误分类到类C，记这一类样本数为 FN
            属于类别C的样本被错误分类到类C的其他类，记这一类样本数为 TN
            不属于类别C的样本被正确分类到了类别C的其他类，记这一类样本数为 FP
            '''
            if(classNumStr == i and classifierResult == i):
                TPs[i] += 1.0
            if(classNumStr != i and classifierResult == i):
                FNs[i] += 1.0
            if(classNumStr == i and classifierResult != i):
                TNs[i] += 1.0
            if(classNumStr != i and classifierResult != i):
                FPs[i] += 1.0
            
    error_rate = errorCount / float(mTest)
    #计算每个类别的评价指标并返回
    for i in range(10):
        precisions[i] = TPs[i]/(TPs[i]+FPs[i])
        recalls[i] = TPs[i]/(TPs[i]+FNs[i])
        f1s[i] = (2*TPs[i])/(2*TPs[i]+FPs[i]+FNs[i])
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (error_rate)
    return errorCount, error_rate,precisions,recalls,f1s
'''
对训练集中的全为0的列进行删除
'''
def clean(dataSet):
    dataSetColumnSize = dataSet.shape[1]
    #通过判断列的和判断某一列是否全为0
    # 对列求和 0是列 1是行 
    columnSum = dataSet.sum(axis=0)
    zeroColumns = []
    for i in range(dataSetColumnSize):
        if(columnSum[i] == 0):
            zeroColumns.append(i)
    # 0是行 1是列
    nonZeroDataSet = np.delete(dataSet, zeroColumns, 1)
    return nonZeroDataSet, zeroColumns

'''
按照马氏距离分类
'''
def classifyByMD(inX, dataSet, labels, k, invD):
    #获得对全零列删减后的训练数据集的行数
    dataSetSize = dataSet.shape[0]
    MDs = []
    #计算测试向量与训练集的每个向量的马氏距离保存在MDs中
    for i in range(dataSetSize):
        tp = inX[0] - dataSet[i]
        md = np.sqrt(np.dot(np.dot(tp, invD), tp.T))
        MDs.append(md)
    NAMDs = np.array(MDs)
    #对计算出的马氏距离进行排序
    sortedDistIndicies = NAMDs.argsort()
    ##############################
    #这是一个多类分类问题的技巧，通过用字典保存label和数量，然后按数量进行排序
    #最后返回第一个字典的label值即sortedClassCount[0][0]
    ##############################
    classCount = {}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

'''
对应于classifyByMD，用马氏距离作为度量距离的分类方法，并对训练集中的全０列进行删掉操作，测试向量的相应位置也删掉
'''
def handwritingClassTestMD(k):
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    ##############################
    #计算对全0列进行删除之后的训练数据集的协方差及其逆矩阵
    dataSet, zeroColumns = clean(trainingMat)
    dataSetSize = dataSet.shape[0]
    dataSetT = dataSet.transpose()
    D = np.cov(dataSetT)
    invD = np.linalg.pinv(D)
    ##############################
    testFileList = listdir('testDigits')  # iterate through the test set
    
    #相同的方法计算 评价指标
    TPs = np.zeros([10])
    FNs = np.zeros([10])
    TNs = np.zeros([10])
    FPs = np.zeros([10])
    precisions = np.zeros([10])
    recalls =  np.zeros([10])
    f1s =  np.zeros([10]) 
    
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        #这一点很重要，训练数据集中的全0列删除之后，对测试向量中相应的列也要删除
        cutted_inX = np.delete(vectorUnderTest, zeroColumns, 1)
        classifierResult = classifyByMD(cutted_inX, dataSet, hwLabels, k, invD)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
        
        for i in range(10):
            if(classNumStr == i and classifierResult == i):
                TPs[i] += 1.0
            if(classNumStr != i and classifierResult == i):
                FNs[i] += 1.0
            if(classNumStr == i and classifierResult != i):
                TNs[i] += 1.0
            if(classNumStr != i and classifierResult != i):
                FPs[i] += 1.0
        
    
    error_rate = errorCount / float(mTest)
    
    for i in range(10):
        precisions[i] = TPs[i]/(TPs[i]+FPs[i])
        recalls[i] = TPs[i]/(TPs[i]+FNs[i])
        f1s[i] = (2*TPs[i])/(2*TPs[i]+FPs[i]+FNs[i])
        
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (error_rate)
    return errorCount, error_rate,precisions,recalls,f1s
    

'''
对应于classifyByMD，用马氏距离作为度量距离的分类方法，不对训练集中的全０列进行删掉操作
'''
def handwritingClassTestMDNoCutting(k):
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    ##############################
    #计算对全0列进行删除之后的训练数据集的协方差及其逆矩阵
    dataSetT = trainingMat.transpose()
    D = np.cov(dataSetT)
    invD = np.linalg.pinv(D)
    ##############################
    testFileList = listdir('testDigits')  # iterate through the test set
    
    #相同的方法计算 评价指标
    TPs = np.zeros([10])
    FNs = np.zeros([10])
    TNs = np.zeros([10])
    FPs = np.zeros([10])
    precisions = np.zeros([10])
    recalls =  np.zeros([10])
    f1s =  np.zeros([10]) 
    
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        #这一点很重要，训练数据集中的全0列删除之后，对测试向量中相应的列也要删除
    
        classifierResult = classifyByMD(vectorUnderTest, trainingMat, hwLabels, k, invD)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
        
        for i in range(10):
            if(classNumStr == i and classifierResult == i):
                TPs[i] += 1.0
            if(classNumStr != i and classifierResult == i):
                FNs[i] += 1.0
            if(classNumStr == i and classifierResult != i):
                TNs[i] += 1.0
            if(classNumStr != i and classifierResult != i):
                FPs[i] += 1.0
        
    
    error_rate = errorCount / float(mTest)
    
    for i in range(10):
        precisions[i] = TPs[i]/(TPs[i]+FPs[i])
        recalls[i] = TPs[i]/(TPs[i]+FNs[i])
        f1s[i] = (2*TPs[i])/(2*TPs[i]+FPs[i]+FNs[i])
        
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (error_rate)
    return errorCount, error_rate,precisions,recalls,f1s

def handwritingClassTestKDTree(k):
    #加载数据与前面相同
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    
    #相同的方法计算 评价指标
    TPs = np.zeros([10])
    FNs = np.zeros([10])
    TNs = np.zeros([10])
    FPs = np.zeros([10])
    precisions = np.zeros([10])
    recalls =  np.zeros([10])
    f1s =  np.zeros([10]) 
    
    
    mTest = len(testFileList)
    # 建立KD树
    # print "trainningMat.shape",trainingMat.shape
    tree = KDTree(trainingMat, hwLabels)
    
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        # vectorUnderTest的shape是（1,1024），是[[0,1,2,...,1023]]
        # print "vectorUnderTest",vectorUnderTest
        # print "len of vectorUnderTest",len(vectorUnderTest)
        # classifierResult = classifyByMD(vectorUnderTest, trainingMat, hwLabels, 3)
        # KD树查询
        nearests, labels = tree.search(vectorUnderTest[0], k)
        #同样的方法进行排序
        classCount = {}          
        for i in range(3):
            voteIlabel = labels[i]
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
            sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        classifierResult = sortedClassCount[0][0]
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): 
            errorCount += 1.0
        for i in range(10):
            if(classNumStr == i and classifierResult == i):
                TPs[i] += 1.0
            if(classNumStr != i and classifierResult == i):
                FNs[i] += 1.0
            if(classNumStr == i and classifierResult != i):
                TNs[i] += 1.0
            if(classNumStr != i and classifierResult != i):
                FPs[i] += 1.0
    error_rate = errorCount / float(mTest)
    for i in range(10):
        precisions[i] = TPs[i]/(TPs[i]+FPs[i])
        recalls[i] = TPs[i]/(TPs[i]+FNs[i])
        f1s[i] = (2*TPs[i])/(2*TPs[i]+FPs[i]+FNs[i])
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (error_rate)
    
    return errorCount, error_rate,precisions,recalls,f1s

