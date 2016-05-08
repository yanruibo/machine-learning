#!/usr/bin/python
# encoding: utf-8

'''
Created on Nov 6, 2015

@author: yanruibo

用UCI breast cancer数据集测试的，其实大部分内容都和bayes.py中的内容相同

'''


import numpy as np
import random
                 
'''
朴素贝叶斯分类器训练函数，求出训练出先验概率和似然。
'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)  # change to ones() 
    p0Denom = 2.0; p1Denom = 2.0  # change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)  # change to log()
    p0Vect = np.log(p0Num / p0Denom)  # change to log()
    return p0Vect, p1Vect, pAbusive

'''
根据前面计算出的先验概率和似然对输入的一个向量进行分类
'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
'''
用UCI的数据测试SBC算法
'''    
def testSBC():
    loadData = np.loadtxt("formatted-data.txt")
    totalNum = len(loadData)
    #随机选取140个索引作为训练集的索引，即从总的测试集中随机选取140个测试数据，剩下的作为训练集
    #为什么选140个，因为总数为699,测试数据一般为总数的10%-20%
    testSet = random.sample(range(0, totalNum), 140)
    trainSet = list(set(range(totalNum)) - set(testSet))
    # print testSet
    # 0 row 1 column
    #把测试集从总的数据中删除得到训练集
    trainMatrix = np.delete(loadData, testSet, axis=0)
    testMatrix = np.delete(loadData, trainSet, axis=0)
    #把label列删掉
    trainMat = np.delete(trainMatrix, [len(trainMatrix[0]) - 1], axis=1)
    trainClasses = trainMatrix[:, -1]
    testMat = np.delete(testMatrix, [len(testMatrix[0]) - 1], axis=1)
    testClasses = testMatrix[:, -1]
    
    #print trainMat.shape
    #print trainClasses.shape
    #print testMat.shape
    #print testClasses.shape
    return SBC(trainMat, trainClasses, testMat, testClasses)
'''
用UCI breast cancer数据测试NBC算法
'''    
def testNBC():
    
    loadData = np.loadtxt("formatted-data.txt")
    totalNum = len(loadData)
    #思路和上面测试SBC的算法相同，也是从总的数据集中随机选取140个作为测试集，剩下的作为训练集
    testSet = random.sample(range(0, totalNum), 140)
    trainSet = list(set(range(totalNum)) - set(testSet))
    trainMatrix = np.delete(loadData, testSet, axis=0)
    testMatrix = np.delete(loadData, trainSet, axis=0)
    
    trainMat = np.delete(trainMatrix, [len(trainMatrix[0]) - 1], axis=1)
    trainClasses = trainMatrix[:, -1]
    testMat = np.delete(testMatrix, [len(testMatrix[0]) - 1], axis=1)
    testClasses = testMatrix[:, -1]
    
    p0V, p1V, pSpam = trainNB0(trainMat, trainClasses)
    errorCount = 0
    for i in range(len(testSet)):
        if classifyNB(testMat[i], p0V, p1V, pSpam) != testClasses[i]:
            errorCount += 1
    errrRate = float(errorCount) / len(testSet)
    print 'the accuracy is: ',(1-errrRate)
    accuracy = 1-errrRate
    return accuracy
    
'''
Selective Bayesian Classifier
从空集到全集搜索
最主要的算法
'''
def SBC(trainMat, trainClasses, testMat, testClasses):
    # 属性就是vocabList 训练集：trainMat trainClasses 测试集: testMat testClasses
    trainMat = np.asarray(trainMat)
    trainClasses = np.asarray(trainClasses)
    
    # 以第一个特征计算　目的是初始化值
    errorRate, p0V, p1V, pSpam = cross_validation(trainMat[:, 0].reshape((len(trainMat[:, 0]), 1)), trainClasses)
    bestErrorRate = errorRate
    
    #totalFeatureNum记录共有多少列
    totalFeatureNum = len(trainMat[0])
    fullIndexes = range(totalFeatureNum)
    #remainedIndexes记录选取之后剩下的列的索引
    remainedIndexes = fullIndexes
    #bestIndex记录每一次要添加的最好的特征的列索引也就是第几列
    bestIndex = remainedIndexes[0]
    #选取一个属性时测试第一列之后的列，选取最好的那一列，这个没有放入循环中，主要是做一些初始化工作
    for i in range(1, len(remainedIndexes)):    
        errorRate, p0V, p1V, pSpam = cross_validation(trainMat[:, i].reshape((len(trainMat[:, i]), 1)), trainClasses)
        if(errorRate < bestErrorRate):
            bestErrorRate = errorRate
            bestIndex = remainedIndexes[i]
    #将当前矩阵初始化为最好的那一列的值，这里需要将矩阵变成2维的，因为trainMat[:, bestIndex]是一维的。
    currentMatrix = trainMat[:, bestIndex].reshape(len(trainMat[:, bestIndex]), 1)
    #selectedColumnIndexes记录当前所有的最好的特征的列索引
    selectedColumnIndexes = [bestIndex]
    
    remainedIndexes = list(set(fullIndexes) - set(selectedColumnIndexes))
    #decisionRemainedIndexes记录最终确定的剩余的特征的索引，这个需要返回，因为测试向量中相应的列要删掉
    decisionRemainedIndexes = remainedIndexes
    decisionP0V = p0V
    decisionP1V = p1V
    decisionPSpam = pSpam
    #算法的核心两重循环
    while(len(selectedColumnIndexes) < totalFeatureNum):
        isChanged = False
        bestIndex = remainedIndexes[0]
        #测试剩余的列与当前已选则列的所有组合的准确率，选出最好的
        for i in range(len(remainedIndexes)):
            errorRate, p0V, p1V, pSpam = cross_validation(
            np.append(currentMatrix, trainMat[:, remainedIndexes[i]].reshape(len(trainMat[:, remainedIndexes[i]]), 1), axis=1),
            trainClasses)
            #只要准确率没有降低就继续添加特征
            if(errorRate <= bestErrorRate):
                isChanged = True
                bestErrorRate = errorRate
                bestIndex = remainedIndexes[i]
                print "iterate bestErrorRate",bestErrorRate
        #如果当前准确率有提高，就更新记录变量的值，如果没有提高就停止添加特征跳出循环        
        if(isChanged):
            currentMatrix = np.append(currentMatrix, trainMat[:, bestIndex].reshape(len(trainMat[:, bestIndex]), 1), axis=1)
            selectedColumnIndexes.append(bestIndex)
            remainedIndexes = list(set(fullIndexes) - set(selectedColumnIndexes))
            print "selectedColumnIndexes", selectedColumnIndexes
            decisionRemainedIndexes = remainedIndexes
            decisionP0V = p0V
            decisionP1V = p1V
            decisionPSpam = pSpam
            
        else:
            break
    
    
    # 计算所有的特征集，文章中提到了要计算一下所有的特征集
    errorRate, p0V, p1V, pSpam = cross_validation(trainMat, trainClasses)
    if(errorRate < bestErrorRate):
        decisionRemainedIndexes = []
        decisionP0V = p0V
        decisionP1V = p1V
        decisionPSpam = pSpam
    
    print "decisionRemainedIndexes", decisionRemainedIndexes
    print "decisionP0V", decisionP0V
    print "decisionP1V", decisionP1V
    print "decisionPSpam", decisionPSpam
    
    # test 测试集: testMat testClasses 将无用的属性列去掉
    remainedTestMat = np.delete(testMat, decisionRemainedIndexes, axis=1)
    
    errorCount = 0
    for i in range(len(remainedTestMat)):
        if classifyNB(remainedTestMat[i], decisionP0V, decisionP1V, decisionPSpam) != testClasses[i]:
            errorCount += 1
    
    errorRate = float(errorCount) / len(remainedTestMat)
    accuracy = 1-errorRate
    print "accuracy: ",accuracy
    return accuracy
'''
leave-one-out　cross validation
留一法验证
'''

def cross_validation(trainMat, trainClasses):
    errorCount = 0
    # 如果不是ndarray转化为ndarray
    trainMat = np.asarray(trainMat)
    trainClasses = np.asarray(trainClasses)
    p0V = None
    p1V = None
    pSpam = None
    #每次用第i行做测试向量
    for i in range(len(trainMat)):
        # 0是行 1是列
        remainedTrainMat = np.delete(trainMat, [i], 0)
        remainedTrainClasses = np.delete(trainClasses, [i], 0)
        p0V, p1V, pSpam = trainNB0(remainedTrainMat, remainedTrainClasses)
        if classifyNB(trainMat[i], p0V, p1V, pSpam) != trainClasses[i]:
            errorCount += 1
    errorRate = float(errorCount) / len(trainMat)
    return errorRate, p0V, p1V, pSpam
    
