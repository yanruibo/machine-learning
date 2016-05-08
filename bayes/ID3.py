#!/usr/bin/python
# encoding: utf-8

'''
Created on Nov 6, 2015

@author: yanruibo

ID3算法

'''

from math import log
import operator
import numpy as np
import random
import treePlotter
'''
计算整个数据集的经验熵　对应统计学习方法中的H(D)
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt
'''
切分数据集，将第axis列，值为value的元素去掉
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # print featVec
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            # reducedFeatVec = np.vstack((reducedFeatVec,featVec[axis + 1:]))
            # print reducedFeatVec
            retDataSet.append(reducedFeatVec)
    return retDataSet
'''
选择最好的特征来切分，输入数据集和阀值
是按照信息增益来切分的，如果一个特征的信息增益小于这个阀值，则返回-1，
如果一个特征的信息增益大于阀值，则返回信息增益最大的特征所在的列的索引
'''    
def chooseBestFeatureToSplit(dataSet, threshhold=0):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature即第i列
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    if(bestInfoGain < threshhold):
        return -1
    return bestFeature  # returns an integer
'''
投票函数，选择类别集中数量最多的那个值
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
'''
创建决策树，这里对代码进行了一些改动，不改变传入的labels的值。
'''
def createTree(dataSet, labels, threshhold=0):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    
    bestFeat = chooseBestFeatureToSplit(dataSet, threshhold)
    
    if(bestFeat == -1):
        return majorityCnt(classList)
    
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    #在这里对书中的代码进行了改动，将labels去掉最好的特征标签之后赋值给另外一个变量
    remainedLabels = list(set(labels) - set(labels[bestFeat]))
    # del(labels[bestFeat])　原书中的代码会改变传入参数label的值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        subLabels = remainedLabels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
'''
分类函数
输入创建好的决策树，类别标签和测试向量　原书代码有bug不适合UCI的数据集
'''    
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    # print 'in classify method featLabels',featLabels
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: 
        classLabel = valueOfFeat
    return classLabel
'''
分类函数
输入创建好的决策树，类别标签和测试向量，对书中的代码进行改进
如果在决策树中，在某一层没有测试数据的值，就在该层的任意一个分支往下找，这里取的是第一个分支，也就是最左边的分支。
'''
def classify1(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = None
    
    if testVec[featIndex] in secondDict.keys():
        for key in secondDict.keys():
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = classify1(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
    else:
        key = secondDict.keys()[0]
        if(type(secondDict[key]).__name__ == 'dict'):
            classLabel = classify1(secondDict[key], featLabels, testVec)
        else:
            classLabel = secondDict[key]
    return classLabel
'''
pickle存储树结构
'''
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()
'''
pickle读取树结构
'''
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
'''
用UCI　bread cancer数据测试ID3算法
'''    
def testID3UCI():
##################################################
#     2. Clump Thickness               1 - 10
#     3. Uniformity of Cell Size       1 - 10
#     4. Uniformity of Cell Shape      1 - 10
#     5. Marginal Adhesion             1 - 10
#     6. Single Epithelial Cell Size   1 - 10
#     7. Bare Nuclei                   1 - 10
#     8. Bland Chromatin               1 - 10
#     9. Normal Nucleoli               1 - 10
#     10. Mitoses                       1 - 10
#########################################################
    #设置十个类别标签
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    loadData = []
    for line in open('formatted-data.txt'):
        loadData.append([int(item) for item in line.strip().split(' ')])
        
    totalNum = len(loadData)
    #随机选取140个数据作为测试集，剩下的作为训练集
    testSet = random.sample(range(0, totalNum), 140)
    trainSet = list(set(range(totalNum)) - set(testSet))
    trainMatrix = []
    testMatrix = []
    for item in trainSet:
        trainMatrix.append(loadData[item])
    for item in testSet:
        testMatrix.append(loadData[item])
        
    tree = createTree(trainMatrix, labels, 0)
    #print "tree", tree
    #print 'labels', labels
    #treePlotter.createPlot(tree)
    featLabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    correctCount = 0
    #对测试集的每个测试向量做测试
    for item in testMatrix:
        if(classify1(tree, featLabels, item[:len(item) - 1]) == item[len(item) - 1]):
            correctCount += 1
    accuracy = float(correctCount) / len(testMatrix)
    print "In ID3　classifier, accuracy is", accuracy
    return accuracy 
    
