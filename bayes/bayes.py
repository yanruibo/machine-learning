#!/usr/bin/python
# encoding: utf-8

'''
Created on Oct 19, 2010

@author: Peter

bayes算法

'''
import numpy as np
'''
作者自己模拟的一个数据集
'''
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec
'''
计算出单词的集合，去除重复的单词
'''                 
def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)
'''
将文档转化为向量
'''
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

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
词袋模型，将文档转化为单词数的向量
'''    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
'''
测试是否是侮辱性词汇的测试函数
'''
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
'''
将大字符串切分成小字符串
'''
def textParse(bigString):  # input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
'''
测试垃圾邮件
'''    
def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    trainingSet = range(50); testSet = []  # create test set
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error", docList[docIndex]
    print 'the error rate is: ', float(errorCount) / len(testSet)
    # return vocabList,fullText
'''
计算最常出现的单词，选取前30个
'''
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        # count() 方法用于统计某个元素在列表中出现的次数　这个函数很有用
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]
'''
测试用feedparser抓取的两个rss文档测试NBC
'''
def localWords(feed1, feed0):
    import feedparser
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    #print "minLen", minLen
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2 * minLen); testSet = []  # create test set
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    errorRate = float(errorCount) / len(testSet)
    print "accuracy : ",1-errorRate
    return vocabList, p0V, p1V,errorRate

'''
Selective Bayesian Classifier
从空集到全集搜索
'''
def SBC(feed1, feed0):
    
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    #print "minLen", minLen
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
        
    
    trainingSet = range(2 * minLen); testSet = []  # create test set
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    testMat = []; testClasses = []
    for docIndex in testSet:
        testMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        testClasses.append(classList[docIndex])
    
    # 属性就是vocabList 训练集：trainMat trainClasses 测试集: testMat testClasses
    
    
    #以第一个特征计算　目的是初始化值
    trainMat = np.asarray(trainMat)
    #print "trainMat",trainMat
    #print "trainMat.shape",trainMat.shape
    trainClasses = np.asarray(trainClasses)
    errorRate, p0V, p1V, pSpam = cross_validation(trainMat[:,0].reshape((len(trainMat[:,0]),1)), trainClasses)
    bestErrorRate = errorRate

    totalFeatureNum = len(trainMat[0])
    fullIndexes = range(totalFeatureNum)
    remainedIndexes = fullIndexes
    
    
    bestIndex = remainedIndexes[0]
    for i in range(1,len(remainedIndexes)):
        errorRate, p0V, p1V, pSpam = cross_validation(trainMat[:,i].reshape((len(trainMat[:,i]),1)), trainClasses)
        if(errorRate<bestErrorRate):
            bestErrorRate=errorRate
            bestIndex = remainedIndexes[i]
    
    currentMatrix = trainMat[:,bestIndex].reshape(len(trainMat[:,bestIndex]),1)

    selectedColumnIndexes = [bestIndex]
    remainedIndexes = list(set(fullIndexes) - set(selectedColumnIndexes))
    
    decisionRemainedIndexes = remainedIndexes
    decisionP0V = p0V
    decisionP1V = p1V
    decisionPSpam = pSpam
    
    while(len(selectedColumnIndexes)<=totalFeatureNum):
        isChanged = False
        bestIndex = remainedIndexes[0]
        for i in range(len(remainedIndexes)):
            errorRate, p0V, p1V, pSpam = cross_validation(
            np.append(currentMatrix,trainMat[:,remainedIndexes[i]].reshape(len(trainMat[:,remainedIndexes[i]]),1),axis=1), 
            trainClasses)
            if(errorRate<=bestErrorRate):
                isChanged = True
                bestErrorRate = errorRate
                bestIndex = remainedIndexes[i]
                
        if(isChanged):
            currentMatrix = np.append(currentMatrix,trainMat[:,bestIndex].reshape(len(trainMat[:,bestIndex]),1),axis=1)
            selectedColumnIndexes.append(bestIndex)
            remainedIndexes = list(set(fullIndexes) - set(selectedColumnIndexes))
            print "selectedColumnIndexes",selectedColumnIndexes
            decisionRemainedIndexes = remainedIndexes
            decisionP0V = p0V
            decisionP1V = p1V
            decisionPSpam = pSpam
            
        else:
            break
    
    
    #计算所有的属性集
    errorRate, p0V, p1V, pSpam = cross_validation(trainMat, trainClasses)
    if(errorRate<bestErrorRate):
        decisionRemainedIndexes = []
        decisionP0V = p0V
        decisionP1V = p1V
        decisionPSpam = pSpam
    
    
    # test 测试集: testMat testClasses
    remainedTestMat = np.delete(testMat, decisionRemainedIndexes, axis=1)
    errorCount = 0
    for i in range(len(remainedTestMat)):
        if classifyNB(remainedTestMat[i], decisionP0V, decisionP1V, decisionPSpam) != testClasses[i]:
            errorCount += 1
    
    errorRate = float(errorCount) / len(remainedTestMat)
    print 'the error rate is: ', errorRate
    return errorRate

'''
leave-one-out　cross validation
留一法验证
'''

def cross_validation(trainMat, trainClasses):
    #print "in cross_validation trainMat",trainMat
    #print "in cross_validation trainMat.shape",trainMat.shape
    #print "in cross_validation trainClasses",trainClasses
    #print "in cross_validation trainClasses.shape",trainClasses.shape
    errorCount = 0
    # 如果不是ndarray转化为ndarray
    trainMat = np.asarray(trainMat)
    trainClasses = np.asarray(trainClasses)
    p0V = None
    p1V = None
    pSpam = None
    for i in range(len(trainMat)):
        # 0是行 1是列
        remainedTrainMat = np.delete(trainMat, [i], 0)
        remainedTrainClasses = np.delete(trainClasses, [i], 0)
        p0V, p1V, pSpam = trainNB0(remainedTrainMat, remainedTrainClasses)
        if classifyNB(trainMat[i], p0V, p1V, pSpam) != trainClasses[i]:
            errorCount += 1
    errorRate = float(errorCount) / len(trainMat)
    #print 'in cross validation, the error rate is: ', errorRate
    return errorRate, p0V, p1V, pSpam
    
def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []; topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]
