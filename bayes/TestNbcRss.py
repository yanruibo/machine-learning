#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
Created on Oct 31, 2015

@author: yanruibo

用feedparser数据集测试NBC(朴素贝叶斯)算法
因为前面测试是从所有数据中随机选择20个文档数据做为测试集，剩下的作为训练集
这里测试count次，取精确度的平均值
'''
import feedparser
import bayes
import time
if __name__ == '__main__':
    
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    errorRateSum = 0.0
    count = 200
    for i in range(count):
        vocabList, p0V, p1V,errorRate = bayes.localWords(ny, sf)
        errorRateSum+=errorRate
    averageAccuracy =(1-errorRateSum/count)
    print "average accuracy is: ", averageAccuracy
    