#!/usr/bin/python
# encoding: utf-8

'''
Created on Nov 7, 2015

@author: yanruibo

用UCI　breast cancer数据集测试C4.5算法
因为前面测试C4.5是从所有数据中随机选择140个数据做为测试集，剩下的作为训练集
这里测试count次，取精确度的平均值

'''
import C45

if __name__ == '__main__':
    
    accuracySum = 0.0
    count = 200
    for i in range(count):
        accuracy = C45.testC45UCI()
        accuracySum += accuracy
    averageAccuracy = accuracySum/count
    print "average accuracy : ",averageAccuracy
    
    