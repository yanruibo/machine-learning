#!/usr/bin/python
# encoding: utf-8

'''
Created on Nov 7, 2015

@author: yanruibo

用UCI　breast cancer数据集测试ID3算法
因为前面测试ID3是从所有数据中随机选择140个数据做为测试集，剩下的作为训练集
这里测试count次，取精确度的平均值

'''
import ID3
if __name__ == '__main__':
    
    accuracySum = 0.0
    count = 200
    for i in range(count):
        accuracy = ID3.testID3UCI()
        accuracySum += accuracy
    averageAccuracy = accuracySum/count
    print "average accuracy : ",averageAccuracy