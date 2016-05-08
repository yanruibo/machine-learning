#!/usr/bin/python
# encoding: utf-8

'''
Created on Nov 6, 2015

@author: yanruibo


用UCI　breast cancer数据集测试NBC算法
因为前面测试NBC是从所有数据中随机选择140个数据做为测试集，剩下的作为训练集
这里测试count次，取精确度的平均值


'''
import bayes_uci

if __name__ == '__main__':
    
    accuracySum = 0
    count = 200
    for i in range(count):
        accuracy = bayes_uci.testNBC()
        accuracySum += accuracy
    print "average accuracy: ",accuracySum/count