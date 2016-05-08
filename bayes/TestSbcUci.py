#!/usr/bin/python
# encoding: utf-8

'''
Created on Nov 6, 2015

@author: yanruibo

用UCI　breast cancer数据集测试SBC算法
因为前面测试ID3是从所有数据中随机选择140个数据做为测试集，剩下的作为训练集
这里测试count次，取精确度的平均值

'''
import bayes_uci
import time
if __name__ == '__main__':
    startTimeStamp = time.time()
    accuracySum = 0.0
    count = 200
    for i in range(count):
        accuracy = bayes_uci.testSBC()
        accuracySum += accuracy
    averageAccuracy = accuracySum/count
    print "average accuracy: ",averageAccuracy
    endTimeStamp = time.time()
    total_time = endTimeStamp - startTimeStamp
    ft = open("test_sbc_uci.txt", "a")
    ft.write("Normal Total Time : " + str(total_time) + "\n")
    ft.write("average accuracy : " + str(averageAccuracy) + "\n")
    ft.close()
