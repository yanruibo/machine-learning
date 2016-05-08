#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
Created on 2015年10月25日

@author: yanruibo
'''
import numpy as np
from kNN import *
from kdtree import *
import time
'''
测试书上算法的类
'''
if __name__ == '__main__':
    k = 3
    startTimeStamp = time.time()
    error_count, error_rate,precisions,recalls,f1s = handwritingClassTest(k)
    # 统计写入时间
    endTimeStamp = time.time()
    total_time = endTimeStamp - startTimeStamp
    ft = open("original_total_time.txt", "a")
    ft.write("k : " + str(k) + "\n")
    ft.write("Total Time : " + str(total_time) + "\n")
    ft.write("error_count : " + str(error_count) + "\n")
    ft.write("Error Rate : " + str(error_rate) + "\n")
    ft.write("precisions : " + str(precisions) + "\n")
    ft.write("recalls : " + str(recalls) + "\n")
    ft.write("f1s : " + str(f1s) + "\n")
    ft.close()