#! /usr/bin/env python
#coding=utf-8

'''
Created on 2015年10月15日

@author: yanruibo
'''

import numpy as np
from kNN import *
from kdtree import *
import time
'''
测试KD树算法的类
'''
if __name__ == '__main__':
    '''
    group = array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    print group.shape
    print group.shape[0]
    print group.shape[1]
    print tile([1,2,3],(4,2))
    '''
    '''
    x = array([[3,4],[5,6],[2,2],[8,4]])
    xT = x.T
    D = cov(xT)
    print D.shape
    invD = linalg.inv(D)
    tp = x[0] - x[1]
    print sqrt(dot(dot(tp, invD), tp.T))
    '''
    
    '''
    point_list = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
    labels= np.array([2,5,9,4,8,7])
    data = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
    point = [7,0]
    tree = KDTree(data,labels)
    nearest,label = tree.search(point, 2)
    
    print nearest,label
    '''
    
    '''
    point_list = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
    label = np.array([2,5,9,4,8,7])
    sorted_index = np.argsort(point_list[:,1]) #按第axis列排序
    point_list = point_list[sorted_index]
    sorted_label = []
    for idx in sorted_index:
        sorted_label.append(label[idx])
    print point_list
    print sorted_label
    '''
    k = 3
    startTimeStamp = time.time()
    error_count, error_rate,precisions,recalls,f1s = handwritingClassTestKDTree(k)
    # 统计写入时间
    endTimeStamp = time.time()
    total_time = endTimeStamp - startTimeStamp
    ft = open("kdtree_total_time.txt", "a")
    ft.write("k : " + str(k) + "\n")
    ft.write("Total Time : " + str(total_time) + "\n")
    ft.write("error_count : " + str(error_count) + "\n")
    ft.write("Error Rate : " + str(error_rate) + "\n")
    ft.write("precisions : " + str(precisions) + "\n")
    ft.write("recalls : " + str(recalls) + "\n")
    ft.write("f1s : " + str(f1s) + "\n")
    ft.close()
    
    
    