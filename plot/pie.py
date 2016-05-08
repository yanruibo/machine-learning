#! /usr/bin/env python
#coding=utf-8

'''
Created on 2015年9月29日
@author: yanruibo
饼图的绘制
attributes:
android.permission.INTERNET
android.permission.CAMERA
android.permission.ACCESS_FINE_LOCATION
android.permission.WRITE_EXTERNAL_STORAGE
android.permission.RECORD_AUDIO
android.permission.RECORD_VIDEO
android.permission.READ_CONTACTS
android.permission.GET_ACCOUNTS
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl

if __name__ == '__main__':
    #饼图的数据处理有点不同，不同的是按照排序的结果从大到小进行绘制，并且将数量较少的三种权限进行求和归并成一种others
    columns = ['internet','camera','location','storage','audio','video','contacts','accounts']
    frame = pd.read_csv('vulnerable.dat',names=columns)
    res = frame.apply(sum, 0)# 等价于frame.sum()
    #对求和进行排序
    sorted_res = res.order(na_last=True, ascending=False, kind='mergesort')
    #pandas的Series的index成员可以获得Series的key，即这里的column，取得前五个，再添加一个others
    sorted_index = [ sorted_res.index[i] for i in range(5) ]
    sorted_index.append('others')
    #值同样也是取得前五个，并把后三个取和当做第六个值加入序列
    sorted_values = [sorted_res.values[i] for i in range(5)]
    others_value = sorted_res.values[5:].sum()
    sorted_values.append(others_value)
    #绘图
    # The slices will be ordered and plotted counter-clockwise.
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    explode = (0.1, 0, 0, 0, 0, 0)
    #绘图，以sorted_values六个数值和sorted_index为标注和指定的颜色绘制饼图
    plt.pie(sorted_values, explode=explode, labels=sorted_index, colors=colors,
    autopct='%1.1f%%', shadow=True, startangle=90)
    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.title('Permission Distribution on apps')#指定标题
    plt.show()
    