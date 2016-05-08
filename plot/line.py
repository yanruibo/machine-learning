#! /usr/bin/env python
#coding=utf-8

'''
Created on 2015年9月29日
@author: yanruibo
折线图的绘制
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
    #声明列名 
    columns = ['internet','camera','location','storage','audio','video','contacts','accounts']
    #用pandas库加载数据文件，非常神奇的是pandas库可以只能识别列与列之间的分隔符，逗号不用制定就可以识别
    frame = pd.read_csv('vulnerable.dat',names=columns)
    #对加载到内存的frame进行各列的求和
    res = frame.apply(sum, 0)# 等价于frame.sum()
    #对求和的结果进行排序
    sorted_res = res.order(na_last=True, ascending=False, kind='mergesort')
    #下面是绘图程序
    #定义x为[1,9)即[1,8]
    x = [i for i in range(1,9)]
    #定义y为各列的求和，即有多少的数量申请了某一个特定的权限，这个数量是在总的382个app中有多少app申请了该权限，
    #但是这八个数的总和不等于382，因为基本上一个app会申请多个权限。
    y = [ res.values[i] for i in range(8)]
    #获得子绘图
    fig,ax = plt.subplots()
    #定义x轴的显示名称
    plt.xticks(x, columns)
    #根据x y序列的值绘图
    ax.plot(x, y, marker='o')
    #在折线图的点上标示出具体的数量
    for i in range(8):
        #控制位置的显示 这儿控制的不太好
        if (y[i] > 50):        
            yloc = y[i]*1.05
            align = 'top'
        else:
            yloc = y[i]
            align = 'bottom'
        #标示出每一个点的具体的y值
        ax.text(x[i], y[i]*1.05, str(y[i]) , horizontalalignment='center',verticalalignment=align, color='black', weight='bold')
    #设置x y轴的范围 #ax.set_xlim((0.0, 9.0))
    ax.set_ylim((0.0, 450.0))
    plt.show()
    
    