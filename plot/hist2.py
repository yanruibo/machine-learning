#! /usr/bin/env python
#coding=utf-8

'''
Created on 2015年9月29日
@author: yanruibo
直方图的绘制
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
    #加载数据和折线图类似
    columns = ['internet','camera','location','storage','audio','video','contacts','accounts']
    frame = pd.read_csv('vulnerable.dat',names=columns)
    res = frame.apply(sum, 0)# 等价于frame.sum()
    sorted_res = res.order(na_last=True, ascending=False, kind='mergesort')
    print type(res.values)
    print res.values
    #画折线图
    n_groups = 8 #设置八个分组
    permission_data = res.values #设置直方图的数据
    
    fig, ax = plt.subplots()#获得子绘图
    index = np.arange(n_groups)#获得x轴的值[0 1 2 3 4 5 6 7]
    bar_width = 1 #设置直方图的宽度
    #绘制直方图，以index为x轴值，permission_data为y轴值，bar_width为宽度，颜色为蓝色绘制直方图
    rects = plt.bar(index, permission_data, bar_width, color='b')
    plt.xlabel('Permissions')#指定x轴的含义
    plt.ylabel('Number of Apps')#指定y轴的含义
    plt.title('Permission Distribution on apps')#指定标题
    #plt.xticks(index, columns)#制定x轴上的分隔
    plt.ylim(0,400)#设置y的范围
    plt.tight_layout()#设置紧凑格式
    #下面的代码是控制直方图上面的y轴数值的显示
    for rect in rects:
        #直方图每个方图的高度
        height = int(rect.get_height())
        #直方图上面显示的标注即y值
        noteStr = str(height)
        #计算显示标注的y值和依靠上面显示还是依靠下面显示
        if (height < 1):        # The bars aren't wide enough to print the ranking inside
            yloc = height + 1   # Shift the text to the right side of the right edge
            align = 'top'
        else:
            yloc = height
            align = 'bottom'
        '''
        yloc = height*1.05
        align = 'top'
        '''
        # Center the text horizontally in the bar
        #指定标注的x值
        xloc = rect.get_x()+rect.get_width()/2.0
        #绘制标注 在xloc yloc坐标处绘制noteStr的值，水平居中对其，垂直对齐由上面设置的align控制，字体为黑色 粗体
        ax.text(xloc, yloc, noteStr, horizontalalignment='center',verticalalignment=align, color='black', weight='bold')
    plt.show()    
    
    