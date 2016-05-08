#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
Created on Oct 31, 2015

@author: yanruibo

用feedparser抓取的rss数据集测试SBC算法
因为前面测试SBC是从所有数据中随机选择20个数据做为测试集，剩下的作为训练集
这里只测试了一次，因为rss数据集的特征多达500多个，SBC的计算复杂度较高

'''
import feedparser
import bayes
import time
if __name__ == '__main__':
    
    startTimeStamp = time.time()
    
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    
    errorRate = bayes.SBC(ny, sf)
    
    endTimeStamp = time.time()
    total_time = endTimeStamp - startTimeStamp
    ft = open("test_sbc_rss.txt", "a")
    ft.write("Normal Total Time : " + str(total_time) + "\n")
    ft.write("errorRate" + str(errorRate) + "\n")
    ft.close()