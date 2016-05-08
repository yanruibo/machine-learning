#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
Created on Oct 26, 2015

@author: yanruibo
'''
import numpy as np
if __name__ == '__main__':
    TPs = np.zeros([10])
    print TPs.shape
    print TPs
    print TPs[0]
    ft = open("test.txt", "a")
    ft.write("k : " + str(TPs) + "\n")
    ft.close()