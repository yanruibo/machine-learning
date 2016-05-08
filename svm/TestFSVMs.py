#!/usr/bin/python
# encoding: utf-8

'''
Created on Dec 2, 2015

@author: yanruibo
'''
import FSVMs
'''
测试四种核函数的FSVM
'''
if __name__ == '__main__':
    #FSVMs.testDigits(('lin',))
    #FSVMs.testDigits(('rbf',2))
    FSVMs.testDigits(('polynomial',100))
    #FSVMs.testDigits(('sigmoid',0.00125,0.4))
