#!/usr/bin/python
# encoding: utf-8

'''
Created on Dec 2, 2015

@author: yanruibo
'''
import svmMLiA

'''
测试四种核函数的SVM
'''
if __name__ == '__main__':
    svmMLiA.testDigits(('lin',))
    #svmMLiA.testDigits(('rbf',50))
    #svmMLiA.testDigits(('polynomial',100))
    #svmMLiA.testDigits(('sigmoid',0.00125,0.4))
    