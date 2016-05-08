#!/usr/bin/python
# encoding: utf-8

'''
Created on Nov 28, 2015

@author: yanruibo
'''
import logRegres
import numpy as np
if __name__ == '__main__':
    dataArr,labelMat = logRegres.loadDataSet()
    #weights = logRegres.gradAscent(dataArr, labelMat)
    weights = logRegres.stocGradAscent0(np.array(dataArr), labelMat)
    print weights
    #logRegres.plotBestFit(weights.getA())
    logRegres.plotBestFit(weights)