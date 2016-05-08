#!/usr/bin/python
# encoding: utf-8

'''
Created on Nov 6, 2015

@author: yanruibo
'''

import numpy as np

if __name__ == '__main__':
    data = np.loadtxt(fname='unformatted-data.txt', dtype=np.int64, delimiter=',')
    data = np.delete(data, [0], axis=1)
    for i in range(len(data)):
        if(data[i,0]==2):
            data[i,0]=0
        else:
            data[i,0]=1
    print data
    
    trainMat = np.delete(data,[0],axis=1)
    classes = data[:,0].reshape((len(data[:,0]),1))
    resultMat = np.append(trainMat, classes, axis=1)
    print resultMat
    
    #np.savetxt("formatted-data.txt", resultMat,fmt='%d')
    
    loadData = np.loadtxt("formatted-data.txt")
    
    print "loadData",loadData

    