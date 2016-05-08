#!/usr/bin/python
# encoding: utf-8

'''
Created on 2015年10月19日

@author: yanruibo
'''
from kdtree import KDTree

if __name__ == '__main__':

    data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    point = [7,0]
            
    tree = KDTree.construct_from_data(data)
    #tree.print_tree()
    
    nearest = tree.query(point, t=2) # find nearest 4 points
    print nearest