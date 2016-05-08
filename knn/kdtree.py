#!/usr/bin/python
# encoding: utf-8

'''
Created on 2015年10月19日

@author: yanruibo
'''

import numpy as np
import math
'''
求两个点的欧式距离
'''
def get_distance(point1, point2):
    sum_value = 0
    
    for i in range(len(point1)):
        sum_value += np.power(point1[i] - point2[i], 2)
    return np.sqrt(sum_value)

'''
用列表实现的有界优先权队列，用来保存通过KD树查找出来的最近邻的k个节点的数据结构
成员变量
list_ : 定长的有序的list
fixed_size:即k，保存最近邻的节点的数量
'''
class BoundedPriorityQueue:
    '''
    构造函数 传入k
    '''
    def __init__(self, size):
        self.list_ = []
        self.fixed_size = size
    '''
    向有界优先权队列中添加元素
    数据格式为：
    data： [point, distance,label]
    
    '''
    def add(self, data):
        # distance = data[1]
        self.list_.append(data)
        #每添加一个数据，都要按第二列也就是distance列从小到大进行排序
        self.list_.sort(key=lambda entry:entry[1])
        #排序之后如果超出了k个就将超出的（距离大的）删掉
        self.resize()
        
    '''
    返回有界优先权队列的大小
    '''
    def size(self):
        return len(self.list_)
    '''
    超出k个元素时将超出的k个删掉
    '''
    def resize(self):
        if(len(self.list_) > self.fixed_size):
            more_number = len(self.list_) - self.fixed_size
            for i in range(more_number):
                self.list_.pop()
    '''
    返回最近的k个
    返回的结果是点的集合和label的集合
    也就是x的集合和y的集合，x[i]与y[i]一一对应
    '''
    def get_knearest(self):
        return [item[0] for item in self.list_],[item[2] for item in self.list_ ]
        #return self.list_
    '''
    得到最近的k个中的最大的距离
    '''
    def get_largest_distance(self):
        if(len(self.list_) <= self.fixed_size):
            return self.list_[len(self.list_) - 1][1]
        else:
            return self.list_[self.fixed_size - 1][1]
    '''
    判断有界优先权队列是否已满
    '''
    def isfull(self):
        return (len(self.list_) == self.fixed_size)
        
class KDTreeNode(object):
    '''
    kd树中的节点结构
    point：点 即训练集中x[i] 
    label:标签，即训练集的y[i]
    left_child:左子树
    right_child：右子树
    '''
    def __init__(self, point, label, left_child, right_child):
        '''
        Constructor
        '''
        self.point = point
        self.label = label
        self.left_child = left_child
        self.right_child = right_child
    '''
    判断是否是叶子节点
    '''
    def is_leaf(self):
        return (self.left_child == None and self.right_child == None)
        
class KDTree():
    '''
    kd树结构
    成员变量：
    root_node：根节点
    在C语言中，一个树就是一个指针，这里也类似
    '''
    def __init__(self, data, labels):
        def build_kdtree(point_list, label_list, depth):
            #递归终止条件，point_list为空
            if not point_list.size:
                return None
            #计算按哪一维排序
            dimension = depth % len(point_list[0])
            # point_list.sort(key=lambda entry:entry[dimension])
            # sorting arrays in numpy by column
            # point_list = np.array(point_list)
            # print point_list
            #将点集也就是x按照计算出来的那一维进行排序
            sorted_index = np.argsort(point_list[:, dimension])  # 按第dimension列排序
            point_list = point_list[sorted_index]
            #因为y得与x一一对应，所以也得将相应的y进行与x相同的排序
            sorted_label_list = []
            for idx in sorted_index:
                sorted_label_list.append(label_list[idx])
            #计算中心点
            median = len(point_list) // 2
            #建树
            node = KDTreeNode(point=point_list[median], label=sorted_label_list[median],
                              left_child=build_kdtree(point_list[0:median], sorted_label_list[0:median], depth + 1),
                              right_child=build_kdtree(point_list[median + 1:], sorted_label_list[median + 1:], depth + 1))
            # print node.point
            return node
        
        self.root_node = build_kdtree(data, labels, 0)
        
    '''
    查找最近邻的k个节点
    主要思路是根据报告中的伪代码进行实现的
    '''
    def search(self, test_point, k):
        '''
        node：当前节点
        test_point：测试节点
        depth：树的深度
        bpq：有界优先权队列，用于保存查询出的k个节点
        '''
        def recursive_search(node, test_point, depth, bpq):
            current_node = node
            if(current_node == None):
                return
            distance = get_distance(current_node.point, test_point)
            bpq.add([current_node.point, distance,current_node.label])
            #计算按照哪一维查询
            dimension = depth % len(test_point)
            #远的那一端
            far_subtree = None
            if(test_point[dimension] < current_node.point[dimension]):
                far_subtree = current_node.right_child
                recursive_search(current_node.left_child, test_point, depth + 1, bpq)
            else:
                far_subtree = current_node.left_child
                recursive_search(current_node.right_child, test_point, depth + 1, bpq)
            if((not bpq.isfull()) or (math.fabs(current_node.point[dimension] - test_point[dimension]) < bpq.get_largest_distance())):
                recursive_search(far_subtree, test_point, depth + 1, bpq)
            return
        #这里递归程序结束#
        if(self.root_node != None):
            bpq = BoundedPriorityQueue(k)
            recursive_search(self.root_node, test_point, 0, bpq)
            return bpq.get_knearest()
        else:
            print "Tree is empty"
            return
    '''
    遍历KD树
    '''
    def print_tree(self):
        def traverse(node):
            current = node
            if(current == None):
                return
            print current.point
            traverse(current.left_child)
            traverse(current.right_child)
            return
        traverse(self.root_node)
        
        
        
