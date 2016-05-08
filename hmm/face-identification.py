#!/usr/bin/python
# encoding: utf-8

'''
Created on Dec 27, 2015

@author: yanruibo
'''

import cv2
import numpy as np
import sys
from mainLBP import LBP
from hmmlearn.hmm import GaussianHMM

# from sklearn.hmm import GaussianHMM

'''
生成训练集
取每个类别的前五张作为训练集，共40个类别，共200张训练图片
'''
def generate_train_set():
    train_imagepathes = []
    train_labels = []
    for i in range(1, 41):
        for j in range(1, 6):
            # './att_faces/s1/1.pgm'
            train_imagepathes.append('./att_faces/s%s/%s.pgm' % (str(i), str(j)))
            train_labels.append(i)  
    # print train_imagepathes
    # print train_labels
    return train_imagepathes, train_labels

'''
生成测试集
取每个类别的后五张作为测试集，共40个类别，共200张测试图片
'''
def generate_test_set():
    test_imagepathes = []
    test_labels = []
    for i in range(1, 41):
        for j in range(6, 11):
            # './att_faces/s1/1.pgm'
            test_imagepathes.append('./att_faces/s%s/%s.pgm' % (str(i), str(j)))
            test_labels.append(i)
    return test_imagepathes, test_labels

'''
对每一张图片生成观测序列，该方法是论文中的方法，
仅仅是把图片每一块的每一行的像素首尾连接形成一个长的向量，
然后长向量按行连接形成矩阵
'''
def generate_observations_original(imagepath, L=10, M=4):
    image = cv2.imread(imagepath)
    # (112,92)
    # print image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Y = gray_image.shape[0]
    # L = 10
    # M = 4
    obs_length = np.floor((Y - L) / (L - M)) + 1
    obs_list = []
    counter = 0
    while(counter < obs_length):
        begin = counter * (L - M)
        end = (counter + 1) * L - counter * M
        sub_mat = gray_image[range(begin, end), :]
        obsith = sub_mat.reshape(sub_mat.shape[0] * sub_mat.shape[1])
        obs_list.append(obsith.tolist())
        counter = counter + 1
    return np.array(obs_list)

'''
对每一张图片生成观测序列，该方法是把图片每一块的图像进行LBP提取特征之后变成36维的向量，然后按行放到一个矩阵中
'''
def generate_observations_lbp(imagepath, L=10, M=4):
    # './att_faces/s1/1.pgm'
    image = cv2.imread(imagepath)
    # (112,92)
    # print image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Y = gray_image.shape[0]
    # L = 10
    # M = 4
    obs_length = np.floor((Y - L) / (L - M)) + 1
    obs_list = []
    counter = 0
    while(counter < obs_length):
        begin = counter * (L - M)
        end = (counter + 1) * L - counter * M
        sub_mat = gray_image[range(begin, end), :]
        #print begin,end,sub_mat.shape
        obsith = LBP(sub_mat)
        obs_list.append(obsith)
        counter = counter + 1
    return np.array(obs_list)

'''
人脸检测过程
'''
def face_identification(n_components=5, L=10, M=4):
    train_imagepathes, train_labels = generate_train_set()
    test_imagepathes, test_labels = generate_test_set()
    #print train_imagepathes
    #print train_labels
    #print test_imagepathes
    #print test_labels
    
    # 保存所有的模型便于后面测试
    models = []
    model_labels = []
    print "begin training models"
    for i in range(len(train_labels)):
        if(i % 5 == 0):
            # 0 5 10 15 : 1 2 3 4
            # 选取同一个人的五张图片训练一个Model
            #print train_imagepathes[i]
            
            X1 = generate_observations_lbp(train_imagepathes[i], L, M);
            X2 = generate_observations_lbp(train_imagepathes[i + 1], L, M);
            X3 = generate_observations_lbp(train_imagepathes[i + 2], L, M);
            X4 = generate_observations_lbp(train_imagepathes[i + 3], L, M);
            X5 = generate_observations_lbp(train_imagepathes[i + 4], L, M);
            # print X1.shape
            X = np.append(X1, X2, axis=0)
            X = np.append(X, X3, axis=0)
            X = np.append(X, X4, axis=0)
            X = np.append(X, X5, axis=0)
            #print X
            # print X.shape
            # model = GaussianHMM(n_components, covariance_type="diag", n_iter=1000).fit([X1, X2, X3, X4, X5])
            # model = GaussianHMM(n_components, covariance_type="diag", n_iter=1000).fit([X])
            model = GaussianHMM(n_components, covariance_type="diag", n_iter=1000).fit(X)
            models.append(model)
            model_labels.append(train_labels[i])
            #print train_labels[i]
            print "training models %s%%" % str(float(i + 5) / len(train_imagepathes) * 100)
    
    # 测试
    
    correct_count = 0
    for j in range(len(test_imagepathes)):
        test_one = generate_observations_lbp(test_imagepathes[j], L, M)
        max_score = - np.inf
        max_label = None
        for k in range(len(models)):
            current_score = models[k].score(test_one)
            
            if(current_score > max_score):
                max_score = current_score
                max_label = model_labels[k]
                
        if(max_label == test_labels[j]):
            print "correct: real", test_labels[j], "predict", max_label
            correct_count += 1
        else:
            print "error: real", test_labels[j], "predict", max_label
             
    accracy = float(correct_count) / len(test_imagepathes)
    print "Accracy is", accracy
    return accracy

if __name__ == '__main__':
    #print generate_observations_lbp("./att_faces/s1/1.pgm", L=10, M=7)
    '''
    fw = open("log.txt","w")
    for states in range(5,9):
        for L in range(8,15):
            for M in range(3,8):
                if(L==M):
                    continue
                accuracy = face_identification(states,L,M)
                fw.write("states:"+str(states)+" L:"+str(L)+" M:"+str(M)+" accuracy:"+str(accuracy)+"\n")
                fw.flush()
    fw.close()
    '''
    face_identification(n_components=7, L=10, M=7)
    # 5 10 4 Accracy is : 0.565
    
