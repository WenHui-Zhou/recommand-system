#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-01-02
# Copyright (C)2021 All rights reserved.
 
from tqdm import tqdm
import  numpy as np
import os
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from model.lib.GCForest import gcForest

global_args = None

class GcForest:
    def __init__(self):
        self.file_root = './data/criteo_ctr/'
        self._init_frame()
        self.config = self._init_param()
    
    def _init_frame(self):
        # 处理数据集
        names = ['label','I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13',
                'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
                 'C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26']
        self.data = pd.read_csv(os.path.join(self.file_root,'tiny.csv'),sep=',',
                               header=None,names=names)
        self.sparse_features = ['C' + str(i) for i in range(1,27)]
        self.dense_features = ['I' + str(i) for i in range(1,14)]
        self.data[self.sparse_features] = self.data[self.sparse_features].fillna('-1')
        self.data[self.dense_features] = self.data[self.dense_features].fillna(0)
        # 离散数据转化为数字，进行编码
        for feat in self.sparse_features:
            le = LabelEncoder()
            self.data[feat] = le.fit_transform(self.data[feat].tolist())
        # 连续数据归一化
        mms = MinMaxScaler(feature_range=(0,1))
        self.data[self.dense_features] = mms.fit_transform(self.data[self.dense_features])

    def _init_param(self):
        # 定义网络参数，以及训练的时候用到的参数
        parser = argparse.ArgumentParser()
        parser.add_argument('algrithom',type=str,default="gcforest",help = 'the pratice method')
        global global_args
        global_args = parser.parse_args()

    def _build_train_model(self):
        print('build model...')
        train,test = train_test_split(self.data,test_size=0.2)
#        train_X = train[self.dense_features + self.sparse_features].values
        train_X = train[self.dense_features].values
        train_X = np.hstack((train_X,train[self.sparse_features].values.astype('int32')))
#        train_X = pd.DataFrame(train_X, columns = self.dense_features + self.sparse_features)
        train_Y = train['label'].values.astype('int32')
#        train_Y = pd.DataFrame(train_Y, columns = ['label'])
#        test_X = test[self.dense_features+self.sparse_features].values
        test_X = test[self.dense_features].values
        test_X = np.hstack((test_X,test[self.sparse_features].values.astype('int32')))
#        test_X = pd.DataFrame(test_X, columns = self.dense_features + self.sparse_features)
        test_Y = test['label'].values.astype('int32')
#        test_Y = pd.DataFrame(test_Y, columns = ['label'])

        print('train begin...')
        gc = gcForest(shape_1X=39, window=6, tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7)
        gc.fit(train_X,train_Y)
        print('model evaluate...')
        y_pred = gc.predict(test_X)
        print('a part of predict result: ')
        print(y_pred[:10])
        acc = accuracy_score(test_Y,y_pred)
        print('Test accuracy of the gcFroest = {:.2f}%'.format(acc*100))


        # 预测结果的好坏



    def calculate(self,*args):
        self._build_train_model()
        return 'done'
