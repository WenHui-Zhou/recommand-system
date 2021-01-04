#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-01-02
# Copyright (C)2021 All rights reserved.
 
import tensorflow as tf
from tqdm import tqdm
import  numpy as np
import os
import argparse
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn import metrics

global_args = None

class GBDTLR:
    def __init__(self):
        self.file_root = './data/criteo_ctr/'
        self._init_frame()
        self._init_param()
    
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
        parser.add_argument('algrithom',type=str,default="gbdt_lr",help = 'the pratice method')
        parser.add_argument('--num_estimators',type=int,default=32,help = 'the num of decision trees')
        parser.add_argument('--num_leaves',type=int,default=64,help = 'the num of tree leaves ')
        parser.add_argument('--learning_rate',type=float,default=0.05,help = 'the learning rate of the model')
        parser.add_argument('--test_size',type=float,default=0.2,help = 'the rate of test set / all data')
        parser.add_argument('--embed_dim',type=float,default=0.2,help = 'the rate of test set / all data')
        global global_args
        global_args = parser.parse_args()


    def _build_train_model(self):
        print('build model...')
        train,test = train_test_split(self.data,test_size=global_args.test_size)
#        train_X = train[self.dense_features + self.sparse_features].values
#        sparse_inputs = tf.concat(
#            [tf.one_hot(tf.dtypes.cast(sparse_inputs[:,i],tf.int32), depth=self.sparse_feature_columns[i]['embed_dim']) \
#            for i in range(sparse_inputs.shape[1])],axis=1)
        train_X = train[self.dense_features].values
        train_X = np.hstack((train_X,train[self.sparse_features].values.astype('int32')))
        train_X = pd.DataFrame(train_X, columns = self.dense_features + self.sparse_features)
        train_Y = train['label'].values.astype('int32')
        train_Y = pd.DataFrame(train_Y, columns = ['label'])
#        test_X = test[self.dense_features+self.sparse_features].values
        test_X = test[self.dense_features].values
        test_X = np.hstack((test_X,test[self.sparse_features].values.astype('int32')))
        test_X = pd.DataFrame(test_X, columns = self.dense_features + self.sparse_features)
        test_Y = test['label'].values.astype('int32')
        test_Y = pd.DataFrame(test_Y, columns = ['label'])

        print('train begin...')
        # 开始训练gbdt，使用100棵树，每棵树64个节点
        # 将parse和dense数据同时输入到网络中
        gbdt_model = lgb.LGBMRegressor(objective='binary',
                                       subsample=0.8,
                                       min_child_weight=0.5,
                                       colsample_bytree=0.7,
                                       num_leaves=global_args.num_leaves,
                                       learning_rate = global_args.learning_rate,
                                       n_estimators = global_args.num_estimators,
                                       random_state = 2021)
        gbdt_model.fit(train_X,train_Y,
                      eval_set = [(train_X,train_Y),(test_X,test_Y)],
                      eval_names = ['train','val'],
                      eval_metric = 'binary_logloss',
                      verbose=0)

        # 得到每一条训练数据落在每颗树的叶子节点上,预测结果返回叶子节点的序号(0-63)
        gbdt_feat_train = gbdt_model.predict(train_X,pred_leaf = True)
        gbdt_feat_test = gbdt_model.predict(test_X,pred_leaf = True)
        # 将32课树的序号构成DF,方便one-hot
        gbdt_feat_name = ['gbdt_tree_' + str(i) for i in range(global_args.num_estimators)]
        df_train_gbdt_feats = pd.DataFrame(gbdt_feat_train,columns = gbdt_feat_name)
        df_test_gbdt_feats =  pd.DataFrame(gbdt_feat_test,columns = gbdt_feat_name)
        train_len = df_train_gbdt_feats.shape[0]
        result = pd.concat([df_train_gbdt_feats,df_test_gbdt_feats])

        for feat in gbdt_feat_name:
            le = LabelEncoder()
            result[feat] = le.fit_transform(result[feat].tolist())

        train_data = result[:train_len]
        test_data = result[train_len:]
        
        # lr 模型
        lr = LogisticRegression()
        lr.fit(train_data,train_Y)
        tr_logloss = log_loss(train_Y,lr.predict_proba(train_data)[:,1])
        val_logloss = log_loss(test_Y,lr.predict_proba(test_data)[:,1])
        print("train_loss:{} | val_loss: {}".format(tr_logloss,val_logloss))

        print('model evaluate...')
        # 预测结果的好坏
        y_pred = lr.predict_proba(test_data)[:,1]
        score = metrics.roc_auc_score(test_Y.values, y_pred)
        print(score)



    def calculate(self,*args):
        self._build_train_model()
        return 'done'
