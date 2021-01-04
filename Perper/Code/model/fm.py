#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2020-12-29
# Copyright (C)2020 All rights reserved.

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import AUC
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
import argparse
from tqdm import tqdm
import os

global_args = None

class FM_Layer(Layer):
    # 需要重写__init__,build(),call() 方法
    def __init__(self,feature_columns):
        super(FM_Layer,self).__init__()
        self.dense_feature_columns,self.sparse_feature_columns = feature_columns
        # 将所有特征直接拼接起来，作为输入，输入的总长度
#        self.feature_length = sum([feat['feat_num'] for feat in self.sparse_feature_columns]) + \
        self.feature_length = sum([feat['embed_dim'] for feat in self.sparse_feature_columns]) + \
                len(self.dense_feature_columns)

    def build(self,input_shape):
        self.w0 = self.add_weight(name='w0',shape=(1,),\
                                  initializer=tf.zeros_initializer(),trainable=True)
        self.w = self.add_weight(name='w',shape=(self.feature_length, 1), \
                                 initializer = tf.random_normal_initializer(), \
                                 regularizer=l2(global_args.w_reg),trainable=True)
        self.V = self.add_weight(name='V',shape=(global_args.latent_size,self.feature_length), \
                                 initializer = tf.random_normal_initializer(), \
                                 regularizer=l2(global_args.v_reg),trainable=True)

    def call(self,inputs,**kwargs):
        dense_inputs,sparse_inputs = inputs
        # one-hot embedding 
        sparse_inputs = tf.concat(
            [tf.one_hot(tf.dtypes.cast(sparse_inputs[:,i],tf.int32), depth=self.sparse_feature_columns[i]['embed_dim']) \
            for i in range(sparse_inputs.shape[1])],axis=1)
        stack = tf.concat([dense_inputs, sparse_inputs],axis=1)
        # first order
        first_order = self.w0 + tf.matmul(stack,self.w)
        # second_order
        second_order = 0.5 * tf.reduce_sum(
            tf.pow(tf.matmul(stack,tf.transpose(self.V)),2) -
            tf.matmul(tf.pow(stack,2) , tf.pow(tf.transpose(self.V), 2)),axis=1,keepdims=True)
        outputs = first_order + second_order
        return outputs

class FM_model(tf.keras.Model):

    def __init__(self,feature_columns):
        super(FM_model,self).__init__()
        self.dense_feature_columns,self.sparse_feature_columns = feature_columns
        self.fm = FM_Layer(feature_columns)

    def call(self,inputs,**kwargs):
        fm_outputs = self.fm(inputs)
        outputs = tf.nn.sigmoid(fm_outputs)
        return outputs

    def summary(self,**kwargs):
        dense_inputs = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype = tf.float32)
        sparse_inputs = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype = tf.float32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs],outputs = self.call([dense_inputs, sparse_inputs])).summary()

class FM:
    def __init__(self):
        super(FM,self).__init__()
        self.file_root = './data/criteo_ctr/'
        self._init_param()
        # 训练集，测试集，dense column，sparse column
        self.trainset,self.testset,self.feature_columns = self._init_frame()
        self.dense_feature_columns,self.sparse_feature_columns = self.feature_columns

    def _init_param(self):
        # 定义网络参数，以及训练的时候用到的参数
        parser = argparse.ArgumentParser()
        parser.add_argument('algrithom',type=str,default="FM",help = 'the pratice method')
        parser.add_argument('--embed_dim',type=int,default=8,help = 'the len of feature embeding')
        parser.add_argument('--method',type=str,default="",help = 'the test method')
        parser.add_argument('--learning_rate',type=float,default=0.001,help = 'the learning rate of the model')
        parser.add_argument('--epochs',type=int,default=100,help = 'how many time we train the model')
        parser.add_argument('--batch_size',type=int,default=256,help = 'a batch data size')
        parser.add_argument('--test_size',type=float,default=0.2,help = 'the rate of test set / all data')
        parser.add_argument('--w_reg',type=float,default=1e-4,help = 'w l2 normalization penalty')
        parser.add_argument('--v_reg',type=float,default=1e-4,help = 'v l2 normalization penalty')
        parser.add_argument('--latent_size',type=int,default=10,help = 'the latent vector size')
        global global_args
        global_args = parser.parse_args()
    
    def _init_frame(self):
        # 处理数据集
        names = ['label','I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13',
                'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
                 'C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26']
        data_df = pd.read_csv(os.path.join(self.file_root,'tiny.csv'),sep=',',
                               header=None,names=names)
        sparse_features = ['C' + str(i) for i in range(1,27)]
        dense_features = ['I' + str(i) for i in range(1,14)]
        # 数据记录填补空记录
#        data_df.dropna(axis=0, how='any', inplace=True)
        data_df[sparse_features] = data_df[sparse_features].fillna('-1')
        data_df[dense_features] = data_df[dense_features].fillna(0)

        # 对离散数据进行labelEncoder处理
        # LabelEncoder可以将标签分配一个0—n_classes-1之间的编码,将各种标签分配一个可数的连续编号
        for feat in sparse_features:
            le = LabelEncoder()
            data_df[feat] = le.fit_transform(data_df[feat].tolist())
            data_df[feat] = data_df[feat].astype(int)

        # 对dense的数据进行归一化处理
        dense_features = [feat for feat in data_df.columns if feat not in sparse_features + ['label']]
        mms = MinMaxScaler(feature_range=(0,1))
        data_df[dense_features] = mms.fit_transform(data_df[dense_features])
        feature_columns = [[self._denseFeature(feat) for feat in dense_features]] + \
                [[self._sparseFeature(feat,len(data_df[feat].unique()),embed_dim=global_args.embed_dim) \
                 for feat in sparse_features]]

        train,test = train_test_split(data_df,test_size=global_args.test_size)
        train_X = [train[dense_features].values, train[sparse_features].values.astype('int32')]
        train_Y = train['label'].values.astype('int32')
        test_X = [test[dense_features].values, test[sparse_features].values.astype('int32')]
        test_Y = test['label'].values.astype('int32')

        return (train_X, train_Y), (test_X, test_Y),feature_columns

    def _denseFeature(self,feat):
        return {'feat':feat}

    def _sparseFeature(self,feat, feat_num, embed_dim = 4):
        # feat：特征名字
        # feat_num：这种特征有多少种值
        # embed的尺寸是多少
        return {'feat':feat,'feat_num':feat_num,'embed_dim':embed_dim}

    def calculate(self,*args):
        self.model = FM_model(self.feature_columns)
        self.model.summary()
        # ======== model checkpoint === 
        # check_path = '../save/fm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
        # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
        #                                                 verbose=1, period=5)
        # ======== compile
        self.model.compile(loss = binary_crossentropy,optimizer=Adam(learning_rate = global_args.learning_rate), \
                     metrics = [AUC()])
        # ======== model fit 
        train_X,train_Y= self.trainset
        test_X,test_Y = self.testset
        self.model.fit(
            train_X,
            train_Y,
            epochs = global_args.epochs,
            batch_size = global_args.batch_size,
            validation_split=0.1
        )
        # ======== evaluate
        print('test AUC: %f' % self.model.evaluate(test_X,test_Y)[1])
        return 'done'
