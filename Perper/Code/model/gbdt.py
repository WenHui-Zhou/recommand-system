#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2020-12-25
# Copyright (C)2020 All rights reserved.

from __future__ import print_function
import tensorflow as tf
from tqdm import tqdm
import os
import copy
import argparse
from tensorflow.keras.datasets import boston_housing

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

global_args = None

class GBDT:
    def __init__(self):
        self._init_frame()
        self._init_param()
    
    def _init_frame(self):
        # 处理数据集
        (self.x_train,self.y_train),(self.x_test,self.y_test) = boston_housing.load_data()

        # 对数据大于或小于23进行二分
        def to_binary_class(y):
            for i,label in enumerate(y):
                if label >=2.3:
                    y[i] = 1
                else:
                    y[i] = 0
            return y
        self.y_train_bin = to_binary_class(copy.deepcopy(self.y_train))
        self.y_test_bin = to_binary_class(copy.deepcopy(self.y_test))

    def _init_param(self):
        # 定义网络参数，以及训练的时候用到的参数
        parser = argparse.ArgumentParser()
        parser.add_argument('algrithom',type=str,default="gbdt",help = 'the pratice method')
        parser.add_argument('--num_classes',type=int,default=2,help = 'total classes')
        parser.add_argument('--num_features',type=int,default=13,help = 'feature size')
        parser.add_argument('--learning_rate',type=float,default=1.0,help = 'the learning rate of the model')
        parser.add_argument('--epochs',type=int,default=2000,help = 'how many time we train the model')
        parser.add_argument('--batch_size',type=int,default=256,help = 'a batch data size')
        parser.add_argument('--test_size',type=float,default=0.2,help = 'the rate of test set / all data')
        parser.add_argument('--l1_reg',type=float,default=0.0,help = 'l1 normalization penalty')
        parser.add_argument('--l2_reg',type=float,default=0.1,help = 'l2 normalization penalty')

        parser.add_argument('--num_batches_per_layer',type=int,default=1000,help = 'gbdt parameters')
        parser.add_argument('--num_trees',type=int,default=1000,help = 'gbdt parameters')
        parser.add_argument('--max_depth',type=int,default=4,help = 'gbdt parameters')
        global global_args
        global_args = parser.parse_args()


    def _build_train_model(self):
        print('build model...')
        train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x = {'x':self.x_train}, y =self.y_train_bin,batch_size = global_args.batch_size,
            num_epochs=None,shuffle=True)
        test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x = {'x':self.x_test}, y =self.y_test_bin,batch_size = global_args.batch_size,
            num_epochs=1,shuffle=False)
        test_train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x = {'x':self.x_train}, y =self.y_train_bin,batch_size = global_args.batch_size,
            num_epochs=1,shuffle=False)
        feature_columns = [tf.feature_column.numeric_column(key='x',shape=(global_args.num_features,))]
        gbdt_classifer = tf.estimator.BoostedTreesClassifier(
            n_batches_per_layer = global_args.num_batches_per_layer,
            feature_columns = feature_columns,
            n_classes = global_args.num_classes,
            learning_rate = global_args.learning_rate,
            n_trees = global_args.num_trees,
            max_depth = global_args.max_depth,
            l1_regularization = global_args.l1_reg,
            l2_regularization = global_args.l2_reg
        )
        print('train begin...')
        gbdt_classifer.train(train_input_fn,max_steps = global_args.epochs)

        print('model evaluate...')
        gbdt_classifer.evaluate(test_train_input_fn)
        gbdt_classifer.evaluate(test_input_fn)

    def calculate(self,*args):
        self._build_train_model()
        return 'done'
