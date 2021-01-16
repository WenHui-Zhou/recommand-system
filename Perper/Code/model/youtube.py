#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2020-12-29
# Copyright (C)2020 All rights reserved.

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import Embedding,Dropout,Dense,Input,Layer
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
import argparse
from tqdm import tqdm
import os

global_args = None

class youtube_recall_model(tf.keras.Model):

    def __init__(self,feature_columns,hidden_units=[256,128,64],
                 activation='relu',embed_reg = 1e-4):
        super(toutube_recall_model,self).__init__()
        self.dense_feature_columns,self.sparse_feature_columns = feature_columns
        # 为每一个sparse向量提供一个embedding层
        self.embed_layers = {
            'embed_' + str(i): Embedding(
                input_dim = feat['feat_num'],
                input_length = 1,
                output_dim = feat['embed_dim'],
                embeddings_initializer = 'random_uniform',
                embeddings_regularizer = l2(embed_reg)
            ) for i ,feat in enumerate(self.sparse_feature_columns)
        }
        self.dnn = DNN(hidden_units,activation,dnn_dropout)

    def call(self,inputs,**kwargs):
        dense_inputs,sparse_inputs = inputs
        #将每一个vector输入到各自的dense结构中
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:,i])
                                 for i in range(sparse_inputs.shape[1])],axis=-1)
        stack = tf.concat([dense_inputs,sparse_embed],axis=-1)
        outputs = self.dnn(stack) # (input[0],64)
        return outputs

    def summary(self,**kwargs):
        dense_inputs = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype = tf.float32)
        sparse_inputs = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype = tf.float32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs],outputs = self.call([dense_inputs, sparse_inputs])).summary()

class Youtube:
    def __init__(self):
        super(Youtube,self).__init__()
        self.file_root = './data/ml-latest-small/'
        self._init_param()
        #item的个数以及dimension 训练集，验证集，测试集
        self.item_fea_col, self.train, self.val, self.test = self._init_frame()

    def _init_param(self):
        # 定义网络参数，以及训练的时候用到的参数
        parser = argparse.ArgumentParser()
        parser.add_argument('algrithom',type=str,default="youtube",help = 'the pratice method')
        parser.add_argument('--embed_dim',type=int,default=8,help = 'the len of feature embeding')
        parser.add_argument('--learning_rate',type=float,default=0.001,help = 'the learning rate of the model')
        parser.add_argument('--epochs',type=int,default=100,help = 'how many time we train the model')
        parser.add_argument('--batch_size',type=int,default=4096,help = 'a batch data size')
        parser.add_argument('--test_size',type=float,default=0.2,help = 'the rate of test set / all data')
        parser.add_argument('--dnn_dropout',type=float,default=0.5,help = 'v l2 normalization penalty')
        parser.add_argument('--hidden_units',type=list,default=[256,128,64],help = 'deep forward network layer')
        parser.add_argument('--k',type=int,default=10,help = 'fm latent vector size')
        global global_args
        global_args = parser.parse_args()
    
    def _init_frame(self):
        # 处理数据集
        print('prepare dataset...')
        # ratings: 
        # userId,movieId,rating,timestamp
        # 1,1,4.0,964982703
        data_df = pd.read_csv(os.path.join(self.file_root,'ratings.csv'),sep=',',header=0)
        import pdb
        pdb.set_trace()

        train,test = train_test_split(data_df,test_size=global_args.test_size)
        train_X = [train[dense_features].values, train[sparse_features].values.astype('int32')]
        train_Y = train['label'].values.astype('int32')
        test_X = [test[dense_features].values, test[sparse_features].values.astype('int32')]
        test_Y = test['label'].values.astype('int32')

        return (train_X, train_Y), (test_X, test_Y),feature_columns

    def calculate(self,*args):
        # prepare data
        train_X,train_Y= self.trainset
        test_X,test_Y = self.testset
        # build model
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            self.model = DeepFM_model(self.feature_columns)
            self.model.summary()
            # ======== model checkpoint === 
            # check_path = '../save/fm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
            # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
            #                                                 verbose=1, period=5)
            # ======== compile
            self.model.compile(loss = binary_crossentropy,optimizer=Adam(learning_rate = global_args.learning_rate), \
                         metrics = [AUC()])
        # ======== model fit 
        self.model.fit(
            train_X,
            train_Y,
            epochs = global_args.epochs,
            callbacks = [EarlyStopping(monitor='val_loss',patience=1,restore_best_weights=True)],
            batch_size = global_args.batch_size,
            validation_split=0.1
        )
        # ======== evaluate
        print('test AUC: %f' % self.model.evaluate(test_X,test_Y,batch_size = global_args.batch_size)[1])
        return 'done'
