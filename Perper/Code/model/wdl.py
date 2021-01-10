#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2020-12-29
# Copyright (C)2020 All rights reserved.

import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Embedding,Concatenate,Dropout,Input
from tensorflow.keras.callbacks import EarlyStopping
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

class Linear(Layer):
    # 需要重写__init__,build(),call() 方法
    def __init__(self):
        super(Linear,self).__init__()
        self.dense = Dense(1, activation=None)

    def call(self,inputs,**kwargs):
        result = self.dense(inputs)
        return result

class DNN(Layer):

    def __init__(self,hidden_units,activation='relu',dropout=0.5):
        super(DNN,self).__init__()
        self.dnn_network = [Dense(units = unit,activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self,inputs,**kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x


class WDL_model(tf.keras.Model):

    def __init__(self,feature_columns,hidden_units,activation='relu',
                dnn_dropout=0.5, embed_reg=1e-4):
        super(WDL_model,self).__init__()
        self.dense_feature_columns,self.sparse_feature_columns = feature_columns
        # 此处对原始的数据进行embedding，使其变成稠密的向量
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim = feat['feat_num'],
                                        input_length = 1,
                                        output_dim = feat['embed_dim'],
                                        embeddings_initializer = 'random_uniform',
                                        embeddings_regularizer = l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.dnn_network = DNN(hidden_units,activation,dnn_dropout)
        self.linear = Linear()
        self.final_dense = Dense(1,activation=None)

    def call(self,inputs,**kwargs):
        dense_inputs, sparse_inputs = inputs
        # 将所有sparse数据都经过embedding变成dense数据
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:,i])
                                for i in range(sparse_inputs.shape[1])],axis = 1)
        x = tf.concat([sparse_embed, dense_inputs], axis=-1)
        # 下面看的很清楚，dense的数据输入到linear中，将sparse和dense融合之后输入到deep中
        # 书上说的是，应该先对稀疏的数据做交叉，然后输入到linear上，这个感觉待会可以改改看
        # wide 
        wide_out = self.linear(dense_inputs)
        #deep
        deep_out = self.dnn_network(x)
        deep_out = self.final_dense(deep_out)
        outputs = tf.nn.sigmoid(0.5*wide_out + 0.5*deep_out)
        return outputs

    def summary(self,**kwargs):
        dense_inputs = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype = tf.float32)
        sparse_inputs = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype = tf.float32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs],outputs = self.call([dense_inputs, sparse_inputs])).summary()


class WDL:
    def __init__(self):
        super(WDL,self).__init__()
        self.file_root = './data/criteo_ctr/'
        self._init_param()
        # 训练集，测试集，dense column，sparse column
        self.trainset,self.testset,self.feature_columns = self._init_frame()
        self.dense_feature_columns,self.sparse_feature_columns = self.feature_columns

    def _init_param(self):
        # 定义网络参数，以及训练的时候用到的参数
        parser = argparse.ArgumentParser()
        parser.add_argument('algrithom',type=str,default="WDL",help = 'the pratice method')
        parser.add_argument('--embed_dim',type=int,default=8,help = 'the len of feature embeding')
        parser.add_argument('--learning_rate',type=float,default=0.001,help = 'the learning rate of the model')
        parser.add_argument('--epochs',type=int,default=40,help = 'how many time we train the model')
        parser.add_argument('--batch_size',type=int,default=4096,help = 'a batch data size')
        parser.add_argument('--test_size',type=float,default=0.2,help = 'the rate of test set / all data')
        parser.add_argument('--sample_num',type=int,default=5000000,help = 'v l2 normalization penalty')
        parser.add_argument('--dnn_dropout',type=float,default=0.5,help = 'v l2 normalization penalty')
        parser.add_argument('--hidden_units',type=list,default=[256,128,64],help = 'v l2 normalization penalty')
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
        train_X,train_Y= self.trainset
        test_X,test_Y = self.testset
        # build model
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            self.model = WDL_model(self.feature_columns,hidden_units=global_args.hidden_units,\
                                   dnn_dropout= global_args.dnn_dropout)
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
            callbacks=[EarlyStopping(monitor='val_loss',patience=1,restore_best_weights=True)],
            batch_size = global_args.batch_size,
            validation_split=0.1
        )
        # ======== evaluate
        print('test AUC: %f' % self.model.evaluate(test_X,test_Y,batch_size = global_args.batch_size)[1])
        return 'done'
