#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-01-07
# Copyright (C)2020 All rights reserved.

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,Embedding,Input
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os
import random

from model.lib.dnn_evaluate import evaluate_model

global_args = None

class DNN_model(tf.keras.Model):

    def __init__(self,item_feat_col,maxlen=40,hidden_units=128,activation='relu',embed_reg=1e-6):
        super(DNN_model,self).__init__()
        self.maxlen = maxlen
        self.item_feat_col = item_feat_col
        embed_dim = self.item_feat_col['embed_dim']
        # item embedding 
        self.item_embedding = Embedding(
            input_dim = self.item_feat_col['feat_num'],
            input_length = 1,
            output_dim = self.item_feat_col['embed_dim'],
            mask_zero = True,
            embeddings_initializer = 'random_normal',
            embeddings_regularizer = l2(embed_reg)
        )
        self.dense = Dense(hidden_units,activation=activation)
        self.dense2 = Dense(embed_dim,activation=activation)

    def call(self,inputs,**kwargs):
        seq_inputs, item_inputs = inputs
        mask = tf.cast(tf.not_equal(seq_inputs,0),dtype = tf.float32)
        seq_embed = self.item_embedding(seq_inputs)
        seq_embed *= tf.expand_dims(mask,axis = -1)
        seq_embed_mean = tf.reduce_mean(seq_embed,axis=1)
        user_embed = self.dense(seq_embed_mean)
        user_embed = self.dense2(user_embed)

        item_embed = self.item_embedding(tf.squeeze(item_inputs,axis=-1))
        outputs = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(user_embed,item_embed),axis=1,keepdims=True))
        return outputs

    def summary(self,**kwargs):
        seq_inputs = tf.keras.Input(shape=(self.maxlen,), dtype = tf.int32)
        item_inputs = tf.keras.Input(shape=(1,), dtype = tf.int32)
        tf.keras.Model(inputs=[seq_inputs, item_inputs],outputs = self.call([seq_inputs, item_inputs])).summary()

class DNN:
    def __init__(self):
        super(DNN,self).__init__()
        self.file_root = './data/ml-latest-small/'
        self._init_param()
        #item的个数以及dimension 训练集，验证集，测试集
        self.item_fea_col, self.train, self.val, self.test = self._init_frame()

    def _init_param(self):
        # 定义网络参数，以及训练的时候用到的参数
        parser = argparse.ArgumentParser()
        parser.add_argument('algrithom',type=str,default="dnn",help = 'the pratice method')
        parser.add_argument('--activation',type=str,default="relu",help = 'the pratice method')
        parser.add_argument('--learning_rate',type=float,default=0.001,help = 'the learning rate of the model')
        parser.add_argument('--epochs',type=int,default=30,help = 'how many time we train the model')
        parser.add_argument('--batch_size',type=int,default=512,help = 'a batch data size')
        parser.add_argument('--embed_reg',type=float,default=1e-6,help = 'v l2 normalization penalty')
        parser.add_argument('--hidden_unit',type=int,default=256,help = 'the latent vector size')
        parser.add_argument('--pos_score',type=int,default=2,help = 'the positive score')
        parser.add_argument('--embed_dim',type=int,default=64,help = 'item embedding size')
        parser.add_argument('--maxlen',type=int,default=100,help = 'the max length of the vector')
        parser.add_argument('--K',type=int,default=10,help = 'parameter for evaluate')
        global global_args
        global_args = parser.parse_args()
    
    def _init_frame(self):
        # 处理数据集
        # userId,movieId,rating,timestamp
        print('prepare dataset...')
        data_df = pd.read_csv(os.path.join(self.file_root,'ratings.csv'),sep=',',header=0)
        """
        下面产生数据的思路是原有样本均为正样本，随机生成样本为负样本
        因此我们剔除评分比较低的样本，将剩余样本作为正样本进行训练
        """
        data_df = data_df[data_df.rating >= global_args.pos_score]
        # 对用户id和时间进行排序
        data_df = data_df.sort_values(by=['userId','timestamp'])
        train_data,test_data,val_data = [],[],[]

        item_id_max = data_df['movieId'].max()
        for user_id, df in tqdm(data_df[['userId','movieId']].groupby('userId')):
            pos_list = df['movieId'].tolist() # 这个用户看过的所有的电影

            def generate_neg(): # 生成负样本电影 确保负样本不在正样本中
                neg = pos_list[0]
                while neg in pos_list:
                    neg = random.randint(1,item_id_max)
                return neg
            neg_list = [generate_neg() for i in range(len(pos_list) + 100)]
            # 生成用户行为序列
            for i in range(1,len(pos_list)):
                hist_i = pos_list[:i] # 生成用户行为系列，记录用户历史行为
                if i == len(pos_list) - 1:
                    test_data.append([hist_i,pos_list[i],[user_id,1]])
                    for neg in neg_list[i:]:
                        test_data.append([hist_i, neg, [user_id,0]])
                elif i == len(pos_list) - 2:
                    val_data.append([hist_i,pos_list[i],1])
                    val_data.append([hist_i,neg_list[i],0])
                else:
                    train_data.append([hist_i,pos_list[i],1])
                    train_data.append([hist_i,neg_list[i],0])
        user_num, item_num = data_df['userId'].max() + 1 ,data_df['movieId'].max() + 1
        item_feat_col = self._sparseFeature('userId',item_num,global_args.embed_dim)

        random.shuffle(train_data)
        random.shuffle(val_data)

        # create a dataframe
        train = pd.DataFrame(train_data,columns = ['hist','target_item','label'])
        val   = pd.DataFrame(val_data,columns = ['hist','target_item','label'])
        test  = pd.DataFrame(test_data,columns = ['hist','target_item','label'])

        # 由于hist尺寸不同，需要padding到一样大小
        print('padding...')
        train_X = [pad_sequences(train['hist'],maxlen = global_args.maxlen),train['target_item'].values]
        train_Y = train['label'].values
        val_X = [pad_sequences(val['hist'],maxlen = global_args.maxlen),val['target_item'].values]
        val_Y = val['label'].values
        test_X = [pad_sequences(test['hist'],maxlen = global_args.maxlen),test['target_item'].values]
        test_Y = test['label'].values.tolist()
        return item_feat_col, (train_X,train_Y),(val_X,val_Y),(test_X,test_Y)


    def _sparseFeature(self,feat, feat_num, embed_dim = 4):
        # feat：特征名字
        # feat_num：这种特征有多少种值
        # embed的尺寸是多少
        return {'feat':feat,'feat_num':feat_num,'embed_dim':embed_dim}

    def calculate(self,*args):

        train_X,train_Y = self.train
        val_X,val_Y = self.val
        # build model
        self.model = DNN_model(self.item_fea_col,global_args.maxlen,global_args.hidden_unit, \
                               global_args.activation,global_args.embed_reg)
        self.model.summary() # 输出模型的参数
        # ======== model checkpoint === 
        # check_path = '../save/fm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
        # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
        #                                                 verbose=1, period=5)
        # ======== compile
        self.model.compile(loss = binary_crossentropy,optimizer=Adam(learning_rate = global_args.learning_rate))
        # ======== model fit 
        result = []
        for epoch in range(1,global_args.epochs+1):
            self.model.fit(
                train_X,
                train_Y,
                validation_data = (val_X, val_Y),
                epochs = 1,
                batch_size = global_args.batch_size,
            )
        # ======== evaluate
            if epoch % 5 == 0:
                hit_rate,ndcg = evaluate_model(self.model,self.test,global_args.K)
                print('iteration %d fit, evaluate: HR = %.4f, NDCG = %.4f' % (epoch,hit_rate,ndcg))
