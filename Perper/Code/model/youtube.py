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

    def __init__(self,userid_max,movieid_max,hist_len,hidden_units=[256,128,64],
                 activation='relu',embed_reg = 1e-4):
        super(youtube_recall_model,self).__init__()
        self.userid_max = userid_max
        self.movieid_max = movieid_max
        self.hist_len = hist_len # 用户历史记录使用的top k
        self.embed_his_layers = {
            'embed_his_' + str(i): Embedding(
                input_dim = self.movieid_max,
                input_length = 1,
                output_dim = global_args.embed_dim,
                embeddings_initializer = 'random_uniform',
                embeddings_regularizer = l2(embed_reg)
            ) for i in range(self.hist_len)
        }
        #将每一个vector输入到各自的dense结构中
        self.embed_user_layer = Embedding(
            input_dim = self.userid_max,
            input_length = 1,
            output_dim = global_args.embed_dim,
            embeddings_initializer = 'random_uniform',
            embeddings_regularizer = l2(embed_reg)
        )
        self.dnn_network = [Dense(units=unit,activation=activation) for unit in hidden_units]
        softmax_dense = Dense(units=self.movieid_max,activation='softmax')
        self.dnn_network.append(softmax_dense)

    def call(self,inputs,**kwargs):
        userid,history = inputs
        history_movie = history[:,0]
        history_rate = history[:,1]
        his_embed = tf.concat([self.embed_his_layers['embed_his_{}'.format(i)](history_movie[:,i])*history_rate[:,i]
                                 for i in range(self.hist_len)],axis=-1)
        user_embed = self.embed_user_layer(userid)
        user_embed = tf.reshape(user_embed,[-1,user_embed.shape[-1]])
        stack = tf.concat([user_embed,his_embed],axis=-1)
        outputs = stack# (input[0],168)
        for dnn in self.dnn_network:
            outputs = dnn(outputs)
        return outputs

    def summary(self,**kwargs):
        user_inputs = tf.keras.Input(shape=(1,), dtype = tf.float32)
        hist_inputs = tf.keras.Input(shape=(2,self.hist_len,), dtype = tf.float32)
        tf.keras.Model(inputs=[user_inputs, hist_inputs],outputs = self.call([user_inputs, hist_inputs])).summary()

class Youtube:
    def __init__(self):
        super(Youtube,self).__init__()
        self.file_root = './data/ml-latest-small/'
        self._init_param()
        #item的个数以及dimension 训练集，验证集，测试集
        self.trainset,self.testset = self._init_frame()

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
        parser.add_argument('--hist_len',type=int,default=5,help = 'fm latent vector size')
        global global_args
        global_args = parser.parse_args()
    
    def _init_frame(self):
        # 处理数据集
        print('prepare dataset...')
        # ratings: 
        # userId,movieId,rating,timestamp
        # 1,1,4.0,964982703
        data_df = pd.read_csv(os.path.join(self.file_root,'ratings.csv'),sep=',',header=0)
        self.userid_max = data_df['userId'].max()
        self.movieid_max = data_df['movieId'].max()
        train_data,test_data = [],[]
#        data_df.groupby('userId').sort_values(by=['timestamp'])
        train,test = train_test_split(data_df,test_size=global_args.test_size)
        train_y, test_y = [],[]
        userids_train = []
        for user_id, df in tqdm(train.groupby('userId')):
            df = df.sort_values(by=['timestamp'])
            movie_id = df['movieId'].tolist()
            label_id = movie_id[0]
            movie_id = movie_id[1:]
            movie_rate = df['rating'].tolist()[1:]
            history = [movie_id[:global_args.hist_len],movie_rate[:global_args.hist_len]]
            train_data.append(np.array(history))
            train_y.append(label_id)
            userids_train.append(user_id)
        train_data = [userids_train,train_data]
        userids_test = []
        for user_id, df in tqdm(test.groupby('userId')):
            df = df.sort_values(by=['timestamp'])
            movie_id = df['movieId'].tolist()
            label_id = movie_id[0]
            movie_id = movie_id[1:]
            movie_rate = df['rating'].tolist()[1:]
            history = [movie_id[:global_args.hist_len],movie_rate[:global_args.hist_len]]
            test_data.append(np.array(history))
            test_y.append(label_id)
            userids_test.append(user_id)
        test_data = [userids_test,test_data]

        return (train_data,train_y),(test_data,test_y)

    def calculate(self,*args):
        # prepare data
        train_X,train_Y= self.trainset
        test_X,test_Y = self.testset

        # build model
   #     mirrored_strategy = tf.distribute.MirroredStrategy()
   #     with mirrored_strategy.scope():
        self.model = youtube_recall_model(self.userid_max,self.movieid_max,global_args.hist_len)
        self.model.summary()
        # ======== model checkpoint === 
        # check_path = '../save/fm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
        # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
        #                                                 verbose=1, period=5)
        # ======== compile
        def sample_loss(y_true,y_pred,**kwags):
            # y_pred: 64 vector
            # y_true: a ditgit
            output_layer = self.model.layers[-1]
            inputs = output_layer.input
            weights = output_layer.weights
           # y_true = tf.reshape(y_true,[-1,1])
            loss = tf.nn.sampled_softmax_loss(
                    weights=tf.transpose(weights[0]),
                    biases=weights[1],
                    labels=y_true,
                    inputs=inputs,
                    num_sampled = 128,
                    num_classes = self.movieid_max)
            return loss
        loss_calculator = SampledSoftmaxLoss(self.model,self.movieid_max)   
        self.model.compile(loss =sample_loss,optimizer=Adam(learning_rate = global_args.learning_rate),\
                           experimental_run_tf_function=False)
        # ======== model fit 
        train_X[0] = np.array(train_X[0])
        train_X[1] = np.array(train_X[1])
        new_trainY = []
        for val in train_Y:
            label = np.zeros([1,self.movieid_max])[0]
            label[val-1] = 1
            new_trainY.append(label)
#                labels = tf.reshape(b,[-1,1])
#        train_Y = np.array(new_trainY)
        train_Y = np.array(train_Y)

   #     test_X[0] = np.array(test_X[0])
   #     test_X[1] = np.array(test_X[1])
   #     test_Y = np.array(test_Y)
        self.model.fit(
            train_X,
            train_Y,
            epochs = global_args.epochs,
 #           callbacks = [EarlyStopping(monitor='val_loss',patience=1,restore_best_weights=True)],
            batch_size = global_args.batch_size,
 #           validation_split=0.1
        )
        # ======== evaluate
 #       print('test AUC: %f' % self.model.evaluate(train_X,train_Y,batch_size = global_args.batch_size)[1])
        return 'done'
