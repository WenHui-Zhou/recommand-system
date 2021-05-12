#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-05-12
# Copyright (C)2021 All rights reserved.
import keras.preprocessing import sequence
import keras.models import Sequential
from keras.layers import Dense,Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# 确定一些超参数
max_features = 20000 # 使用最多的单词数
max_len = 80 # 循环截断的长度
batch_size = 32

# 加载、整理数据
(trainX,trainY),(testX,testY) = imdb.load_data(num_word = max_features)
trainX = sequence.pad_sequences(trainX,maxlen = maxlen)
testX = sequence.pad_sequences(testX,maxlen = maxlen)

# 模型构建
model = Sequential()
model.add(
    Embedding(max_features,128)
)
model.add(
    LSTM(128,dropout=0.2,recurrent_dropout=0.2)
)
model.add(
    Dense(1,activation='sigmoid')
)

# 损失函数、优化函数的配置
model.compile(loss = 'binary_crossentropy',
             optimizer = 'adam',metrics=['accuracy'])
model.fit(trainX,trainY,batch_size = batch_size,epoch=15,validation_data = (testX,testY))

score = model.evaluate(testX,testY)
print('test loss:',score[0])
print('test accuracy:',score[1])
