#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-05-12
# Copyright (C)2021 All rights reserved.
import keras
import keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from keras import backend as K

num_classes = 10
img_rows,img_cols = 28,28

# 数据准备的部分
(trainX,trainY),(testX,testY) = mnist.load_data()
if K.image_data_format() == 'channel_first':
    trainX = trainX.reshape(trainX.shape[0],1,img_rows,img_cols)
    testX = testX.reshape(testX.shape[0],1,img_rows,img_cols)
    input_shape = (1,img_rows,img_cols)
else:
    trainX = trainX.reshape(trainX.shape[0],img_rows,img_cols,1)
    testX = testX.reshape(testX.shape[0],img_rows,img_cols,1)
    input_shape = (img_rows,img_cols,1)

# 将像素值转化到0-1之间
trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255.0
testX /= 255.0

# 将数据转成one-hot
trainY = keras.utils.to_categorical(trainY,num_classes)
testY = keras.utils.to_categorical(testY,num_classes)

# 构建网络
model = Sequential()
model.add(
    Conv2D(32,kernel_size = (5,5),activation = 'relu',input_shape = input_shape)
)
model.add(
    MaxPooling2D(pool_size = (2,2))
)
model.add(
    Conv2D(64,(5,5),activation='relu')
)

model.add(
    MaxPooling2D(pool_size=(2,2))
)

model.add(
    Flatten()
)
model.add(
    Dense(500,activation='relu')
)
model.add(
    Dense(num_classes,activation = 'softmax')
)

model.compile(loss = keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.SGD(),
             metrics = ['accuracy'])
model.fit(trainX,trainY,batch_size = 128,epochs=20,validation_data = (testX,testY))

score = model.evaluate(testX,testY)
print('test loss:', score[0])
print('test accuracy:', score[1])

'''
基本流程就是：
1. 使用dataloader或者其他准备数据集
2. 数据集的处理，包括shape, 预处理步骤
3. 搭建网络模型，使用sequential、add，或者使用输入的形式
4. 使用compile定义损失函数、优化函数、评价函数等
5. 使用fit完成epoch，batch_size，validation的配置，开始训练
6. 输出结果，model.evaluate(testX,testY)
'''
