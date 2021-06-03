[TOC]

# Tensorflow 高层封装

原生的tensorflow api可以灵活支持不同的神经网络，但是其代码冗长， 不断定义变量等，不便于直接使用。因此tensorflow的高层封装有很多，下面进行简单的介绍。

## tensorflow-slim

tensorflow slim中封装了较多的图像分析模型，简化网络搭建以及训练的过程。下面是一段minst上的代码。

```python
def lenet5(inputs):
    inputs = tf.reshape(inputs,[-1,28,28,-1])
    net = slim.conv2d(inputs,32,[5,5],padding='SAME',scope='layer1-conv')
    net = slim.max_pool2d(net,2,stride=2,scope='layer2-max-pool')
    net = slim.conv2d(inputs,64,[5,5],padding='SAME',scope='layer3-conv')
    net = slim.max_pool2d(net,2,stride=2,scope='layer4-max-pool')

    net = slim.flatten(net, scope='flatten')
    net = slim.fully_connected(net,500,scope='layer5')
    net = slim.fully_connected(net,10,scope='output')
    return net
```

slim在定义网络层的时候，不用关心网络变量的定义以及初始化。

与slim类似的还有TFLearn，但网络的执行过程依然是：

1. 网络结构的定义
2. 训练数据的准备
3. 损失函数，优化器的定义（placeholder传入数据）
4. 网络的训练



## keras

keras是tensorflow官方提供的高层封装库，keras API对模型的定义、损失函数、训练过程进行封装，可以分为下面三个步骤：

1. 数据处理
2. 模型定义（sequential，add）
3. 模型训练（compile，fit）

下面展示一个用keras完成的例子：

```python
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
```

keras但同样对循环神经网络，nlp有着很好的支持，下面用一段代码说明。

```python
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
```

上述的kears的用法，使用Sequential，只支持顺序结构，对一些并列的结构，keras可以用返回值的形式定义网络层，也就和正常的Python代码类似。

```python
import keras
from keras.datasets import mnist
from keras.model import Model

inputs = Input(shape = (784,))
x = Dense(500,activation = 'relu')(inputs)
predictions = Dense(10,activation='relu')(x)
model = Model(inputs = inputs,outputs=predictions)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
             metrics=['accuracy'])
model.fit(trainX,trainY,batch_size=128,
          epochs=20,validation_data = (testX,testY))
```

keras也有它的问题，即原生态的keras代码无法支持分布式训练，对数据的处理流程支持不好，需要将所有数据都载入内存中。因此通过混用tensorflow代码以及keras代码，提升建模的灵活性。即数据载入，和分布式训练用tensorflow的代码，但是网络结构可以用keras的代码：

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('path',one_hot= True)
x = tf.placeholder(tf.float32,shape=(None,784))
y_ = tf.placeholder(tf.float32,shape=(None,10))

net = tf.keras.layers.Dense(500,activation='relu')(x)
y = tf.keras.layers.Dense(10,activation='softmax')(net)

loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_,y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

acc_value = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_,y))

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(10000):
        xs,ys = mnist_data.train.next_batch(100)
        x,loss_value = sess.run([train_step,loss],feed_dict = {x:xs,y_:ys})
        if i % 1000 == 0:
            print('loss: ',loss_value)
    print(acc_value.eval(feed_dict={x:mnist_data.test.images,y_:mnist_data.test.labels}))
```



