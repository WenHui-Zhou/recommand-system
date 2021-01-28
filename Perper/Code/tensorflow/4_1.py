#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-01-28
# Copyright (C)2021 All rights reserved.
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# 定义输入接口
x = tf.placeholder(tf.float32,shape=(None,2),name="x-input")
y_ = tf.placeholder(tf.float32,shape=(None,1),name="y-input")
# 定义网络结构
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)

#定义损失
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*loss_more,(y_ - y)*loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 生成随机数据集
rdm = RandomState(1)
dataset_size = 128

X = rdm.rand(dataset_size,2)
Y = [[x1 + x2 + rdm.rand() / 10 - 0.05] for (x1,x2) in X]

# 训练网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    step = 5000
    for i in range(step):
        start = (i*batch_size)% dataset_size
        end = min(start + batch_size,dataset_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        print(sess.run(w1))

