#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-05-12
# Copyright (C)2021 All rights reserved.
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

