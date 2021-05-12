#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-05-08
# Copyright (C)2021 All rights reserved.
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

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

def train(mnist):
    x = tf.placeholder(tf.float32,[None,784],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,10],name='y-input')
    y = lenet5(x)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = y,labels = tf.argmax(y_,1)
    )
    loss = tf.reduce_mean(cross_entropy)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(10000):
            xs,ys = mnist.train.next_batch(100)
            _,loss_value = sess.run([train_op,loss],feed_dict = {x:xs,y_:ys})
            if i % 1000 == 0:
                print("after %d training step(s), loss on training batch is %g" %(i,loss_value))

def main(argv= None):
    mnist = input_data.read_data_sets('/path/to/mnist_data',oen_hot = True)
    train(mnist)

if __name__ == '__main__':
    main()
