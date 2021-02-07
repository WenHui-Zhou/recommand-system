#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-02-03
# Copyright (C)2021 All rights reserved.
import tensorflow as tf
import numpy as np

#parameter
input_node = 784
output_onde = 10
# 设置一些层的参数
image_size = 28
num_channels = 1
conv1_deep = 32
conv1_size = 5
conv2_deep = 64
conv2_size = 5
fc_size = 512

def inference(input_tensor,train,regularizer):

    with tf.variable_scope('layer1'):
        conv1_weights = tf.get_variable("weight",[conv1_size,conv1_size,num_channels,conv1_deep],\
                                        initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias",[conv1_deep],initializer = tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.variable_scope("layer2-pooling"):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weight = tf.get_variable("weight",[conv2_size,conv2_size,conv1_deep,conv2_deep],
                                      initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv2_bias = tf.get_variable("bias",[conv2_deep],initializer = tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,conv2_weight,strides= [1,1,1,1],padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_bias))

    with tf.variable_scope("layer4-pool"):
        pool2 = tf.nn.max_pool(relu2,ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])

    with tf.variable_scope("layer5"):
        fc1_weights = tf.get_variable("weights",[nodes,fc_size],\
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable("biases",[fc_size],initializer = tf.constant_initializer(0.0))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)
    with tf.variable_scope('layer5-fc2'):
        fc2_weight = tf.get_variable('weights',[fc_size,output_onde],initializer = tf.truncated_normal_initializer(stddev=0.1))
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc2_weight))
        fc2_biases = tf.get_variable('bias',[output_onde],initializer = tf.constant_initializer(0.0))
        logits =  tf.matmul(fc1,fc2_weight) + fc2_biases
    return logits
