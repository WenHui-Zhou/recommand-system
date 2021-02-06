#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-02-03
# Copyright (C)2021 All rights reserved.
import tensorflow as tf

#parameter
input_node = 784
output_onde = 10
middle_node = 500
# 设置一些层的参数

def get_weight_variable(shape,regularizer):
    weights = tf.get_variable("weights",shape,initializer = tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

def inference(input_tensor,regularizer):

    with tf.variable_scope('layer1'):
        weights = get_weight_variable([5,5,3,16],regularizer)
        biases = tf.get_variable("biases",[16],initializer = tf.constant_initializer(0.0))
        layer1 = tf.nn.conv2d(input_tensor,weights,strides=[1,1,1,1],paddiing = "SAME")
        layer1 = tf.nn.add_conv(layer1,biases)
#        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights) + biases)
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([middle_node,output_onde],regularizer)
        biases = tf.get_variable("biases",[output_onde],initializer = tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights) + biases
    return layer2

    
