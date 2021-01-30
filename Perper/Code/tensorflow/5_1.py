#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-01-30
# Copyright (C)2021 All rights reserved.
import tensorflow as tf
from tensorflow.examples.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1) + biases1)
        return tf.matmul(layer1,weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2)) + avg_class.average(biases2)

def train(mnist):
    x  = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
    # 生成隐藏层
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER_NODE],stddev = 0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER_NODE,OUTPUT_NODE],stddev = 0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape[OUTPUT_NODE]))

    y = inference(x,None,weights1,biases1,weights2,biases2)
    global_step = tf.Variable(0,trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,global_step
    )
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = y,labels = tf.argmax(y_,1)
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l1_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)


