#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-01-30
# Copyright (C)2021 All rights reserved.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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
def inference_v2(input_tensor,avg_class):
    with tf.variable_scope('layer1',reuse = tf.AUTO_REUSE):
        weights = tf.get_variable("weights",shape = [INPUT_NODE,LAYER_NODE],
                                 initializer = tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases",[LAYER_NODE],
                                initializer = tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights) + biases)
    with tf.variable_scope('layer2',reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('weights',shape=[LAYER_NODE,OUTPUT_NODE],
                                 initializer = tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases",shape = [OUTPUT_NODE],
                               initializer = tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights) + biases
    return layer2

def train(mnist):
	# 定义网络各种参数
    x  = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
    # 生成隐藏层
#    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER_NODE],stddev = 0.1))
#    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER_NODE]))
#    weights2 = tf.Variable(tf.truncated_normal([LAYER_NODE,OUTPUT_NODE],stddev = 0.1))
#    biases2 = tf.Variable(tf.constant(0.1,shape = [OUTPUT_NODE]))

#    y = inference(x,None,weights1,biases1,weights2,biases2)
    y = inference_v2(x,None)
    global_step = tf.Variable(0,trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,global_step
    )
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference_v2(x,variable_averages)
    # 定义损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = y,labels = tf.argmax(y_,1)
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l1_regularizer(REGULARIZATION_RATE)
    with tf.variable_scope('layer1',reuse=True):
        regularization = regularizer(tf.get_variable("weights",shape = [INPUT_NODE,LAYER_NODE]))
    with tf.variable_scope('layer2',reuse = True):
        regularization += regularizer(tf.get_variable("weights",shape = [LAYER_NODE,OUTPUT_NODE]))
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    train_op = tf.group(train_step,variable_averages_op)
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    # 定义训练过程，以及测试的代码
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x:mnist.validation.images,
                         y_:mnist.validation.labels}
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print('after %d training step(s).,validation accuracy is %g' %(i, validate_acc))
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict = {x:xs,y_:ys})
        test_acc = sess.run(accuracy,feed_dict =test_feed)
        print('after %d training step(s).,test accuracy is %g' %(TRAINING_STEPS, test_acc))
def main(argv=None):
#    import pdb
#    pdb.set_trace()
    mnist = input_data.read_data_sets('../data/mnist/',one_hot = True)
    train(mnist)

if __name__ == '__main__':
#    tf.app.run()
    main()
        


