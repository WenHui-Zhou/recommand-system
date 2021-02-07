#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-02-03
# Copyright (C)2021 All rights reserved.
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference_letnet5
import numpy as np
import os 

batch_size = 100
learning_rate_base = 0.8
learning_rate_decay = 0.99
regularization_rate = 0.0001
training_step = 3000
moving_average_decay = 0.99
model_save_path = "./"
model_name = "model.ckpt"
image_size = mnist_inference_letnet5.image_size
def train(mnist):
#    x = tf.placeholder(tf.float32,[None,mnist_inference.input_node],name = "x_input")
    x = tf.placeholder(tf.float32,[batch_size,mnist_inference_letnet5.image_size,\
                                  mnist_inference_letnet5.image_size,\
                                  mnist_inference_letnet5.num_channels],name = "x-input")
    y_ = tf.placeholder(tf.float32,[None,mnist_inference_letnet5.output_onde],name = "y_input")
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    y = mnist_inference_letnet5.inference(x,True,regularizer)
    global_step = tf.Variable(0,trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay,global_step
    )
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = y, labels = tf.argmax(y_,1)
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(learning_rate_base,
                                               global_step,
                                               mnist.train.num_examples / batch_size,
                                               learning_rate_decay)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name = "train")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(training_step):
            xs,ys = mnist.train.next_batch(batch_size)
            xs = np.reshape(xs,(batch_size,image_size,image_size,mnist_inference_letnet5.num_channels))
            _, loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})

            if i % 1000 == 0:
                print("after %d training step,loss on training batch is %g" %(step,loss_value))
                saver.save(sess,os.path.join(model_save_path,model_name),global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets('../data/mnist/',one_hot = True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()



