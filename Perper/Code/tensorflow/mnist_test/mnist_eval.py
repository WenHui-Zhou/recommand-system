#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-02-03
# Copyright (C)2021 All rights reserved.
import tensorflow as tf
import time 
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

eval_interval_sec = 5

def evaluate(mnist):
    x = tf.placeholder(tf.float32,shape = [None,mnist_inference.input_node],name = "x_input")
    y_ = tf.placeholder(tf.float32,shape = [None,mnist_inference.output_onde],name = "y_input")
    validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
    y = mnist_inference.inference(x,None)
    correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    variable_average = tf.train.ExponentialMovingAverage(mnist_train.moving_average_decay)
    variable_to_restore = variable_average.variables_to_restore()
    saver = tf.train.Saver(variable_to_restore)
    
    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy,feed_dict = validate_feed)
                print("after %s training step,validation accuracy = %g" %(global_step,accuracy_score))
            else:
                print("no checkpoint found")
            return
        time.sleep(eval_interval_sec)
def main(argv = None):
    mnist = input_data.read_data_sets('../data/mnist/',one_hot = True)
    evaluate(mnist)

if __name__ == "__main__":
    tf.app.run()
