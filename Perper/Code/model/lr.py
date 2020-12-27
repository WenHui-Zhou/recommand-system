#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2020-12-25
# Copyright (C)2020 All rights reserved.

from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

class LR:

    def __init__(self):
        self.file_root = './data/'
        self.FLAGS = ''
        self._init_frame()
        self._init_param()
    
    def _init_frame(self):
        # 处理数据集
        self.dataset = input_data.read_data_sets(self.file_root,one_hot = True)

    def _init_param(self):
        # 定义网络参数，以及训练的时候用到的参数
        tf.flags.DEFINE_float('learning_rate',0.01,"the learning rate of the model")
        tf.flags.DEFINE_integer('training_epochs',25,"how many time we train the model")
        tf.flags.DEFINE_integer('batch_size',100,"a batch data for per train")
        tf.flags.DEFINE_integer('display_step',1,"display the result")
        self.FLAGS = tf.flags.FLAGS
        self.FLAGS.flag_values_dict()

    def _build_train_model(self):
        print('build model...')
        x = tf.placeholder(tf.float32,[None,784]) # mnist 输入尺寸
        y = tf.placeholder(tf.float32,[None,10]) # 输出的类别

        W = tf.Variable(tf.random_uniform([784,10]))
        b = tf.Variable(tf.random_uniform([10]) )

        # 逻辑回归
        pred = tf.nn.softmax(tf.matmul(x,W) + b)
        # 最外围的mean是对batch中所有的样本进行的
        # 内层的sum是计算one-hot为1的那一类的输出概率
        loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1)) 
        optimizer = tf.train.GradientDescentOptimizer(self.FLAGS.learning_rate).minimize(loss)

        print('train begin...')
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            # train step
            for epoch in range(self.FLAGS.training_epochs):
                avg_loss = 0
                total_batch = int(self.dataset.train.num_examples/self.FLAGS.batch_size)
                for i in range(total_batch):
                    batch_xs,batch_ys = self.dataset.train.next_batch(self.FLAGS.batch_size)
                    _, c = sess.run([optimizer,loss],feed_dict = {x:batch_xs,y:batch_ys})

                    avg_loss += c / total_batch

                if (epoch + 1) % self.FLAGS.display_step == 0:
                    print('Epoch: ','%04d'%(epoch+1),'loss= ', "{:.9f}".format(avg_loss))
            print('optimization finish')

            # test
            correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            print('Accuracy: ',accuracy.eval({x:self.dataset.test.images,y:self.dataset.test.labels}))

    def calculate(self,*args):
        self._build_train_model()
        return 'done'
