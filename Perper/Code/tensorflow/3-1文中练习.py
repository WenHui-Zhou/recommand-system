#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-01-26
# Copyright (C)2021 All rights reserved.
import tensorflow as tf 

a = tf.constant([1,2],name = 'a',dtype = tf.float32)
print(a)

c = tf.Variable(tf.random_normal((2,3),stddev=2),name = "c_input")
print(c)

b = tf.placeholder(tf.float32,shape=(None,2),name = 'a_input')
print(b)

# 看到没，placeholder这三个属性的传入顺序和constant,Variable的不同

bias = tf.zeros([2,3],dtype = tf.float32,name = "bias")
print(bias)


