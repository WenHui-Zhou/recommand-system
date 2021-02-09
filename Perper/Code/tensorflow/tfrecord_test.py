#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-02-09
# Copyright (C)2021 All rights reserved.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def _int64_features(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _bytes_features(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

mnist = input_data.read_data_sets(
    "./data",dtype=tf.uint8,one_hot=True
)
images = mnist.train.images

labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

filename = './output.tfrecords'

writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_features(pixel),
        'label' : _int64_features(np.argmax(labels[index])),
        'image_raw': _bytes_features(image_raw)
    }))
    writer.write(example.SerializeToString())
writer.close()
