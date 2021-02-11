#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-02-11
# Copyright (C)2021 All rights reserved.
def queue_test():
    import tensorflow as tf
    queue = tf.FIFOQueue(100,'float')
    enqueue_op = queue.enqueue([tf.random_normal([1])])

    # 启动5个线程,针对queue，执行enqueue操作
    qr = tf.train.QueueRunner(queue,[enqueue_op]*5)
    tf.train.add_queue_runner(qr)
    out_tensor = queue.dequeue()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        for _ in range(5):
            print(sess.run(out_tensor)[0])
        coord.request_stop()
        coord.join(threads)


def generate_data():
    import tensorflow as tf
    def _int64_feature(value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
    num_shards = 2
    instances_per_shard = 2
    for i in range(num_shards):
        filename = ('./data.tfrecords-%.5d-of-%.5d'%(i,num_shards))
        writer = tf.python_io.TFRecordWriter(filename)
        for j in range(instances_per_shard):
            example = tf.train.Example(features = tf.train.Features(feature={
                "i":_int64_feature(i),
                "j":_int64_feature(j)
            }))
            writer.write(example.SerializeToString())
        writer.close()

#generate_data()

def read_tfrecord():
    import tensorflow as tf
    files = tf.train.match_filenames_once('./data.tfrecords-*')
    # 生成文件队列
    filename_queue = tf.train.string_input_producer(files,shuffle=False)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features = {
            'i':tf.FixedLenFeature([],tf.int64),
            'j':tf.FixedLenFeature([],tf.int64),
        }
    )
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        print(sess.run(files))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord = coord)
        for i in range(6):
            print(sess.run([features['i'],features['j']]))
        coord.request_stop()
        coord.join(threads)
read_tfrecord()


