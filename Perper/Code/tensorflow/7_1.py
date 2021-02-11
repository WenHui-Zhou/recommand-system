#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-02-11
# Copyright (C)2021 All rights reserved.
import tensorflow as tf

q = tf.FIFOQueue(2,"int32")

init = q.enqueue_many(([0,10],))

x = q.dequeue()

y = x + 1
q_inc = q.enqueue([y])

with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v, _ = sess.run([x,q_inc])
        print(v)

print('tf.queuerunner: ')
import tensorflow as tf
import numpy as np
import threading
import time

def Myloop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print('stopppping from id %d \n'%worker_id)
            coord.request_stop()
        else:
            print("working on id %d\n"%worker_id)
        time.sleep(1)

# 声明一个tf.train.Coordinator
coord = tf.train.Coordinator()

threads = [
    threading.Thread(target=Myloop, args = (coord, i,)) for i in range(5)
]
for t in threads:
    t.start()

coord.join(threads)

import tensorflow as tf
import numpy as np
import threading
import time

def Myloop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print('stopppping from id %d \n'%worker_id)
            coord.request_stop()
        else:
            print("working on id %d\n"%worker_id)
        time.sleep(1)

# 声明一个tf.train.Coordinator
coord = tf.train.Coordinator()

threads = [
    threading.Thread(target=Myloop, args = (coord, i,)) for i in range(5)
]
for t in threads:
    t.start()

coord.join(threads)

