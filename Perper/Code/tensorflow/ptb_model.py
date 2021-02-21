#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-02-21
# Copyright (C)2021 All rights reserved.
import numpy as np
import tensorflow as tf

class PTBModel(object):
    def __init__(self,is_training,batch_size,num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.input_data = tf.placeholder(tf.int32,[batch_size,num_steps])
        self.targets = tf.placeholder(tf.int32,[batch_size,num_steps])
        dropout_keep_prob = lstm_keep_prob if is_training else 1.0
        lstm_cells = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_size),
                                                   output_keep_prob = dropout_keep_prob)
                     for _ in range(num_layers)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        self.intial_state = cell.zero_state(batch_size,tf.float32)
        embedding = tf.get_variable('embedding',[vocab_size,hidden_size])
        inputs = tf.nn.embedding_lookup(embedding,self.input_data)
        if is_training:
            inputs = tf.nn.dropout(inputs,embedding_keep_prob)

            # softmax
            outputs = []
            state = self.initial_state
            with tf.variable_scope('RNN'):
                for time_step in range(num_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    cell_output,state = cell(inputs[:,time_step,:],state)
            output = tf.reshape(tf.concat(outputs,1),[-1,hidden_size])
            weight = tf.transpose(embedding)
            bias = tf.get_variable('bias',[vocab_size])
            logits = tf.matmul(outputs,weight) + bias
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = tf.reshape(self.targets,[-1]),
                logits = logits
            )
            self.cost = tf.reduce_sum(loss) / batch_size
            self.final_state = state
            if not is_training:
                return
            trainable_variables = tf.trainable_variables()
            grads,_ = tf.clip_by_global_norm(
                tf.gradients(self.cost,trainable_varables),max_grad_norm)
            optimizer= tf.train.GradientDescentOptimizer(learning_rate=1.0)
            self.train_op = optimizer.apply_gradients(
                zip(grads,trainable_variables))

def run_epoch(sesson,model,batches,train_op,output_log,step):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    for x,y in batches:
        cost,state,_ = session.run(
            [model.cost,model.final_state,train_op],
            {model.input_data:x,model.targets:y,model.initial_state:state}
        )
        total_costs += cost
        iters += model.num_steps
    return step, np.exp(total_costs/iters)

def main():
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_batches = make_batches(
            read_data(train_data),train_batch_size,train_num_step)
        ...
        )
        for i in range(num_epoch):
            step,train_ppix = run_epoch(sess,train_model,train_batches,
                                       train_model.train_op,True,step)
            ...

    
