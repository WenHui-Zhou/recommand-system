#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2020-12-25
# Copyright (C)2020 All rights reserved.
 

import time 
import os
from model.word2vec import Word2Vec

def run(arg):
    assert os.path.exists('./data/Online_Retail.xlsx'), \
            'file not exits in path,please download the dataset before this.'
    print('algrithm begin ... ')
    start = time.time()
    target_id = '84029E' # 某个产品的特殊编号
    top_n = 10   #  前几条记录
    recommands = Word2Vec().calculate(target_id,top_n) # 传入的参数为商品的id，以及topN similar
    print('Cost time: %f' %(time.time() - start))
    

 
