#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2020-12-23
# Copyright (C)2020 All rights reserved.

import time 
import os

def run(arg):
    assert os.path.exists('./data/ml-latest-small/ratings.csv'), \
            'file not exits in path,please download the dataset before this.'
    print('algrithm begin ... ')
    start = time.time()
    target_id = 1 # 用户的id
    top_n = 10   #  前几条记录
    if arg == 'item_cf':
        from model.cf import ItemCf as Cf
    elif arg == 'user_cf':
        from model.cf import UserCf as Cf
    recommands = Cf().calculate(target_id,top_n) # 得到推荐的商品或者用户,传入参数为用户/商品的id，以及topN similar
    print('Cost time: %f' %(time.time() - start))
    

 
