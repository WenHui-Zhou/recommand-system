#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-01-07
# Copyright (C)2021 All rights reserved.
 
import time 
import os
from model.wdl import WDL 

def run(arg):
    assert os.path.exists('./data/criteo_ctr/tiny.csv'), \
            'file not exits in path,please download the dataset before this.'
    print('{} algrithm begin ... '.format(arg))
    start = time.time()
    result = WDL().calculate()  # 或者直接调用call函数
    print('Cost time: %f' %(time.time() - start))
    

 
