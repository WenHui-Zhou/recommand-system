#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2020-12-25
# Copyright (C)2020 All rights reserved.
 

import time 
import os
from model.gbdt import GBDT

def run(arg):
    assert os.path.exists('./data/boston_house/boston.csv'), \
            'file not exits in path,please download the dataset before this.'
    print('{} algrithm begin ... '.format(arg))
    start = time.time()
    recommands = GBDT().calculate()
    print('Cost time: %f' %(time.time() - start))
    

 
