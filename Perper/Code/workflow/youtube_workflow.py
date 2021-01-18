#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-01-15
# Copyright (C)2020 All rights reserved.
 

import time 
import os
from model.youtube import Youtube

def run(arg):
    print('{} algrithm begin ... '.format(arg))
    start = time.time()
    result = Youtube().calculate()  # 或者直接调用call函数
    print('Cost time: %f' %(time.time() - start))
    

 
