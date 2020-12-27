#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2020-12-23
# Copyright (C)2020 All rights reserved.
import sys
from workflow.cf_workflow import run as cf
from workflow.w2v_workflow import run as w2v
from workflow.LR_workflow import run as LR

def main():
    arg = sys.argv[1]
    if 'cf' in arg:
        cf(arg) # arg in [item_cf,user_cf]
    elif arg == 'word2vec':
        w2v(arg)
    elif arg == 'LR':
        LR(arg)
    else:
        print('args must in ["item_cf,user_cf,LR"]')
    sys.exit()

if __name__ == '__main__':
    main()
