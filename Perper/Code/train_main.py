#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2020-12-23
# Copyright (C)2020 All rights reserved.
import sys

def main():
    arg = sys.argv[1]
    if 'cf' in arg:
        from workflow.cf_workflow import run as cf
        cf(arg) # arg in [item_cf,user_cf]
    elif arg == 'word2vec':
        from workflow.w2v_workflow import run as w2v
        w2v(arg)
    elif arg == 'LR':
        from workflow.LR_workflow import run as LR
        LR(arg)
    elif arg == 'FM':
        from workflow.fm_workflow import run as FM
        FM(arg)
    elif arg == 'gbdt':
        from workflow.gbdt_workflow import run as GBDT
        GBDT(arg)
    elif arg == 'gbdt_lr':
        from workflow.gbdtLR_workflow import run as GBDTLR
        GBDTLR(arg)
    else:
        print('args must in ["item_cf,user_cf,LR,FM,gbdt"]')
    sys.exit()

if __name__ == '__main__':
    main()
