#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2020-12-23
# Copyright (C)2020 All rights reserved.
import sys

def main():
    arg = sys.argv[1]
    if arg in ['item_cf','user_cf']:
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
    elif arg == 'gcforest':
        from workflow.gcForest_workflow import run as gcForest 
        gcForest(arg)
    elif arg == 'dnn':
        from workflow.dnn_workflow import run as DNN 
        DNN(arg)
    else:
        print('args must in ["item_cf,user_cf,LR,FM,gbdt,fbdt_lr,gcforest,dnn"]')
    sys.exit()

if __name__ == '__main__':
    main()
