#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2020-12-23
# Copyright (C)2020 All rights reserved.
import sys
from workflow.cd_workflow import run as user_cf

def main():
    arg = sys.argv[1]
    if arg == 'cf':
        user_cf()
    else:
        print('args must in ["cf"]')
    sys.exit()

if __name__ == '__main__':
    main()
