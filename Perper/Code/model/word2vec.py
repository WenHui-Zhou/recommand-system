#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2020-12-25
# Copyright (C)2020 All rights reserved.

import math
import pandas as pd
import random
from tqdm import tqdm
import gensim

pd.options.mode.chained_assignment = None  # default='warn'

class Word2Vec:

    def __init__(self):
        self.file_root = './data/'
        self._init_frame()
    
    def _init_frame(self):
        # 处理数据集
        self.frame = pd.read_excel(self.file_root + 'Online_Retail.xlsx')
        print('处理数据')
        print(self.frame.head())
        self.frame.dropna(inplace=True)
        self.frame['StockCode'] = self.frame['StockCode'].astype(str)
        customers = self.frame['CustomerID'].unique().tolist()
        random.shuffle(customers)
        customers_train = [customers[i] for i in  range(round(0.9 * len(customers)))]
        self.train_df = self.frame[self.frame['CustomerID'].isin(customers_train)]
        self.valid_df = self.frame[~self.frame['CustomerID'].isin(customers_train)]

    def _build_model(self):
        purchases_train = []
        for i in tqdm(self.train_df['CustomerID'].unique()):
            temp = self.train_df[self.train_df['CustomerID'] == i]['StockCode'].tolist() 
            purchases_train.append(temp)
        model = gensim.models.Word2Vec(window = 10, sg = 1, hs = 0,
                                     negative = 10, alpha = 0.03, min_alpha = 0.0007, seed = 14)
        model.build_vocab(purchases_train,progress_per = 200)
        model.train(purchases_train, total_examples = model.corpus_count,
                   epochs=10,report_delay=1)
        model.init_sims(replace=True)
        self.model = model

    def _similar_products(self,target_id,topN = 6):
        target = self.model[target_id]
        ms = self.model.similar_by_vector(target,topn = topN + 1)
        
        products = self.train_df[['StockCode','Description']]
        products.drop_duplicates(inplace=True,subset='StockCode',keep='last')
        products_dict = products.groupby('StockCode')['Description'].apply(list).tolist()
        print('| 描述 | 相似度 |')
        for j in ms:
            print('| {} | {} |'.format(str(products_dict[int(j[0][0])]),str(j[1])))
        
    def calculate(self,*args):
        self._build_model()
        target = args[0]
        topN = args[1]
        self._similar_products(target,topN)
        return 'done'

        


