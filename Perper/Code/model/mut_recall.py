#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2021-01-21
# Copyright (C)2021 All rights reserved.

import math,time,os
from tqdm import tqdm
import gc
import pickle
import random
from datetime import datetime
from operator import itemgetter
import numpy as np
import pandas as pd
from collections import defaultdict
import collections
from abc import ABC,abstractmethod

global_args = None

class BaseCf(ABC):

    def __init__(self,debug=True):
        self.file_root = './data/news/'
        self.save_root = './data/save_news/'
        self.debug = debug
        self._init_frame()
    
    def _init_frame(self):
        if self.debug:
            # read part data,默认读取sample的数据
            self.frame = self.get_all_click_sample()
        else:
            # read all data
            self.frame = self.get_all_click_df()

    def get_all_click_sample(self,sample_nums = 10000):
        all_click = pd.read_csv(self.file_root + 'train_click_log.csv')
        all_user_ids = all_click.user_id.unique()
        # 选择部分的user_id
        sample_user_ids = np.random.choice(all_user_ids,size = sample_nums,replace = False)
        all_click = all_click[all_click['user_id'].isin(sample_user_ids)]
        # 去重
        all_click = all_click.drop_duplicates(['user_id','click_article_id','click_timestamp'])
        return all_click

    def get_all_click_df(self,offline = True):
        if offline:
            all_click = pd.read_csv(self.file_root + 'train_click_log.csv')
        else:
            trn_click = pd.read_csv(self.file_root + 'train_click_log.csv')
            tsn_click = pd.read_csv(self.file_root + 'testA_click_log.csv')

            all_click = trn_click.append(tsn_click)
        all_click = all_click.drop_duplicates(['user_id','click_article_id','click_timestamp'])
        return all_click

    @staticmethod # 声明这个函数独立与类的存在，静态函数，使用类名.函数调用,第一个参数不用是self
    def _cosine_similar(target_vec,vec):
        """
        target_vec: 用户或者item
        vec：用户库或item库中的一条记录
        返回值：计算目标用户（item）和库中的cos相似度
        例如：
        x = [1 0 1 1 0] y = [0 1 1 0 1]
        cosine = (x1*y1 + x2*y2) / (sqrt(x1^2 + x2^2...) * sqrt(y1^2 + y2^2 ...))
        """
        union_len = len(set(target_vec) & set(vec))
        if union_len == 0:
            return 0.0
        product = len(target_vec)*len(vec)
        cosine = union_len / math.sqrt(product)
        return cosine

    @staticmethod
    def _euclidean_silimar(target_vec,vec):
        """
        计算两个vec之间的距离，然后套用导数 1/(1 + d) 作为相似度(距离越大，相似度越小)
        距离公式为：sqrt((x1-y1)^2 + (x2-y2)^2...))
        对于一个one-hot 编码的数据来说，两个向量求异或
        """
        XOR_len = math.sqrt(len(set(target_vec) ^ set(vec)))
        return 1.0 / (1 + XOR_len)

    @staticmethod
    def _coOccurrence_similar(target_vec,vec):
        """
        同现相似度
        计算方式为：len(A & B)/ sqrt(len(A) * len(B))
        """
        union_len = len(set(target_vec) & set(vec))
        product = len(target_vec) * len(vec)
        return (union_len *1.0)/math.sqrt(product)

    @staticmethod
    def _tanimoto_similar(target_vec,vec):
        """
        计算公式：len(A & B) / (len(A) + len(B) - len(A & B))
        """
        union_len = len(set(target_vec) & set(vec))
        return (union_len*1.0) / (len(target_vec) + len(vec) - union_len)
        

    @abstractmethod
    def calculate(self,*args):
        """
        作用：CF的对外接口，提供协同过滤服务,需要在子类中实现这个方法
        参数*args：不定长参数，允许输入多个参数，最后args是一个参数list
        """
        raise NotImplementedError

#class ItemCf(BaseCf):
class Mut_Recall(BaseCf):
    
    def __init__(self):
    #    super(ItemCf,self).__init__()
        super(Mut_Recall,self).__init__()


    def _init_param(self):
        # 定义网络参数，以及训练的时候用到的参数
        parser = argparse.ArgumentParser()
        parser.add_argument('algrithom',type=str,default="mut_recall",help = 'the pratice method')
        parser.add_argument('--sim_item_topk',type=int,default=10,help = 'the len of feature embeding')
        parser.add_argument('--learning_rate',type=float,default=0.001,help = 'the learning rate of the model')
        parser.add_argument('--epochs',type=int,default=100,help = 'how many time we train the model')
        parser.add_argument('--batch_size',type=int,default=256,help = 'a batch data size')
        parser.add_argument('--test_size',type=float,default=0.2,help = 'the rate of test set / all data')
        parser.add_argument('--w_reg',type=float,default=1e-4,help = 'w l2 normalization penalty')
        parser.add_argument('--v_reg',type=float,default=1e-4,help = 'v l2 normalization penalty')
        parser.add_argument('--recall_item_num',type=int,default=10,help = 'the latent vector size')
        global global_args
        global_args = parser.parse_args()

    def get_user_item_time(self):

        # 获取用户点击文章的序列，并将序列按照时间排序
        # 将dataframe 转化成容易操作的dict
        self.frame = self.frame.sort_values('click_timestamp')

        def make_item_time_pair(df):
            return list(zip(df['click_article_id'],df['click_timestamp']))

        user_item_time_df = click_df.groupby('user_id')['click_article_id',\
                                                        'click_timestamp'].apply(
                                                            lambda x:make_item_time_pair(x)).reset_index().rename(columns={
                                                                0:'item_time_list'
                                                            })
        user_item_time_dict = dict(zip(user_item_time_df['user_id'],user_item_time_df['item_time_list']))
        return user_item_time_dict

    def get_item_topk_click(self,k=10):
        # 如果候选不够，则用热门推荐列表来替换
        topk_click = self.frame['click_article_id'].value_counts().index[:k]
        return topk_click

    def itemcf_sim(self):
        user_item_time_dict = self.get_user_item_time()
        #计算物品相似度
        i2i_sim = {}
        item_cnt = defaultdict(int)
        for user,item_time_list in tqdm(user_item_time_dict.items()):
            for i,i_click_time in item_time_list:
                item_cnt[i] += 1 # 统计出现的次数
                for j,j_click_time in item_time_list:
                    if i == j:
                        continue
                    i2i_sim[i].setdefault(j,0)
                    i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)
        i2i_sim_ = i2i_sim.copy()
        for i, relate_items in i2i_sim.items():
            for j,wij in relate_items.items():
                i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i]*item_cnt[j])
#        pickle.dump(i2i_sim_,open(self.save_news + 'itemcf_i2i_sim.pkl','wb'))
        return i2i_sim_

    def item_based_recommand(self,user_id):
        user_hist_items = self.user_item_time_dict[user_id]
        user_hist_items_ = {user_id for user_id,_ in user_hist_items}

        item_rank = {}
        for loc,(i,click_time) in enumerate(user_hist_items):
            for j,wij in sorted(i2i_sim[i].items(),key=lambda x:x[1],reverse=True)[:global_args.sim_item_topk]:
                if j in user_hist_items_:
                    continue
                item_rank.setdefault(j,0)
                item_rank[j] += wij
        if len(item_rank) < global_args.recall_item_num:
            for i ,item in enumerate(self.item_top_click):
                if item in item_rank.items():
                    continue
                item_rank[item] = -i - 100
                if len(item_rank) == global_args.recall_item_num:
                    break
        item_rank = sorted(item_rank.items(),key=lambda x:x[1],reverse=True)[:global_args.recall_item_num]
        return item_rank

    def _get_users_item_recall(self):
        """
        find all the users that user id did not meet the target movie before.
        """
        user_recall_items_dict = collections.defaultdict(dict)
        for user in tqdm(self.frame['user_id'].unique()):
            user_recall_items_dict[user] = self.item_based_recommand(user)

        user_item_score_list = []
        for user,items in tqdm(user_recall_items_dict.items()):
            for item,score in items:
                user_item_score_list.append([user,item,score])
        recall_df = pd.DataFrame(user_item_score_list,columns=['user_id','click_article_id','pred_score'])
        recall_df.to_csv('../data/mut_recall.csv',index=False,header = True)
        return recall_df
        

    def calculate(self,*args):
        """
        实现抽象类，完成itemCF的服务
        """
        self.i2i_sim = self.itemcf_sim()
        self.user_item_time_dict = self.get_user_item_time()
        self.item_top_click = self.get_item_topk_click()
        recall_df = self._get_users_item_recall()
        return recall_df




