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
from datatime import datetime
from operator import itemgetter
import numpy as np
import pandas as pd
from collections import defaultdict
import collections
from abc import ABC,abstractmethod


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

class UserCf(BaseCf):
    
    def __init__(self):
        super(UserCf,self).__init__()

    def _get_topN_similar_users(self,target_user_id,top_n):
        """
        计算与target用户最相似的top_n个用户
        """
        target_movies = self.frame[self.frame['userId'] == target_user_id]['movieId']
        other_users_id = [i for i in set(self.frame['userId']) if i != target_user_id]
        other_movies = [self.frame[self.frame['userId'] == i]['movieId'] for i in other_users_id]
        similar_list = [self._tanimoto_similar(target_movies,movies) for movies in other_movies]
        similar_list = sorted(zip(other_users_id,similar_list),key = lambda x:x[1],reverse = True)
        return similar_list[:top_n]

    def _get_candidate_movies(self,target_user_id):
        """
        find all the movies that user id did not meet before.
        """
        target_movies = set(self.frame[self.frame['userId'] == target_user_id]['movieId'])
        other_user_movies = set(self.frame[self.frame['userId']!=target_user_id]['movieId'])
        candidate_movies = list(target_movies^other_user_movies)
        return candidate_movies

    def _get_top_n_items(self,top_n_users,candidate_movies,top_n):
        """
        通过感兴趣程度来对电影排序
        interest = sum(sim * normalize_rate)
        """
        top_n_user_data = [self.frame[self.frame['userId'] == k] for k,_ in top_n_users]
        interest_list = []
        for movie_id in candidate_movies:
            tmp = []
            for user_data in top_n_user_data:
                if movie_id in user_data['movieId'].values:
                    tmp.append(user_data[user_data['movieId'] == movie_id]['rating'].values[0]/5.0)
                else:
                    tmp.append(0)
            interest = sum(top_n_users[i][1]*tmp[i] for i in range(len(top_n_users)))
            interest_list.append((movie_id,interest))
        interest_list = sorted(interest_list,key=lambda x:x[1],reverse=True)
        return interest_list[:top_n]

    def print_message(self,target_user_id,top_n,recommands_movies_id):
        mess = "基于用户的协同过滤算法,传入用户id，为用户(id:{})得到的前{}推荐的商品如下：\n".format(target_user_id,top_n)
        example_moviesId_of_target_user = list(zip(self.frame[self.frame['userId'] == target_user_id]['movieId'],\
                                                    self.frame[self.frame['userId'] == target_user_id]['rating']))
        print(mess)
        history = None
        for movie_id,score in example_moviesId_of_target_user[:top_n]:
            user_watch_movies_data = self.movie_frame[self.movie_frame['movieId'] == movie_id]
            user_watch_movies_data.loc[:, '对电影的评分'] = score
            if history is None:
                history = user_watch_movies_data
            else:
                history = history.append(user_watch_movies_data,ignore_index=True)

        recommand_data = None
        for movie_id,similar in recommands_movies_id:
            recommand_movies_data = self.movie_frame[self.movie_frame['movieId'] == movie_id]
            recommand_movies_data.loc[:, '相似度'] = similar
            if recommand_data is None:
                recommand_data = recommand_movies_data
            else:
                recommand_data = recommand_data.append(recommand_movies_data,ignore_index=True)

        print('用户看电影的历史（部分）：\n')
        print(history)

        print('给该用户推荐的电影（部分）：\n')
        print(recommand_data)


    def calculate(self,*args):
        """
        实现抽象类，完成UserCF的服务
        """
        target_user_id = args[0]
        top_n = args[1]
        top_n_users = self._get_topN_similar_users(target_user_id,top_n)
        candidate_movies = self._get_candidate_movies(target_user_id)
        top_n_movies = self._get_top_n_items(top_n_users,candidate_movies,top_n)
        self.print_message(target_user_id,top_n,top_n_movies)
        return top_n_movies




class ItemCf(BaseCf):
    
    def __init__(self):
        super(ItemCf,self).__init__()
        self.i2i_sim = self.itemcf_sim()

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

    def _get_topN_similar_items(self,target_item_id,top_n):
        """
        计算与target item最相似的top_n个item
        """
        target_users = self.frame[self.frame['movieId'] == target_item_id]['userId']
        other_items_id = [i for i in set(self.frame['movieId']) if i != target_item_id]
        other_users = [self.frame[self.frame['movieId'] == i]['userId'] for i in other_items_id]
        similar_list = [self._cosine_similar(target_users,user) for user in other_users]
        similar_list = sorted(zip(other_items_id,similar_list),key = lambda x:x[1],reverse = True)
        return similar_list[:top_n]

    def _get_candidate_users(self,target_item_id):
        """
        find all the users that user id did not meet the target movie before.
        """
        target_users = set(self.frame[self.frame['movieId'] == target_item_id]['userId'])
        other_movie_users = set(self.frame[self.frame['movieId']!=target_item_id]['userId'])
        candidate_users = list(target_users^other_movie_users)
        return candidate_users

    def _get_top_n_users(self,top_n_items,candidate_users,top_n):
        """
        通过感兴趣程度来对用户排序
        interest = sum(sim * normalize_rate)
        """
        top_n_item_data = [self.frame[self.frame['movieId'] == k] for k,_ in top_n_items]
        interest_list = []
        for user_id in candidate_users:
            tmp = []
            for item_data in top_n_item_data:
                if user_id in item_data['userId'].values:
                    tmp.append(item_data[item_data['userId'] == user_id]['rating'].values[0]/5.0)
                else:
                    tmp.append(0)
            interest = sum(top_n_items[i][1]*tmp[i] for i in range(len(top_n_items)))
            interest_list.append((user_id,interest))
        interest_list = sorted(interest_list,key=lambda x:x[1],reverse=True)
        return interest_list[:top_n]

    def print_message(self,target_item_id,top_n,recommands_users_id):
        mess = "基于商品的协同过滤算法,传入商品id，为商品(id:{})得到的前{}推荐的用户如下：\n".format(target_item_id,top_n)
        print(mess)

        target_movie_data = self.movie_frame[self.movie_frame['movieId'] == target_item_id]

        print('给该电影推荐的用户（部分）：\n')
        print('| 用户 | 感兴趣度 |')
        for user_id,interest in recommands_users_id:
            print("| {} | {} |".format(user_id,str(interest)))

    def calculate(self,*args):
        """
        实现抽象类，完成itemCF的服务
        """
        target_item_id = args[0]
        top_n = args[1]
        top_n_items = self._get_topN_similar_items(target_item_id,top_n)
        candidate_users = self._get_candidate_users(target_item_id)
        top_n_users = self._get_top_n_users(top_n_items,candidate_users,top_n)
        self.print_message(target_item_id,top_n,top_n_users)
        return top_n_users




