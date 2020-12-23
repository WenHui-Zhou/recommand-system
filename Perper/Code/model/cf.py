#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# Author      : perperzhou <765647930@qq.com>
# Create Time : 2020-12-23
# Copyright (C)2020 All rights reserved.

import math
import pandas as pd
from abc import ABC,abstractmethod

pd.options.mode.chained_assignment = None  # default='warn'

class BaseCf(ABC):

    def __init__(self):
        self.file_root = './data/ml-latest-small/'
        self._init_frame()
    
    def _init_frame(self):
        self.frame = pd.read_csv(self.file_root + 'ratings.csv')
        self.movie_frame = pd.read_csv(self.file_root + 'movies.csv')
        

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
        similar_list = [self._cosine_similar(target_movies,movies) for movies in other_movies]
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
        实现抽象类，完成itemCF的服务
        """
        target_user_id = args[0]
        top_n = args[1]
        top_n_users = self._get_topN_similar_users(target_user_id,top_n)
        candidate_movies = self._get_candidate_movies(target_user_id)
        top_n_movies = self._get_top_n_items(top_n_users,candidate_movies,top_n)
        self.print_message(target_user_id,top_n,top_n_movies)
        return top_n_movies




