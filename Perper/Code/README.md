# 推荐算法实践

## 代码思路

manage选择算法-> workflow加载具体算法的工作流-> model调用模型进行计算

## 说明
代码有一部分使用tensorflow1.x，一部分是2.x，实现的，现在考虑到大部分公司使用tensorflow1.x,接下来代码打算迁移到1.x上[2021.01.19]

## 已实现的算法

- 协同过滤算法[tf1.x]
- word2vec算法[tf1.x]
- FM算法[tf1.x]
- GBDT算法[tf1.x]
- gcforest算法[tf1.x]
- LR算法[tf1.x]
- DNN算法[tf2.x]
- WDL算法[tf2.x]
- deepfm算法[tf2.x]
- youtube算法[tf2.x]
- youtube_rank算法[tf2.x]
- mut_recall算法[tf1.x]



## 代码执行方法

python train_main.py [item_cf/user_cf/gbdt/gbdt_lr/gcforest/word2vec/LR/FM/dnn/wdl/deepfm/youtube/youtube_rank,mut_recall]
