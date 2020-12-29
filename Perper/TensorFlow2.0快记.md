[TOC]

# TensorFlow 2.0 快记

## 数据读取



## 数据的预处理



## 超参设置



## 模型搭建



## 优化器以及损失函数



## 度量方法



## 训练过程



## 模型的加载与保存



## 使用anaconda 创建、安装tensorflow2.0 虚拟环境

### conda 使用语法

- 创建环境：conda create –name <env_name> <package_names>
- 激活环境：activate <env_name>
- 退出环境：deactivate
- 删除环境：conda remove –name <env_name> –all
- 环境列表：conda env list

### 安装步骤

1. 创建独立环境并激活

```
conda create --name tensorflow2.0 python==3.7
conda activate ~/anaconda3/envs/tensorflow2.0/
```

2. 安装TensorFlow2.0
   pip install tensorflow==2.0.0-beta -i https://pypi.tuna.tsinghua.edu.cn/simple