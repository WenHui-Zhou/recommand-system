[TOC]

# mnist 数据识别问题

## 前言

下面将从mnist数字图像识别的例子中，简单地实现一个tensorflow代码。

## mnist 数据处理

tensorflow中提供了mnist的数据集，我们可以通过下面的方式直接得到：

```python
from tensorflow.examples.tutorials.mnist import input_data
# 如何./data中没有数据的话，这个函数就回去下载这个数据集
mnist = input_data.read_data_sets('./data/',one_hot)
# 随后便可以得到三个数据集
print(mnist.train.num_examples)
print(mnist.validation.num_examples)
print(mnist.test.num_examples)
print(mnist.train.labels[0]) # onehot的形式
```

为了方便训练，mnist还提供了next_batch函数：

```python
batch_size = 100
xs,ys = mnist.train.next_batch(batch_size)
```

### tensorflow 实现mnist初版

代码位置：[地址](../Code/5_1.py)













