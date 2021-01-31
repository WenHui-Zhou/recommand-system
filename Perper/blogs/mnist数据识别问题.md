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

**代码解释：**

<img src = '../images/mnist_2.png'>

展示了模型的网络结构，即一层全连接层，激活函数为relu，随后又接一层全连接层。

<img src = "../images/mnist_3.png">

<img src = "../images/mnist_4.png">

上述过程定义了网络训练的参数，以及网络的损失函数，网络选择的优化器。预测结果，模型的训练过程。

**代码结果：**

<img src = '../images/mnist_1.png'>



## 变量管理

















