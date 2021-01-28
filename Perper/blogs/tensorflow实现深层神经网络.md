[TOC]

# tensorflow 实现深层神经网络

## 深度学习与深层神经网络

深度学习有着两个非常重要的特征，即多层和非线性。

### 非线性

倘若只有线性变换，任意层的全连接层神经网络和单层的神经网络模型的表达能力没有任何的区别，因此我们需要在网络中引入非线性部分。

在tensorflow中提供了7种不同的非线性激活函数，`tf..relu,tf.sigmoid,tf.tanh`是比较常用的几个。下面代码展示了网络前向传播的过程：

```python
a = tf.nn.relu(tf.matmul(x,w1) + bias1)
y = tf.nn.relu(tf.matmul(a,w2) + bias2)
```

### 多层网络

最早的神经网络是单层的感知机，然而单层的网络难以解决异或的问题。深层神经网络实际上有着组合特征提取的功能，能够拟合复杂的数据，能够解决异或的问题。

## 损失函数的定义

损失函数是衡量网络输出结果和GT之间的差异的方法。通常可以分为分类问题，和回归问题。

### 分类问题损失函数

例如对于一个k分类的问题，正常情况下一个样本属于只属于一个类别，其他项都为0。例如[0,0,1,0,0]。因此我们希望网络的输出和期望的向量越接近越好。交叉熵是常用的评判标志之一，他能够刻画两个概率分布之间的距离。
$$
H(p,q) = -\sum_xp(x)log q(x)
$$
因此我们希望模型的输出结果每个位置的值也在0-1之间。一个常用的变形是使用softmax：
$$
softmax(y)_i = \frac{e^{y_i}}{\sum_{j=1}^{n}e^{y_j}}
$$
上一篇文章中提到的损失即是交叉熵损失：

```python
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1)))
```

其中`tf.clip_by_value`将输出的张量限制在一个范围内，防止出现log0等这种错误。

`tf.log` 作用是逐个将tensor中的元素取log。

`tf.reduce_mean` 对矩阵中所有数据相加去平均。

含义就是对于一个batch中每一个样本都算交叉熵，然后取平均交叉熵作为这个batch的熵。

因为交叉熵一般会与softmax一起使用，因此tensorflow对这两个功能进行了封装：

```python
cross_entropy = - tf.nn.softmax_cross_entropy_with_logits(
label = y_,logits=y)
```

在只有一个正确答案的分类问题中tensorflow提供了更加高效的实现：

```python
tf.nn.sparse_softmax_cross_entropy_with_logits
```

### 回归问题

当我们要对具体的数值进行预测的时候，例如房价，销售额等，这就是一个经典的回归问题。回归问题常使用的损失是MSE损失：
$$
MSE(y_,y) = \frac{1}{n} \sum_{i=1}^{n}(y_ - y)^2
$$
tensorflow 的实现如下：

```python
mse = tf.reduce_sum(tf.square(y_ - y))
```

### 自定义损失函数

由于一些具体的情况，难以直接使用现成的损失，因此我们就需要自定义损失。例如计算销售额，多卖出去一件商品赚10元，多进货一件商品亏1元，当我们在预测商品数量的时候，应当有某种倾向，倾向于多预测商品数量，因为期望更高。因此不能简单实用MSE损失，可以定义如下损失：

```python
loss = tf.reduce_sum(tf.where(tf.greater(v1,v2),(v1-v2)*a,(v2-v1)*b))
```

`tf.greater`是逐元素判断大小的。`tf.where`也是逐元素进行选择。这里可以把元素理解成一个样本得到的结果。

### 代码

下面定义一个简单的网络模型，说明损失的作用：[代码地址](../Code/tensorflow/4_1.py)

可以看出，经过5000轮迭代之后，参数输出为[[1.0193471，1.0428091]]，即大于1，网络倾向于预测更大的商品数量。