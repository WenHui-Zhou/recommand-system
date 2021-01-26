[TOC]

# tensorflow 入门

本篇将介绍tensorflow的**计算模型**， **数据模型** ， **运行模型**， 以及神经网络的**主要计算流程**。

## 计算模型--计算图

tensorflow 从名字上看，tensor可以被理解成一个多维的数组，flow就是数据的流动。TensorFlow就是一个通过**计算图**形式来表述计算过程的编程系统。每一个计算都是计算图上的一个节点，边则表明了计算之间的关系。

### 计算图的使用

tensorflow程序一般可以分为两个阶段，第一个阶段需要定义计算图中所有的计算。第二个阶段为执行阶段。

```python
import tensorflow as tf
a = tf.constant([1.0,2.0],name="a")
b = tf.constant([2.0,3.0],name='b')
result = a + b
```

在这个过程中，系统会自动维持一个默认的计算图，通过`tf.get_default_graph`函数可以获取当前默认的计算图：

```python
# 向量通过a.graph可以取得所属的计算图
print(a.graph is tf.get_default_graph())
```

除此之外，tensorflow还可以自己定义计算图：`g1 = tf.Graph()`，不同计算图上的张量和运算都不会共享：

```python
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v",shape=[1],initializer=tf.zeros_initializer)
```

计算图还提供了管理张量和计算的机制，可以通过`tf.Graph.device('/gpu:0')` 来指定计算运行的设备。

在一个计算图中，可以通过集合来管理不同类别的资源，具体的方法是：

```python
tf.add_to_collection() # 将资源加入到集合中
tf.get_collection() # 取出一个集合中的所有资源
```

综上，tensorflow会默认生成一个计算图，然后在这个计算图上生成计算节点，完成第一步计算图的构建。

## 数据模型--张量

在tensorflow中所有的数据都是通过张量的形式来表示的，从功能上看，张量就是一个多维的数组。

**tensorflow实现张量并不是直接采用数组的形式，而是对运算结果的引用。他保存的是如何得到这些数字的运算过程。**

```python
import tensorflow as tf
a = tf.constant([1.0,2.0],name="a")
b = tf.constant([2.0,3.0],name='b')
result = tf.add(a,b,name='add')
print(result)

# Tensor('add:0',shape(2,),dtype=float32)
```

从上面过程中可以看出来，tensor保存的不是一个数字，而是一个结构：名称（name），维度（shape），类型（type）。

张量的第一个属性name，是张量的唯一表示，也给出了张量是如何得到的，name命名规则是：node:src_output，node表示运算类型，src_output表示该类型的第几个输出。`add:0`表示add的第一个输出。

tensor在进行计算的时候，会检查张量的类型，如果类型不匹配的话则会报错。例如将int32 + float32将会报错，因此我们在创建变量或常量的时候，通常会指定类型而不是使用默认的类型。

```python
a = tf.constant([1,2],name='a',dtype=tf.float32)
```

注意到创建一个常量的时候，还有一个参数是name，和上呼应，tensor中存储的是三个属性，其中name即由该参数传入。a这个tensor保存着结果的应用。

### 张量的使用

张量有两大用途，第一个是对中间结果的引用。例如要计算两个constant相加，就可以用张量来表示这两个constant（中间结果），然后两个张量相加。

第二个用途是根据张量来获取计算结果。这种情况是当计算图构建完成，启用会话之后，可以通过会话计算得到当前tensor中所保存的值：`tf.Session().run(a)`。

## 运行模型-- 会话

tensorflow程序在执行的时候，需要大量的系统资源。会话（session）机制管理着程序运行时的所有资源，当所有计算完成之后需要关闭会话来回收资源。可以将会话看成是一个上下文环境，当我们要执行代码的时候，需要在指定的会话里头进行。

开启会话的方式：

```python
# 法1，自己开启，自己关闭
sess = tf.Session()
...
sess.run(...)
sess.close()

#法2，使用with上下文管理器
with tf.Session() as sess:
  ...
  sess.run()
```

tensorlfow会自动生成计算图，所有的运算会自动加入到这个图中。会话也类似，但是tensorflow不会默认生成会话，需要手动指定：

```python
sess = tf.Session()
with sess.as_default():
  result.eval()
```

有点像是切入到具体的运行环境，如果没有默认的就需要自己确定使用哪个session进行运算，如果有默认的，直接对tensor取eval()函数，即可得到张量的取值。系统使用默认的session对他进行计算。

因此张量有两种取值方式，一种是在with下的session中，用`sess.run()` 的方式，另一种是指定默认的session，用`tensor.eval()`的方式。

交互式session，系统会指定默认的session，使用一些交互式场景如jupyter：

```python
sess = tf.InteractiveSession()
print(result.eval())
```



## Tensorflow 计算过程--实现神经网络为例











