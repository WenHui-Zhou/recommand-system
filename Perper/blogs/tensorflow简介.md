[TOC]

# tensorflow简介

深度学习得到日益广泛的引用，渗透到工业界和学术界的每一个领域。

## 选择tensorflow

TensorFlow是谷歌2015年开园的框架，可以很好的支撑深度学习的各种算法。最早tensorflow是有jeff Dean牵牵头，在谷歌原有的深度框架上改造而来。基于tensorflow的一些重要的算法也在谷歌内部得到了广泛的使用。



## tensorflow环境搭建

下面介绍tensorflow最重要的工具包：Protocol Buffer以及Bazel。

### Protocol Buffer

Protocol buffer是谷歌开发的处理结构化数据的工具。机构化数据指的是拥有多种属性的数据。比如用户信息包含名字，ID等。要将这些结构化的用户信息持久化或进行网络传输，就需要先将他们序列化，即将结构化数据变成数据流的形式。序列化和反序列化就是protocol buffer主要解决的问题。

xml和json是另外两种比较常用的结构化数据：

```xml
<user>
  <name>张三</name>
  <id>12345</id>
  <email>zhangsan@acb.com</email>
</user>
```

```json
{
  "name":"张三",
  "id": "12345",
  "email":123435@asd.com,
}
```

而protocol buffer序列化后的数据是不可读的字符串。使用protocol buffer时需要先定义数据格式，算法将根据这个数据格式文件进行反序列化。因为这个区别，protocol buffer文件比xml文件要小3-10倍，解析时间要短20-100倍。

protocol buffer定义的数据格式文件一般保存在`.proto`文件中：

```protobuf
message user{
  optional string name=1;
  required int32 id = 2;
  repeated string email = 3;
}
```

message 中定义了每一个属性的类型和名字。protocol buffer中的类型可以是bool，int，float等基本类型，也可以是message。这样的结构大大增加了protocol buffer的灵活性。

此外对于每一个类型，protocol还定义了这个数据是必须的（required，必须有），可选的（optional 例如名字可以没有），重复的（repeated 即该属性是一个列表，比如有多个email）。

protocol buffer是tensorflow中重要的工具，其基本数据都是通过protocol buffer来组织的。



### Bazel

bazel是谷歌开源的自动化构建工具，谷歌内部绝大多数的应用都是通过它来编译生成的。相比于传统的Makefile，Ant或Maven，bazel 在速度，灵活性和不同语言和平台的支持上更加的出色。

**bazel 是如何工作的**











