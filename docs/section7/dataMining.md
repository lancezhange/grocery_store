# 数据科学工具

武库中，要有匕首短刀，也要有大刀长矛，要有神鞭，也要有铁锤，总之要丰富，在不同的场景下用称手的武器，才能手到擒来。

随着越来越多的机器学习工具加入开源社区，整个机器学习工具领域呈现出百花齐放的局面。

### Spark

Spark 被认为是 **_smartphone of data_**

- MLlib
- GraphX
- Spark SQL
- Spark Streaming

其他基于　 spark 的第三方库

- [spark-ts](https://github.com/cloudera/spark-timeseries)

### Flink

### Hadoop

### Scikit-Learn

### H2O

神说，有了火(spark)，当然也要有水，于是 H2O 横空出世。

### Weka

### Vompal Wabbit

### caffe

### torch

### theano

### [Keras](http://keras.io/)

**`Keras`**在德语中的意思为‘号角’

### [TF(tensorFlow)](http://tensorflow.org/)

谷歌开源的第二代深度学习系统，参看 [Jeff Dean 的介绍](http://vdisk.weibo.com/s/tQWXdywczVZO) 和[官方白皮书(译)](http://www.jianshu.com/p/65dc64e4c81f).

第一代是 DistBelief, 其可扩展性较好，但灵活性不足

然而 TF 一出，对其[性能的评测结果](https://github.com/soumith/convnet-benchmarks/issues/66)却并不理想，被戏称为 _TensorSlow_，一时间舆论大哗。

### [DMTK](http://www.dmtk.io/)

DMTK(distributed machine learning toolkit)，由微软开源，包含 DMTK 分布式机器学习框架，LightLDA, Distributed (Multisense) Word Embedding 等组件。

### [MXNet](https://github.com/dmlc/mxnet/)

DMLC 出品。CXXNet 的进化版，支持 python, R, Go, Julia 等多种语言接口，绝对是深度学习领域最为野心勃勃的项目之一。高质量的[设计文档](http://mxnet.rtfd.org/)和示例，是学习的好材料。

特色/优势

1. 借助 mshadow 模块，CPU 与 GPU 不必在代码层面区分，即只写一份代码（这是咋做到的？）。
2. 符号式编程(symbolic programming)和命令式编程的结合。

   符号式编程将定义运算图的步骤和编译运算的步骤分离。
   我个人理解，符号式编程和命令式编程的区别在于，拿多项式求值作例子，后者是直接往式子里代入数值，然后一步一步计算，而前者先做多项式化简，到最后没法化简了再代入数值求结果。有些多项式看着比较复杂，但经过化简可能最后就剩一两项，因此符号式编程更为节省计算量。此外，符号式编程由于明示了最后的结果(计算的边界)，中间结果一概不管，因此可以重复利用内存，节省空间占用，当然，命令式程序也有垃圾回收机制，但因为是编译器通用的垃圾回收策略，可能并不会像符号式程序里的效率这么高。

3. 利用参数服务器，容易分布式扩展

参见　[MXNet 设计和实现简洁]，by 李沐, et al.

### [SystemML](http://systemml.apache.org/)

Apache SystemML，IBM 开源的分布式、命令式机器学习平台，目前仍然处于孵化阶段

### [MicroSoft Azure](https://login.microsoftonline.com/common/oauth2/authorize?api-version=1.0&client_id=0736f41a-0425-4b46-bdb5-1563eff02385&response_type=id_token&scope=openid&redirect_uri=https://login.azureml.net/response.aspx&response_mode=form_post&state=4dabebe2f8bd438d99a6b735b405fb1a&nonce=635846325692222953.ZWQ5MmM1OTYtYTNmMS00NGIxLThjYjctMDJlOTRjMDM5NDI3NTkwM2MyNzctNjEzYS00MzVmLTk3NmItOWU3NDNhZGMzMjA2)

[OpenCC](https://github.com/BYVoid/OpenCC)
中文繁简转换
