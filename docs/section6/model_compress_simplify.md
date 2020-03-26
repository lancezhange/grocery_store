# 模型压缩精简 {ignore=true}

[TOC]

工业上，一些在线模型，对响应时间提出非常严苛的要求，从而一定程度上限制了模型的复杂程度。模型复杂程度的限制可能会导致模型学习能力的降低从而带来效果的下降。

目前有 3 种思路来解决这个问题

1. 通过压缩、编码等方式减小网络规模。量化是最广泛采用的压缩方法之一。
2. 设计更有效的网络架构，用相对较小的模型尺寸达到可接受准确度，例如 MobileNet 和 SequeezeNet。

参数剪枝和共享，低秩分解和稀疏性，传递/紧凑卷积滤波器和知识蒸馏等。

## 压缩精简方法

量化有若干相似的术语。低精度（Low precision）可能是最通用的概念。常规精度一般使用 FP32（32 位浮点，单精度）存储模型权重；低精度则表示 FP16（半精度浮点），INT8（8 位的定点整数）等等数值格式。不过目前低精度往往指代 INT8。

混合精度（Mixed precision）在模型中使用 FP32 和 FP16 。 FP16 减少了一半的内存大小，但有些参数或操作符必须采用 FP32 格式才能保持准确度。

另外一个想法是如何压缩整个模型而非存储一个元素的位数。



### SequeezeNet

使用一部分 1\times1 卷积代替 3\times3 卷积，它对标的模型是AlexNet。

### MobileNetV1

**深度可分离卷积**


### Xception




### 二进制神经网络

### 三元权重网络

权重约束为 +1, 0 和 -1 的神经网络

### ShuffleNet

### Mixed-Precision Training of Deep Neural Networks 。

### MobileNetV1

仅为 4.8MB，这甚至比大多数 GIF 动图还要小！从而可以轻松地部署在任何移动平台上。

### Rocket Training

阿里巴巴
利用复杂的模型来辅助一个精简模型的训练，测试阶段，利用学习好的小模型来进行推断。

### 知识蒸馏

knowledge distilling

将复杂模型(teacher)的知识迁移到简单模型(student)中去，这样相当于在保持精度的同时减少了模型的复杂度，然后简单模型可以直接开跑，不需要像之前做量化那样做定点化了。

### 2019-ICML-Google-EfficientNet

[EfficientNet](): Rethinking Model Scaling for Convolutional Neural Networks






## 参考

- [A Survey of Model Compression and Acceleration for Deep Neural Networks](https://arxiv.org/abs/1710.09282)
