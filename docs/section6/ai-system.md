# AI 系统 {ignore=true}

[TOC]

> 算法-系统-架构 一体化

## 系统设计

### 并行

RAPIDS

### 向量化索引

faiss

[SPTAG](https://github.com/Microsoft/SPTAG)

Space Partition Tree And Graph
分布式的最近邻搜索库

## Online Deep Learning

### 内存同步机制

## 模型工程

### PS

data parallelism。每台 worker 只利用本地的训练数据前代、回代，计算 gradient，并发往 server。Server 汇总（平均）各 worker 发来的 gradient，更新模型，并把更新过的模型同步给各 worker。这里有一个前提，就是数据量超大，但是模型足够小，单台 server 的内存足以容纳。

[byteps](https://github.com/bytedance/byteps)
字节跳动开源的 分布式 DNN 训练框架。宣称速度是 Horovod 的 2 倍！
需要 CUDA and NCCL， 不支持仅 CPU

## Ai 芯片

硬件加速

### VTA

Versatile Tensor Accelerato， 灵活的张量加速器

## 参考

- [AI-Sys 2019](https://ucbrise.github.io/cs294-ai-sys-sp19/)
- [Weld](https://www.weld.rs/)
