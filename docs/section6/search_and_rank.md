# 搜索排序 {ignore=true}

[TOC]

## 传统方法

传统的排序模型主要可以分为以下两类：

1. 相关度排序模型，例如 BM25
2. 重要性排序模型，例如 PageRank

这些传统的排序模型往往**只能考虑某一个方面**(相关度或者重要性)，而机器学习方法能够利用的信息更多。例如，可以包括文档本身的一些特征、相关性得分、重要性排序模型的输出得分，等等。

### BM25

$$
\operatorname { Score } ( Q , d ) = \sum _ { i } ^ { n } W _ { i } \cdot R \left( q _ { i } , d \right)
$$

$$
R \left( q _ { i } , d \right) = \frac { f _ { i } \cdot \left( k _ { 1 } + 1 \right) } { f _ { i } + K } \cdot \frac { q f _ { i } \cdot \left( k _ { 2 } + 1 \right) } { q f _ { i } + k _ { 2 } }
$$

其中

$$
K = k _ { 1 } \cdot \left( 1 - b + b \cdot \frac { d l } { \operatorname { avgdl } } \right)
$$

## 机器学习排序方法

参考 [Learning to Rank](./ltr.md)

## 深度学习方法

DSSM （Deep Structured Semantic Model， 注意和 ESMM 区别）
基于深度网络结构的语义模型

其核心思想是将 query 和 doc 映射到到共同维度的语义空间中，通过最大化 query 和 doc 语义向量之间的余弦相似度，从而训练得到隐含语义模型，达到检索的目的。DSSM 有很广泛的应用，比如：搜索引擎检索，广告相关性，问答系统，机器翻译等。

word hasing: n-gram (类似于我之前在做门店匹配时候的工作)

## 评价方法

### NDCG

### 肯德尔等级相关系数（Kendall Tau）

其实就是逆序数

但是如果两个列表的长度不一致咋办？

### RBO

Rank Biased Overlap

## 数据获取

人工标注还是代价太大，因此一般还是从日志中挖掘。

例如，给定一个查询，搜索引擎返回的结果列表为 L，用户点击的文档的集合为 C，如果一个文档 $d_i$ 被点击过，另外一个文档 $d_j$ 没有被点击过，并且 $d_j$ 在结果列表中排在 $d_i$ 之前，则 $d_i>d_j$ 就是一条训练记录.

## 召回

### 乘积量化(Product Quantization)

## 参考

1. 深度学习在搜索业务中的探索与实践
2.
