# 知识图谱(knowledge graph)

[TOC]

![](./img-knowledge_graph/2019-06-04-19-19-58.png)

##### 知识图谱：机器大脑中的知识库

知识图谱旨在描述真实世界中存在的各种实体或概念。其中，每个实体或概念用一个全局唯一确定的 ID(标识符, identifier)来标识。每个属性-值对（attribute-value pair，又称 AVP）用来刻画实体的内在特性，而关系（relation）用来连接两个实体，刻画它们之间的关联。

##### 存储方式

1. RDF(Resource Description Framework, 资源描述框架)
2. 图数据库(graph database) 例如 neo4j, 这种存储方式现在比较流行，添加新的数据源较为方便

##### 知识表示

- 知识表示代表模型：TransE

  对事实（head, relation, tail），将 relation 看做 head 到 tail 的翻译

  存在的问题：无法解决一对多，多对一和多对多的情形

- 关系路径的表示和建模
-

* 知识图谱的嵌入（knowledge graph embedding）

  即将实体和关系投影到连续的向量空间中去

* [TransG: A Generative Mixture Model for Knowledge Graph Embedding](http://arxiv.org/pdf/1509.05488.pdf)

#### 知识图谱构建

- 数据来源：百科、freebase 　知识、垂直站点的结构化数据、半结构化／非结构化数据挖掘（包括搜索日志）

* 构建过程

  从异构数据源抽取构建知识图谱所需的各种候选实体（概念）及其属性关联，形成了一个个孤立的抽取图谱（ExtractionGraphs）。为了形成一个真正的知识图谱，还需要将这些信息孤岛集成在一起

前面说过，实体要有全局唯一标识，这需要实体对齐。
不一致性的处理

- 维护更新

[Holographic Embeddings of Knowledge Graphs by Maximilian Nickel, et al., 2015](http://arxiv.org/abs/1510.04935)

题目非常炫酷：知识图谱的全息嵌入。ｙ

#### 知识图谱挖掘

构建好知识图谱之后，就能进行基于知识图谱的挖掘了。

- 关系抽取

- 知识推理
  　　　针对属性的，比如，通过出生年月得到年龄

针对关系的，比如，爸爸的爸爸是爷爷

- 相关实体挖掘

- 实体排序

#### 知识图谱实践

知识图谱在智能搜索、反欺诈、异常检测等领域的应用

[Linked Data - Connect Distributed Data across the Web](http://linkeddata.org/)

该项目旨在构建一张计算机能理解的语义数据网络

Google Knowledge Graph

百度知心

搜狗知立方

wikidata

##### 产品化细节

上翻后引导

直接答案满足

Google Knowledge Vault（参见 [论文笔记 1-Knowledge Vault: A Web-Scale Approach to Probabilistic Knowledge Fusion](https://blog.csdn.net/zhang201322/article/details/78790258)）

整个系统主要由三大部分组成：

Extractors：三元组抽取器
Graph-based priors：基于图的先验概率计算器
Knowledge fusion：知识融合器

Path ranking algorithm (PRA)（路径排名算法）

## 参考

- [知识图谱技术原理介绍 by 王昊奋](http://weibo.com/p/23041872d083c70102vye8)
