# 分布式系统

[TOC]

分布式系统的唯一核心终极问题：如何确保一致？分布式理论的几乎所有的东西都是围绕这个核心问题展开。

分布式系统的三大方面

1. 时间模型：同步，异步（区别于处理请求的异步），半同步

   时钟同步算法, 逻辑时钟

2. 中间交流：　信息传递，共享内存

3. 容错模型：如果知道哪些组件失效(failure detection)，如果恢复？

   涉及　 leader 选举，议会(QUORUMS)

##

分布式一致性协议 Paxos

参考　[可靠分布式系统基础　 Paxos 的直观解释](http://drmingdrmer.github.io/tech/distributed/2015/11/11/paxos-slide.html)

### CAP 定理

一个分布式系统必须在一致性(Consistency)、可用性(Availability)和分隔容忍度（Partition tolerance）之间取舍，并且只能做到三者中的两者。(注意，三个维度都应该理解为范围，而不是布尔值)

分布式系统最大的问题可能并不是延迟，而是部分机器宕机（尤其是单点故障），因此分布式系统设计需要'design for failure'（容灾设计）.

### 复制

针对数可能丢失问题，一般的做法是数据的冗余存储，也就是存储多个副本，例如 hadoop 默认备份 3 份

复制算法有：　主从异步复制，主从同步复制，主从半同步复制，多数派写（读）

数据一致性的保证： 事务
系统性能

- 拜占庭将军问题

---

### 分布式数据存储

关系型数据库(RDB)支持**高度一致性**事务，然而在互联网等领域，相比一致性，**可用性**、**性能**、**可扩展性**可能会更为重要。

关系型数据库围绕着 **_ACID_** 的概念而建立，实现 ACID 的最简单方法就是将一切都保持在单机上，因此，传统的 RBD 通常只能垂直型扩展。

##### Partitioned Replica Set

<div>
    <figure>
        <img src="./img-distributedSystem/2020-03-23-20-53-43.png" style="width:500px" />
        <figcaption align='center'>来源 http://www.kuqin.com/shuoit/20150925/348279.html</figcaption>
    </figure>
</div>

---

### 分布式机器学习系统

参数服务器

##### 参考

- [讲给普通人听的分布式数据存储](http://www.kuqin.com/shuoit/20150925/348279.html)

- [what we talk about when we talk about distributed systems](http://videlalvaro.github.io/2015/12/learning-about-distributed-systems.html)

- [分布式计算原理（讲稿合集）](http://dcg.ethz.ch/lectures/podc_allstars/)
