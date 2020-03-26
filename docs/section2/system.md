# 系统设计专题 {ignore=true}

[TOC]

## 分布式 id

如何设计一个分布式 ID 生成器(Distributed ID Generator)？

**需求**

1. 全局唯一
2. 尽可能短以节省内存
3. 按时间粗略有序

### UUID

4 个字节表示的 Unix timestamp,
3 个字节表示的机器的 ID
2 个字节表示的进程 ID
3 个字节表示的计数器

总共 12 个字节，太长了！

### 多台 mysql 服务器

假设用 8 台 MySQL 服务器协同工作，第一台 MySQL 初始值是 1，每次自增 8，第二台 MySQL 初始值是 2，每次自增 8，依次类推。前面用一个 round-robin load balancer 挡着，每来一个请求，由 round-robin balancer 随机地将请求发给 8 台 MySQL 中的任意一个，然后返回一个 ID。

## 短链

当前互联网上的[网页总数大概是 45 亿](0http://www.worldwidewebsize.com)，45 亿超过了 2^{32}=4294967296，但远远小于 64 位整数的上限值，那么用一个 64 位整数足够了。

### 301 还是 302 重定向

301 是永久重定向，302 是临时重定向

## 信息流

## 定时任务调度

## API 限速

## KV 存储引擎

目前开源的 KV 存储引擎中，RocksDB 是流行的一个

RocksDB 最初是从 LevelDB 进化而来的

有一个反直觉的事情是，**内存随机写甚至比硬盘的顺序读还要慢**，磁盘随机写就更慢了

## 异步框架

### SeaStar

现代硬件上的高性能 C++异步框架 - SeaStar

说到 Seastar，不得不说 Scylla。Scylla 是 2015 年 9 月开源，由大神 KVM 之父 Avi Kivity 创建的 NoSQL 数据库，接口协议完全兼容 Cassandra，但性能号称快了 10 倍：每节点 1 millionIOPS。Scylla 完全基于 Seastar 库，由 C++改写的 Cassandra。

所以 Scylla 的惊鸿面世也带来了大家对于 Seastar 的瞩目。实际上，Scylla 只是 Seastar 的一个应用，其他应用如 Pedis 也佐证了 Seastar 是整个应用性能提升的基石.

## 参考资料

关于分布式系统设计问题，参考[分布式系统](./distributedSystem.md)

关于 AI 系统设计问题，参考 [AI 系统设计](../section6/ai-system.md)

- [系统设计面试题精选](https://legacy.gitbook.com/book/soulmachine/system-design/details)
