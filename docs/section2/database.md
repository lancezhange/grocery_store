# 数据库

- 主键： 表的唯一索引，如 id int(10) not null primary key auto_increment;
- 索引： 索引能够加快检索速度（但其创建和维护也需时间，同时也占用一定空间）
    - 添加索引的方式： alter table TABLE_NAME add index INDEX_NAME (COL_NAME);




### MySQL


to read:

[单表60亿记录等大数据场景的MySQL优化和运维之道](http://mp.weixin.qq.com/s?__biz=MzAwMDU1MTE1OQ==&mid=209403337&idx=1&sn=f99429e24e8c591111a355e072f93e05) 


### NoSQL


[NoSQL 数据库模型](http://darkhouse.com.cn/blog/4?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io)

#### Mongodb



### Redis
开源，先进的key-value存储，并用于构建高性能，可扩展的Web应用程序的完美解决方案。
特点：
1. 数据库完全在内存中，使用磁盘仅用于持久性。
2. 相比许多键值数据存储，Redis拥有一套较为丰富的数据类型。
3. Redis可以将数据复制到任意数量的从服务器。


### PostgreSQL

### SQLite



### 大型数据库
sharding （分片）
摘自[初识 sharding 技术](http://www.elecfans.com/news/wangluo/20120215260321.html)

数据库的扩展是一个永恒的话题。传统的关系数据库，采用的是纵向扩展(scale up)的方式，即买更好的机器添加更多的资源来提升性能。但近年来随着数据量的暴增，这种扩展方式已经不能满足需求。因此出现了横向扩展（scale out）模式，这种方式采用一些Ad-hoc的技术，比如说对数据库进行主从配置(Master-Slave)、采用数据库复制(Replication)技术以及服务器的缓存(Server Cache)等，来将负载分布到多个物理节点上去。另外sharding技术也逐步发展，并在近年来吸引了众人的眼球。

形式上，Sharding可以简单定义为将大数据库分布到多个物理节点上的一个分区方案。每一个分区包含数据库的某一部分，称为一个shard，分区方式可以是任意的，并不局限于传统的水平分区和垂直分区。一个shard可以包含多个表的内容甚至可以包含多个数据库实例中的内容。每个shard被放置在一个数据库服务器上。一个数据库服务器可以处理一个或多个shard的数据。系统中需要有服务器进行查询路由转发，负责将查询转发到包含该查询所访问数据的shard或shards节点上去执行。

sharding和分区还是有很大区别的。下面罗列一下：
(1)扩展方式不同。Sharding属于scaleout，而分区则属于scale up方式。
(2)目的不同。分区的目的是为了将一个查询进行并行处理，这样所有的节点能并行处理一个查询;而sharding是让每个
节点尽量处理不同的查询。
(3)应用场景：分区适用与传统的企业应用，尤其是OLAP的应用，基本上每个查询都需要访问大部分的数据;
而sharding适用于云Web应用，特征是有大量的用户和查询，但是每个查询访问到的元组是非常少的，sharding可以
将负载分散到多个物理节点上。
(4)可用性：对于分布式数据库基本上每个查询都需要所有的节点参与，如果某些节点down掉后，系统会大受影响;
而sharding所处理的应用一般只涉及到少数几个节点，所以可用性上sharding要好一些。另外分布式数据库需要有一个
主节点来生成执行计划并协调相关节点执行等，很容易形成单点瓶颈。
(5)分割粒度：分区一般只针对于一个数据库内部进行分割;而sharding可以以数据库为粒度进行分割，
因此可用来构建多租房数据库系统(multi-tenantdatabase)。
