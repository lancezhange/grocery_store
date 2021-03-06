# 区块链

[TOC]

2018 年伊始，区块链的火就烧的越来越厉害，不了解了解似乎都要 out 了，抱着学习的心态，也了解一下。
so， 什么是区块链

最简单的表述： **分布式数据库**

区块链技术最早的应用是比特币，以至于提起区块链总以为它和比特币之间有某种等同关系，其实不然。

区块中记录了交易信息，以及上一个区块的 hash 值，因此构成链条。

区块链提供了通过机器算法解决参与人之间的信任问题的全新方案，其核心的核心就是在不完全信任的各方，通过复杂的密码学技术，可以有效防止记录被篡改。一经产生，无法修改。

由于是分布式数据库，因此具有*去中心化*的特征。

合法性：超过 50% 的节点认证才能视为有效，并写入账本。

那么问题来了

1. 涉及的密码学包括哪些

2) 隐私
   如何保护隐私？
   隐私和不可篡改其实有些相互矛盾，要实现不可篡改，就得让其他人来验证数据，比如公有链是全网用户都来验证；但是隐私又想只有授权的人才可以验证，甚至希望其他人能验证但是不知道数据，比如盲签名、同态算法等

全同态算法

零知识证明

3. 交易的性能
   P2P 网络通信的效率是对性能的影响

4) 监管

## 密码学原理

[高深的密码学+复杂的区块链，其实也可以通俗易懂](http://rdc.hundsun.com/portal/article/750.html)
对称加密又叫传统密码算法，就是加密和解密使用同一个密钥
常见的对称加密方法有 DES、3DES、Blowfish、RC2、AES 以及国密的 SM4。
有同学会问，什么是国密啊？很机密么？没那么夸张，其实它的全称叫“国家商用密码”，是为了保障商用密码安全，国家商用密码管理办公室制定了一系列密码标准

非对称加密
除了 RSA 之外，常见的非对称算法还有 Elgamal、背包算法、Rabin、D-H、ECC（椭圆曲线加密算法）以及国家商用密码 SM2 算法。

题外话：强烈不建议使用 RSA，原因如下：

- 容易被破解：RSA-768 可以在 3 个小时内破解，1024 在理论上 100 小时内也可以破解。所以使用 RSA，长度起步要 2048。但是数学家彼得·舒尔研究了一个针对整数分解问题的量子算法 (舒尔算法)，理论上破解 2048 的 RSA 在 100 秒之内(好在量子机还未投入使用)。
- 慢：密钥长度加到 2048 可以提升安全，但是计算过慢。
  目前，常用椭圆曲线算法 – ECC 来做非对称加密基础算法。ECC 的 210 位算法难度就相当于 RSA 2048 的难度，性能则是数量级的区别。国密中的 SM2 就是基于 ECC 算法的。

哈希算法
常见的摘要算法有 MD5、RIPEMD、SHA 和国密的 SM3。MD5 不建议使用，已经被爆。

哈希指针：结合了内容和位置的 hash

共识机制

## 区块链技术的应用

- [基于区块链技术的去中心化云计算](https://iex.ec/)
  隶属于 企业以太坊联盟(EEA)

## 哈希图(HashGraph)

哈希图是中心化技术的未来？

特点

- 无需工作量证明即可实现共识机制
- 高交易吞吐量
  - 比特币的交易限制为 7 笔/秒。Hashgraph 的交易量高达 250000 笔/秒。

[HashGraph_Explained](https://github.com/llSourcell/HashGraph_Explained)

## 以太坊

以太坊被称为区块链 2.0 时代

Vitalik Buterin（V 神）
: 以太坊联合创始人，著有《以太坊白皮书》，生于 1994 年，获得 2014 年世界科技奖
2016 年 5 月因 The DAO(Decentralized Autonomous Organization) 被盗事件，**以太坊硬分叉**。一派是以太经典（Etherum Classic），他们坚持区块链不可篡改的理念，继续维护原有的以太坊；另一派则是现在的以太坊，由 V 神带队，在数据回滚后的新链上继续开发。
![](./img-block_chain/2019-05-14-23-24-26.png)
