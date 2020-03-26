# 采样 {ignore=true}

[TOC]

## 统计抽样理论

样本量计算公式

$$
n = \frac { z ^ { 2 } S ^ { 2 } } { e ^ { 2 } + \frac { z ^ { 2 } S ^ { 2 } } { N } }
$$

当整体方差未知的时候，可以用 `p(1-p)` 估计（最大估计）

$$
S ^ { 2 } = P ( 1 - P ) \\\
n = \frac { z ^ { 2 } P ( 1 - P ) } { e ^ { 2 } + \frac { z ^ { 2 } P ( 1 - P ) } { N } }
$$

## 高维采样方法

### Alias method （别名采样法）

一种离散分布抽样方法。

> You are given an n-sided die where side $i$ has probability $p_i$ of being rolled. What is the most efficient data structure for simulating rolls of the die?

最简单的想法： 概率线段。
要想判断落在概率线段上的具体哪一段，自然可以用二分搜索的办法，这是 $O(\log n)$ 的复杂度。
特别的，如果每段的概率都是一样的，则直接乘 n 即可，这是 $O(1)$ 的复杂度.

进一步，对每段概率不一样的，其实也可以强行转为一样的，最后标记一下就好了，例如 1 2 3 全都标记为 1, 这样仍然是 $O(1)$ 的复杂度！ （但是前期的工作可能多一些，因为要求最小公因子，还要做 map，这部分的复杂度还是 $O(\log n)$）

参考 [darts-dice-coins](https://www.keithschwarz.com/darts-dice-coins/)
