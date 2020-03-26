# 统计基础 {ignore=true}

[TOC]

统计三大分析：回归分析、方差分析、多元分析

@import "./statistic_distribution.md"

## 充分统计量

## 极大似然 MLE

对数似然技巧

## Statistical Power

## Jansen 不等式

若是一个凸函数 2，且是一个随机变量，那么有

$$
E [ f ( X ) ] \geq f ( E [ X ] )
$$

当且仅当 $X = E[X]$ 概率为 1 时（即为一个常数时），等式成立。

$AIC = -2 对数似然　+ 2待估参数的个数$
不过就是加了惩罚的对数似然

$BIC = AIC + k(log(T) - 2)$
从这里可以看到，BIC 对参数的惩罚是更厉害的

- 共轭分布

  [几何视角看共轭先验](http://www.umiacs.umd.edu/~arvinda/mysite/papers/conjugate.pdf)：选取共轭先验并不仅仅是出于数学上的方便，还有更深层的原因。（很数学化的一篇论文，没耐心读完啊－－－）

@import "em.md"

- GAM(广义可加模型)

[GAM: The Predictive Modeling Silver Bullet](http://multithreaded.stitchfix.com/blog/2015/07/30/gam/?utm_campaign=Data%2BElixir&utm_medium=email&utm_source=Data_Elixir_47)

### 概率图模型(PGM, Probablistic Graphical Model)

参见 [概率图模型](../section6/pgm.md)

### Dirichlet Process 　狄利克雷过程专题

- [Dirichlet Process](http://www.gatsby.ucl.ac.uk/~ywteh/research/npbayes/dp.pdf)

- [A Very Gentle Note on the Destruction of Dirichlet Process ](http://users.cecs.anu.edu.au/~xzhang/pubDoc/notes/dirichlet_process.pdf)

狄利克雷过程混合模型(Dirichlet process mixture models)

求解狄利克雷混合模型的变分推断方法
参考[Variational Inference for Dirichlet Process
Mixtures](http://www.cs.columbia.edu/~blei/papers/BleiJordan2004.pdf)

@import "statistic_test.md"

## 分布生成

### Importance Weighted Sampling

@import "sampling.md"

## 高维统计

所谓的高维模型指的是`模型中未知参数的数目要远远大于样本量`。

稀疏性假设
: 一般都假设与响应变量真实相关的预测变量只有一小部分，剩余的大部分预测变量都是噪声变量。

1996 年提出的 Lasso 回归可以看做是高维统计学的开山之作。

### Concentration Inequality

## 相关性分析

pearson 相关系数

spearman 相关系数,亦即秩相关系数
根据随机变量的等级而不是其原始值衡量相关性的一种方法

spearman 相关系数的计算可以由计算 pearson 系数的方法,只需要把原随机变量中的原始数据替换成其在随机变量中的等级顺序即可:

(1,10,100,101)替换成(1,2,3,4)

(21,10,15,13)替换成(4,1,3,2)
