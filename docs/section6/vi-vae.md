# Variational Inference 变分推断方法

[TOC]

在机器学习尤其是统计机器学习中，变分推断（VI）是逼近概率分布的重要手段，尤其用于贝叶斯建模中的后验概率分布逼近（另外一种逼近方法是采用随机的方法，例如 MCMC）

> 一言以蔽之，如何找相似分布

## 背景

变分推断 针对的是有潜变量 $Z$ 的问题， 并且是在无法精确推断 [^1]的情况下。

[^1]: 精确推断无法成功实现的情况是很多的,例如，存在无解析解情况，计算量过大的情况

高维数据的采样为何如此困难？

## 基本框架

为了计算 后验分布 $p ( z | x )$, 根据贝叶斯公式，需要计算 $p ( x ) = \int p ( x , z ) \mathrm { d } z$, 但是这个积分往往是不可计算( intractable)的， 那么，一个自然的想法是：引入一组参数化的分布 $\mathcal { D } = \left\\{ q _ { \theta } ( z ) \right\\}$（称为 variational distributions，其中 $\theta$ 称为 variational parameters），通过在 $\mathcal{D}$ 里面寻找与 $p(z|x)$ 最“相似”的 distribution 来估计真实的 posterior

$$
P ( z | x ) = \frac { P ( x , z ) } { P ( x ) } \\\
P ( x ) = \frac { P ( x , z ) } { P ( z | x ) } \\\
$$

$$
\begin{align}
\ln P ( x ) & = \ln P ( x , z ) - \ln P ( z | x ) \\\
& = \ln P ( x , z ) - \ln Q ( z ) - \ln \frac { P ( z | x ) } { Q ( z ) }
\end{align}
$$

$$
\begin{align}
\int _ { z } \ln P ( x ) Q ( z ) d z & = \ln P ( x ) \\\
& = \underbrace { \int _ { z } \ln P ( x , z ) Q ( z ) d z - \int _ { z } \ln Q ( z ) Q ( z ) d z} _ { ELBO } - \underbrace{ \int _ { z } \frac { P ( z | x ) } { Q ( z ) } Q ( z ) d z } _ {K L ( Q \| P )}
\end{align}
$$

这就可以写成下面的优化问题：

$$
\theta ^ { * } = \arg \min _ { \theta } \mathrm { KL } \left( q _ { \theta } ( z ) \| p ( z | x ) \right) \tag {1}
$$

\eqref{1} 式 也不好求，因为里面含有我们的目标函数 $p ( z | x )$，但可以证明，minimize \eqref{1} 等价于 maximize 所谓的 ELBO（evidence lower bound），即

$$
\theta ^ { * } = \underset { \theta } { \arg \max } \mathbb { E } _ { q } \left[ \log p ( x , z ) - \log q _ { \theta } ( z ) \right] \tag{2}
$$

以上就是整个 variational inference (VI)的框架了。

## 平均场理论

$$
q ( Z ) = \prod _ { i = 1 } ^ { M } q _ { i } \left( Z _ { i } \right)
$$

## 参考

- [知乎： 请解释下变分推断](https://www.zhihu.com/question/31032863)
