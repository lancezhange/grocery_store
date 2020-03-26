# 线性模型 {ignore=True}

[TOC]

## 线性回归

$$
h _ { \Theta } ( X ) = \Theta ^ { T } X + b
$$

损失函数为

$$
\ell ( \boldsymbol { \theta } ) = \frac { 1 } { 2 n } ( \hat { \boldsymbol { y } } - \boldsymbol { y } ) ^ { \top } ( \hat { \boldsymbol { y } } - \boldsymbol { y } )
$$

## 广义线性模型

Generalized Linear Model, GLM

**三大假设**

1. $Y$ 的分布为指数分布族
2. 广义线性模型的目标是求解 $T ( y ) | x$
3. $\eta = \theta ^ { T } x$


## 逻辑回归

假设： <font color=red >比率的对数是特征的线性组合</font>
由此推导即可

逻辑回归其实是 GLM 的一种， 因此也可以从广义线性模型的假设，从逻辑回归的 $Y$ 是伯努利分布出发推导
$$
\begin{aligned} p ( y ; \phi ) & = \phi ^ { y } ( 1 - \phi ) ^ { 1 - y } \\\ & = \exp \left( \log \phi ^ { y } ( 1 - \phi ) ^ { 1 - y } \right) \\\ & = \exp ( \log \phi + ( 1 - y ) \log ( 1 - \phi ) ) \\\ & = \exp \left( \log \left( \frac { \phi } { 1 - \phi } \right) y + \log ( 1 - \phi ) \right) \end{aligned}
$$



## 广义线性混合模型

Generalized Linear Mixed Model, GLMM

假设误差值并非随机，而是呈一定分布

混合指的是效应的混合。

### Mixture of Gaussian


