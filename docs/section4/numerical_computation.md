# 数值计算 {ignore=true}

[TOC]

## 数值计算的稳定性

_underflow_
: 当接近零的数字四舍五入为零时，发生下溢出。

_overflow_
: 当数值非常大，超过了计算机的表示范围时，发生上溢出

例如，softmax 函数的溢出

$$
\operatorname { softmax } ( \overrightarrow { \mathbf { x } } ) = \left( \frac { \exp \left( x _ { 1 } \right) } { \sum _ { j = 1 } ^ { n } \exp \left( x _ { j } \right) } , \frac { \exp \left( x _ { 2 } \right) } { \sum _ { j = 1 } ^ { n } \exp \left( x _ { j } \right) } , \cdots , \frac { \exp \left( x _ { n } \right) } { \sum _ { j = 1 } ^ { n } \exp \left( x _ { j } \right) } \right) ^ { T }
$$

当所有的 $x_i$ 都等于常数 $c$ 时，softmax 函数的每个分量的理论值都为 $\frac 1 n $

- 考虑*c*是一个非常大的负数（比如趋近负无穷），此时 $exp(c)$ 下溢出。此时分母为零，结果未定义。
- 考虑 $c$ 是一个非常大的正数（比如趋近正无穷），此时 $exp(c)$ 上溢出。结果未定义。

为了解决 softmax 函数的数值稳定性问题，可证明，向量 $\overrightarrow x$ 减去其最大分量，其 softmax 值不变，从而

- 当 $\overrightarrow x$ 的分量较小时， 新的分量至少有一个为零，从而导致分母至少有一项为 1，从而解决了下溢出的问题。
- 当 $\overrightarrow x$ 的分量较大时， 相当于分子分母同时除以一个非常大的数，从而解决了上溢出。

$\log softmax $ 函数的溢出

softmax 名字的来源是 _hardmax_： 最大元素的位置填充 1，其它位置填充 0，softmax 填充的是 0 到 1 之间的数

### Coditioning

Conditioning 刻画了一个函数的如下特性：当函数的输入发生了微小的变化时，函数的输出的变化有多大。

对于 Conditioning 较大的函数，在数值计算中可能有问题。因为函数输入的舍入误差可能导致函数输出的较大变化。

方阵的条件数就是最大的特征值除以最小的特征值。

$$
\text { condition number } = \max _ { 1 \leq i , j \leq n , i \neq j } \left| \frac { \lambda _ { i } } { \lambda _ { j } } \right|
$$

当方阵的条件数很大时，矩阵的求逆将对误差特别敏感（即： 的一个很小的扰动，将导致其逆矩阵一个非常明显的变化）。
条件数是矩阵本身的特性，它会放大那些包含矩阵求逆运算过程中的误差。

## 优化方法

参见最优化章节

## Fast Fourier Transforms

[v 神的介绍](https://vitalik.ca/general/2019/05/12/fft.html)

### discrete Fourier transform

## 曲线拟合和多项式插值

### 拉格朗日插值多项式

### 牛顿插值多项式
