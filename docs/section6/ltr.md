# Learning to Rank (LTR) {ignore=true}

[TOC]

排序学习（Learning2Rank） 即将 ML 技术应用到 ranking 问题，训练 ranking 模型

通常的做法是把搜索结果的每一条分为 5 类：

1. bad（差）
2. fair（一般）
3. good（好）
4. excellent（非常好）
5. perfect（完美）

## 基本原理

LTR 的学习方法分为

1. 单文档方法（Pointwise）
2. 文档对方法（Pairwise）
3. 文档列表方法（Listwise）

Pointwise 和 Pairwise 把排序问题转换成 回归 、分类 或有序分类 问题。Lisewise 把 Query 下整个搜索结果作为一个训练的实例。3 种方法的区别主要体现在损失函数（Loss Function）上。

### Pointwise

样本是单个 doc（和对应 query）构成的特征向量， 输出其相关度。

pointwise 类方法可以进一步分成三类：

1. 基于回归的算法
2. 基于分类的算法
3. 基于有序回归的算法

缺点：

1. ranking 追求的是排序结果，并不要求精确打分，只要有相对打分即可。
2. pointwise 类方法并没有考虑同一个 query 对应的 docs 间的内部依赖性。一方面，导致输入空间内的样本不是 IID 的，违反了 ML 的基本假设，另一方面，没有充分利用这种样本间的结构性。其次，当不同 query 对应不同数量的 docs 时，整体 loss 将会被对应 docs 数量大的 query 组所支配，前面说过应该每组 query 都是等价的。
3. 损失函数也没有 model 到预测排序中的位置信息。因此，损失函数可能无意的过多强调那些不重要的 docs，即那些排序在后面对用户体验影响小的 doc。

### Pairwise

输入空间中样本是（同一 query 对应的）两个 doc（和对应 query）构成的两个特征向量；

pairwise 类方法基本就是使用二分类算法即可。经典的算法有 基于 NN 的 SortNet，基于 NN 的 RankNet，基于 fidelity loss 的 FRank，基于 AdaBoost 的 RankBoost，基于 SVM 的 RankingSVM，基于提升树的 GBRank。

缺点：

1. 对不同级别之间的区分度是一致对待的。
   在信息检索领域，尤其对于搜索引擎而言，人们更倾向于只点击搜索引擎返回的前几页结果，甚至只是前几条。所以我们对相关度高（Perfect）的文档应该作更好的区分。
2. 相关文档集大小带来的模型偏置。
   假设 query1 对应的相关文档集大小为 5，query2 的相关文档集大小为 1000，那么从后者构造的训练样本数远远大于前者，从而使得分类器对相关文档集小的 query 所产生的训练实例区分不好，甚至视若无睹。
3. pairwise 类方法相对 pointwise 类方法对噪声标注更敏感，即一个错误标注会引起多个 doc pair 标注错误。
4. pairwise 类方法仅考虑了 doc pair 的相对位置，损失函数还是没有 model 到预测排序中的位置信息
5. pairwise 类方法也没有考虑同一个 query 对应的 doc pair 间的内部依赖性，即输入空间内的样本并不是 IID 的，违反了 ML 的基本假设，并且也没有充分利用这种样本间的结构性。

#### Ranking SVM

#### RankNet

##### 排名概率

复杂度为 $O(n)$

样本 $x_i$ 的排名得分用 $o_i$ 表示，定义 $o_{i,j}=o_i−o_j$ ，如果 $o_{i,j}>0$ 就说明 $x_i$ 名次高于 $x_j$ 。将这个排名概率化，定义：

$$
P _ { i , j } = \frac { e ^ { o _ { i j } } } { 1 + e ^ { o _ { i j } } }
$$

则有：

$$
\begin{align}
P _ { i , j } & =  \frac { e ^ { o _ { i , j } } } { 1 + e ^ { o _ { i , j } } } \\\
 & = \frac { e ^ { o _ { i } - o _ { j } } } { 1 + e ^ { o _ { i } - o _ { j } } } \\\
 & = \frac { e ^ { o _ { i } - o _ { k } + o _ { k } + o _ { j } } } { 1 + e ^ { o _ { i } - o _ { k } + o _ { k } + o _ { j } } } \\\
 & = \frac { e ^ { o _ { i , k } + o _ { k , j } } } { 1 + e ^ { o _ { i , k } + o _ { k , j } } } \\\
 & = \frac { P _ { i , k } P _ { k , j } } { 1 + 2 P _ { i , k } P _ { k , j } - P _ { i , k } - P _ { k , j } }
\end{align}
$$

因此，想要知道任意两个 item 的排列关系，不需计算 $C^2_n$ 种组合，n-1 个 $P_{i,i+1}$ 已经含有了所有组合的排序信息，这个就是 RankNet 将 $O(C^2_n)$ 复杂度的问题，变为 $O(n)$ 问题的理论基础。

这个概率通常还会有一个系数

$$
P _ { i j } \equiv P \left( U _ { i } > U _ { j } \right) \equiv \frac { 1 } { 1 + e ^ { - \sigma  s _ { ij }  } }
$$

文档对(Ui,Uj)的交叉熵损失函数

$$
C _ { i j } = - \overline { P } _ { i j } \log P _ { i j } - \left( 1 - \overline { P } _ { i j } \right) \log \left( 1 - P _ { i j } \right)
$$

真实概率

$$
\overline { P } _ { i j } = \frac { 1 } { 2 } \left( 1 + S _ { i j } \right)
$$

其中

$$
S _ { i j } = \left\\{ \begin{array} { c } { 1 } \text{ if i 比 j 更相关 } \\\ { 0 } \text{ if i 和 j 相关度一致 } \\\ { - 1 } \text{ if j 比 i 更相关 } \end{array} \right.
$$

从而损失函数可以写为

$$
C = \frac { 1 } { 2 } \left( 1 - S _ { i j } \right) \sigma  s _ { ij }  + \log \left( 1 + e ^ { - \sigma  s _ { ij }  } \right)
$$

若 $S_{ij} = 1$，则有

$$
C = \log \left( 1 + e ^ { - \sigma  s _ { ij }  } \right)
$$

若 $S_{ij} = -1$，则因 $S_{ij} = -S_{ji}$ 有

$$
C =  \sigma  s _ { ij }  + \log \left( 1 + e ^ { - \sigma  s _ { ij }  } \right) =  \log \left( 1 + e ^ { - \sigma  s _ { ji }  } \right)
$$

可以看出损失函数 C 具有对称性，也即交换 i 和 j 的位置，损失函数的值不变。

##### 模型结构

RankNet 是一个三层的神经网络

##### 参数训练

$$
w _ { k } \leftarrow w _ { k } - \eta \frac { \partial C } { \partial w _ { k } } = w _ { k } - \eta \left( \frac { \partial C } { \partial s _ { i } } \frac { \partial s _ { i } } { \partial w _ { k } } + \frac { \partial C } { \partial s _ { j } } \frac { \partial s _ { j } } { \partial w _ { k } } \right)
$$

$$
\frac { \partial C } { \partial s _ { i } } = \sigma \left( \frac { 1 } { 2 } \left( 1 - S _ { i j } \right) - \frac { 1 } { 1 + e ^ { \sigma \left( s _ { i } - s \right) } } \right) = - \frac { \partial C } { \partial s _ { j } }
$$

$s_i$ 和 $s_j$ 对 $w_k$ 的偏导数可根据神经网络求偏导数的方式求得

##### 加速训练过程

$$
\frac { \partial C } { \partial w _ { k } } = \frac { \partial C } { \partial s _ { i } } \frac { \partial s _ { i } } { \partial w _ { k } } + \frac { \partial C } { \partial s _ { j } } \frac { \partial s _ { j } } { \partial w _ { k } } = \sigma \left( \frac { 1 } { 2 } \left( 1 - S _ { i j } \right) - \frac { 1 } { 1 + e ^ { \sigma \left( s _ { i } - s _ { j } \right) } } \right) \left( \frac { \partial s _ { i } } { \partial w _ { k } } - \frac { \partial s _ { j } } { \partial w _ { k } } \right) = \lambda _ { i j } \left( \frac { \partial s _ { i } } { \partial w _ { k } } - \frac { \partial s _ { j } } { \partial w _ { k } } \right)
$$

其中

$$
\lambda _ { i j } = \frac { \partial C \left( s _ { i } - s _ { j } \right) } { \partial s _ { i } } = \sigma \left( \frac { 1 } { 2 } \left( 1 - S _ { i j } \right) - \frac { 1 } { 1 + e ^ { \sigma \left( s _ { i } - s _ { j j } \right) } } \right)
$$

#### LambdaRank

基于 RankNet, 在 RankNet 的基础上修改了梯度的计算方式，也即加入了 lambda 梯度

#### LambdaMART

提升树版本的 LambdaRank, LambdaMART 结合了 lambda 梯度和 MART

MART(Multiple Additive Regression Tree)又称为 GBDT(Gradient Boosting Decision Tree)

#### gbrank

### Online Learning to Rank (OL2R)

## 效果评价

评价方法包括：

1. MAP

某系统对于主题 1 检索出 4 个相关网页，其 rank 分别为 1, 2, 4, 7

对于 query 1，平均准确率为(1/1+2/2+3/4+4/7)/4=0.83
多个 query 取平均即可。

2. NDCG

$$
D C G _ { k } = \sum _ { m = 1 } ^ { k } \frac { 2 ^ { R ( j , m ) } - 1 } { \log ( 1 + m ) }
$$

$IDCG_{k}$ 是完美排序下的 DCG
则 NDCG 为

$$
NDCG_k = \frac {D C G _ { k }} {IDCG_{k}}
$$

无论是 MAP 还是 NDCG，都具有的特点：

1. 基于 query ，即不管一个 query 对应的 docs 排序有多糟糕，也不会严重影响整体的评价过程，因为每组 query-docs 对平均指标都是相同的贡献。
2. 基于 position ，即显式的利用了排序列表中的位置信息，这个特性的副作用就是上述指标是离散不可微的。

一方面，这些指标离散不可微，从而没法应用到某些学习算法模型上；另一方面，这些评估指标较为权威，通常用来评估基于各类方式训练出来的 ranking 模型。因此，即使某些模型提出新颖的损失函数构造方式，也要受这些指标启发，符合上述两个特性才可以。

## 参考

- [机器学习排序算法：RankNet to LambdaRank to LambdaMART](https://www.cnblogs.com/genyuan/p/9788294.html)
