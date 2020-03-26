# FM {ignore=true}

[TOC]

## FM 及其延伸方法

### 2010-FM

因子分解机

$$
y = w _ { 0 } + \sum _ { i = 1 } ^ { n } w _ { i } x _ { i } + \sum _ { i = 1 } ^ { n } \sum _ { j = i + 1 } ^ { n } \left\langle v _ { i } , v _ { j } \right\rangle x _ { i } x _ { j }
$$

FM 为每个特征学习了一个权重（这部分其实就是一个线性模型）和一个隐权重向量（latent vector），**在特征交叉时，使用两个特征隐向量的内积作为交叉特征的权重**。
也就是说，FM 其实就是一种 Embedding.

特征交叉的 naive 版本
暴力地给每两个特征组合一个权重，如下，则是$O(kn^2)$ 级别；

$$
y(x)=w_{0}+\sum_{i=1}^{n} w_{i} x_{i}+\sum_{i=1}^{n} \sum_{j=i+1}^{n} w_{i j} x_{i} x_{j}
$$

但如果每个特征是 $k$ 维向量，则是 $O(kn)$

其实推导的话，也可以从 naive 版本的 W 矩阵的分解出发，参考 [因子分解机 FM-高效的组合高阶特征模型](http://bourneli.github.io/ml/fm/2017/07/02/fm-remove-combine-features-by-yourself.html)

$$
\begin{align}
&\sum _ { i = 1 } ^ { n - 1 } \sum _ { j = i + 1 } ^ { n } \left\langle \mathbf { v } _ { i } , \mathbf { v } _ { j } \right\rangle x _ { i } x _ { j } \\\
= & \frac { 1 } { 2 } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \left\langle \mathbf { v } _ { i } , \mathbf { v } _ { j } \right\rangle x _ { i } x _ { j } - \frac { 1 } { 2 } \sum _ { i = 1 } ^ { n } \left\langle \mathbf { v } _ { i } , \mathbf { v } _ { i } \right\rangle x _ { i } x _ { i } \\\
= & \frac { 1 } { 2 } \left( \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \sum _ { f = 1 } ^ { k } v _ { i , f } v _ { j , f } x _ { i } x _ { j } - \sum _ { i = 1 } ^ { n } \sum _ { f = 1 } ^ { k } v _ { i , f } v _ { i , f } x _ { i } x _ { i } \right) \\\
= & \frac { 1 } { 2 } \sum _ { f = 1 } ^ { k } \left( \left( \sum _ { i = 1 } ^ { n } v _ { i , f } x _ { i } \right) \left( \sum _ { j = 1 } ^ { n } v _ { j , f } x _ { j } \right) - \sum _ { i = 1 } ^ { n } v _ { i , f } ^ { 2 } x _ { i } ^ { 2 } \right) \\\
= & \frac { 1 } { 2 } \sum _ { f = 1 } ^ { k } \left( \left( \sum _ { i = 1 } ^ { n } v _ { i , f } x _ { i } \right) ^ { 2 } - \sum _ { i = 1 } ^ { n } v _ { i , f } ^ { 2 } x _ { i } ^ { 2 } \right)
\end{align}
$$

FM 和基于树的模型（e.g. GBDT）都能够自动学习特征交叉组合。基于树的模型适合连续中低度稀疏数据，容易学到高阶组合。但是**树模型却不适合学习高度稀疏数据的特征组合，一方面高度稀疏数据的特征维度一般很高，这时基于树的模型学习效率很低，甚至不可行；另一方面树模型也不能学习到训练数据中很少或没有出现的特征组合**。相反，FM 模型因为通过隐向量的内积来提取特征组合，对于训练数据中很少或没有出现的特征组合也能够学习到,<font color=red>使得 FM 能更好的解决数据稀疏性的问题</font>。例如，特征 𝑖 和特征 𝑗 在训练数据中从来没有成对出现过，但特征 𝑖 经常和特征 𝑝 成对出现，特征 𝑗 也经常和特征 𝑝 成对出现，因而在 FM 模型中特征 𝑖 和特征 𝑗 也会有一定的相关性。毕竟所有包含特征 𝑖 的训练样本都会导致模型更新特征 𝑖 的隐向量 𝑣𝑖，同理，所有包含特征 𝑗 的样本也会导致模型更新隐向量 𝑣𝑗，这样⟨𝑣𝑖,𝑣𝑗⟩就不太可能为 0。

### FFM

Field-aware FM

$$
\phi _ { \mathrm { FFM } } ( \boldsymbol { w } , \boldsymbol { x } ) = \sum _ { j _ { 1 } = 1 } ^ { n } \sum _ { j _ { 2 } = j _ { 1 } + 1 } ^ { n } \left( \boldsymbol { w } _ { j _ { 1 } , f _ { 2 } } \cdot \boldsymbol { w } _ { j _ { 2 } , f _ { 1 } } \right) x _ { j _ { 1 } } x _ { j _ { 2 } }
$$

FFM 模型学习每个特征在 f 个域上的 k 维隐向量，交叉特征的权重由特征在对方特征域上的隐向量内积得到，权重数量共 n*k*f 个。在训练方面，由于 FFM 的二次项并不能够像 FM 那样简化，因此其复杂度为 $kn^2$?

欲解决特征组合问题，但因为计算复杂度的原因，一般只用到二阶特征组合.

#### 实现

LibFFM 效率高很大程度上是因为优化算法使用了 **hogwild!**。
hogwild!使用一块共享内存来保存模型参数，这些参数可以被多个处理器或者线程访问。虽然更新某一个参数的操作是原子操作，但实现中多个处理器访问的时候并不会对模型参数加锁。很容易看出，在更新过程，更新梯度的结果可能用到过期的参数值。所以算法有个梯度稀疏的前提假设，就是说梯度只依赖极少数的参数。这个假设在大规模机器学习问题中一般都成立，大多模型的总体维度非常高，达到千万或者上亿级别，但是真正取值为 1 的特征只有几十个左右。至于效率，我使用自己的一个数据集，1800w 个样本，维度在百万级别，开 40 个线程，完成一次迭代只需要 9 分钟，如果只开一个线程的话则需要三个小时。

### 2017-DeepFM

华为诺亚方舟 Lab 与 哈工大发表在 IJCAI 2017 的论文 `DeepFM: A Factorization-Machine based Neural Network for CTR Prediction`

DeepFM 对 Wide&Deep 的改进之处在于，它用 FM 替换掉了原来的 Wide 部分，加强了浅层网络部分特征组合的能力。事实上，由于 FM 本身就是由一阶部分和二阶部分组成的，DeepFM 相当于同时组合了原 Wide 部分+二阶特征交叉部分+Deep 部分三种结构，无疑进一步增强了模型的表达能力.

DeepFM 在华为 APP 应用市场的推荐业务中取得了不错的效果。

$$
\hat { y } = \operatorname { sigmoid } \left( y _ { F M } + y _ { D N N } \right)
$$

### NFM

Neural factorization machines

### 2018-xDeepFM

今天介绍 中科大、北大 与 微软 合作发表在 KDD18 的文章 `xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems`

DCN 的 Cross 层接在 Embedding 层之后，虽然可以显示自动构造高阶特征，但它是以 **bit-wise** 的方式。例如，Age Field 对应嵌入向量<a1,b1,c1>，Occupation Field 对应嵌入向量<a2,b2,c2>，在 Cross 层，a1,b1,c1,a2,b2,c2 会拼接后直接作为输入，即它意识不到 Field vector 的概念。Cross 以嵌入向量中的单个 bit 为最细粒度，而 FM 是以向量为最细粒度学习相关性，即 vector-wise。xDeepFM 的动机，正是将 FM 的 **vector-wise** 的思想引入 Cross 部分。

如果说 DeepFM 只是“Deep & FM”，那么 xDeepFm 就真正做到了”Deep” Factorization Machine

## 代码实现

categorical 特征只能所有出现过的都取吗？ 这个对一些 id 类的特征是不适应的。

[python wrapper for libfm](https://github.com/alexeygrigorev/libffm-python)
安装报错： fatal error: 'random' file not found

xlearn 包中也有 FM 和 FFM 的实现

tffm
`pip3 install tffm`
能安装，但

## 参考

- [Factorization Machines with tensorflow tutorial](https://github.com/babakx/fm_tensorflow/blob/master/fm_tensorflow.ipynb)
  用 tensorflow 实现的 FM demo

- [tffm](https://github.com/geffy/tffm/tree/a98c786917f5ca74a249748ddef8b694b7f823c9/tffm)

- [FM_FFM](https://github.com/Aifcce/FM-FFM)
