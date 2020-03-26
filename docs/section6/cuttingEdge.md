# 前沿

- [On Learning to Think: Algorithmic Information Theory for Novel Combinations of Reinforcement Learning Controllers and Recurrent Neural World Models by Juergen Schmidhuber, 2015](http://arxiv.org/abs/1511.09249)

  learning to think.

* [Human-level concept learning through probabilistic program induction](http://www.sciencemag.org/content/350/6266/1332.full.pdf)

  发表在《科学》上的文章，通过概率推理来学习概念，在这到处都是深度学习的万花丛中，真是一抹别样红啊。[matlab 代码在此](https://github.com/brendenlake/BPL)

  > People learning new concepts can often generalize successfully from just a single example, yet machine learning algorithms typically require tens or hundreds of examples to perform with similar accuracy. People can also use learned concepts in richer ways than conventional algorithms—for action, imagination, and explanation.

  作者指出了两点人类学习和机器学习新概念的区别：人类学习新的概念是小样本学习(one-shot learning)，而现在的机器学习是大样本的；人类对学习到的概念能够更为灵活和广泛地使用。

  文中提出的新框架：BPL(Bayesian Program Learning, 贝叶斯程式学习)，宣称结合了三个关键想法：组合、因果、学习如何学习。核心在于，用概率程式表示概念，而概率程式由低级概念的程式结合空间关系等组合出来。生成模型的生成模型。

- [Illustration2Vec: A Semantic Vector Representation of Illustrations](http://illustration2vec.net/papers/illustration2vec-main.pdf)

  漫画的语义向量表示。日本东北大学（Tohoku University，简称东北大）和东京大学的研究人员联合出品。

  难点：漫画相比于普通的图片，其形态更加丰富；缺少相关数据集

  还提出了一种语义形变(semantic morphing)算法用以在大量漫画数据中中搜寻相关漫画。

* [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://github.com/ZhengyaoJiang/PGPortfolio) (利用深度强化学习框架解决金融投资组合管理问题)

* [mug life](https://www.muglife.com/)
  Mug Life is a new innovative mobile app which leverages deep learning to let you instantly create 3D animations from any uploaded photo. 图片也能动起来了，效果惊人

toread
Quora 重复问题检测 https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur

- [Producing flexible behaviours in simulated environments](https://deepmind.com/blog/producing-flexible-behaviours-simulated-environments/)

人偶模仿，真的太厉害了。
![](https://storage.googleapis.com/deepmind-live-cms/documents/ezgif.com-resize.gif)

- [实时多人姿态估计](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)
  太炫酷了。

问题难点 1.人数是不确定的，并且他们所处 location 及 scale 也是不确定的。 2.人们之间的互动产生的遮挡、四肢变化、肢体重叠等有着复杂的空间关系 3.实时性，很多算法随着人数的增多而增加

关键点
Part Affinity Fields
没有采用 top-down 的方式（先检测到人，再估计人的姿态），而是部位检测并关联，属于 bottom up，并且，对部位的检测和关联是联合学习的

点的关联通过计算一个 部位仿射分

- normalizing flow
  标准化流
  Normalizing flows transform simple densities (like Gaussians) into rich complex distributions that can be used for generative models, RL, and variational inference

local reparameterization 局部再参数化

NF 乃是对一个原始分布的一系列可逆变换

[Normalizing Flow in pymc3](https://github.com/pymc-devs/pymc3/blob/master/docs/source/notebooks/normalizing_flows_overview.ipynb)

- Dynamic Routing Between Capsules
  胶囊网络
  胶囊之间的动态路由

所谓胶囊，就是向量（区别于神经元为标量）

- active learning 主动学习

参考

1. [Active Learning: 一个降低深度学习时间，空间，经济成本的解决方案](https://www.jianshu.com/p/42801f031cfa)

Fine-tuning Convolutional Neural Networks for Biomedical Image Analysis: Actively and Incrementally”。
问题：如何使用尽可能少的标签数据来训练一个效果 promising 的分类器
第一种情况：标签数据太少，标注数据的成本太高
第二种情况：数据太多，无法一次性处理

分类器的性能随着样本量的增多，先提高，到一个阈值之后几乎稳定。那么问题是，如果有效地降低这个阈值？
解决办法：主动学习。去主动学习那些信息量大的样本。

哪些是这样的样本呢？很容易想到的选择标准是，1.对熵大的（例如二分类问题中，预测概率在 0.5 附近的），2.多样性（对于来自同一幅 image 的增广 patch 集，如果它们的分类结果高度不统一了，那么这个 image 就是 Important 的，或者 hard sample）

The way to create something beautiful is often to make subtle tweaks to something that already exists, or to combine existing ideas in a slightly new way.
-- "Hackers & Painters"

[Active Learning Playground](https://github.com/google/active-learning)

- 弱监督学习
  weak Supervision
  当你的标注数据噪音很大，质量很差，

参考
[Weak Supervision: The New Programming Paradigm for Machine Learning](https://hazyresearch.github.io/snorkel/blog/ws_blog_post.html)

- AVB(Adversarial Variational Bayes)

对抗式变分贝叶斯
统一自编码器和 GANs

变分自编码对编码器添加了约束，就是强迫它产生服从单位高斯分布的潜在变量。正是这种约束，把 VAE 和标准自编码器给区分开来了。
相当于我们又了两个目标，一是重建误差要小，二是压缩表示变量和单位高斯分布之间的差异要小。
两个目标之间需要权衡。

我们可以让网络自己去决定这种权衡。对于我们的损失函数，我们可以把这两方面进行加和。一方面，是图片的重构误差，我们可以用平均平方误差来度量，另一方面。我们可以用 KL 散度（KL 散度介绍）来度量我们潜在变量的分布和单位高斯分布的差异。

为了优化 KL 散度，我们需要应用一个简单的参数重构技巧：不像标准自编码器那样产生实数值向量，VAE 的编码器会产生两个向量:一个是均值向量，一个是标准差向量。

参考
[Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks](https://arxiv.org/pdf/1701.04722.pdf)

- 量子计算

[MIT 评论： Serious quantum computers are finally here. What are we going to do with them?](https://www.technologyreview.com/s/610250/hello-quantum-world/)

### 其他待看

[TensorMol](https://github.com/jparkhill/TensorMol)

[ModelDepot](https://modeldepot.io/)
一系列模型

[deep review](https://github.com/greenelab/deep-review)
论文协作

[Neural IMage Assessment](https://github.com/kentsyx/Neural-IMage-Assessment)

[Texygen](https://github.com/geek-ai/Texygen/)
文本生成基线平台

[MAgent](https://github.com/geek-ai/MAgent)
多玩家强化学习平台

[predict next purchase](https://github.com/Featuretools/predict_next_purchase)

precision medicine

[FLAME Clustering](https://en.wikipedia.org/wiki/FLAME_clustering)
Fuzzy clustering by Local Approximation of MEmberships

模糊聚类: 一个元素可以属于多个类

局部的成员估计法
做法：

1. 先抽取 KNN 关系，然后根据 KNN 关系，为每一个元素赋予一个密度值。根据这个密度值，将元素分成三类：
   1. 密度值大于其所有近邻的，称为类的支持点
   2. 异常值：密度值小于所有近邻切低于预设的阈值的
   3. 其他
2. 每个支持点独为一类；所有异常值为一类；
