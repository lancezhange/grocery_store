# 强化学习(reinforcement learning) {ignore=true}

[TOC]
又称为增强学习。

学习一种策略，使得对于每一个状态，决策 AI 的动作

和监督学习的主要区别是，强化学习的数据往往需要通过尝试、和环境进行交互获得。算法根据环境给予的反馈来调整策略

强化学习任务通常使用马尔可夫决策过程描述，AI 处在一个环境中，每个状态为 AI 对环境的感知。当 AI 执行一个动作后，会使得环境按概率转移到另一个状态；同时，环境会根据奖励函数给 AI 一个反馈。综合而言，强化学习主要包含四个要素：状态、动作、转移概率以及奖励函数。

##  基本方法演进
从 Q-Learning，PG 到 DDPG

### Q-Learning

1989 年 Watkins 提出了 Q-Learning 方法，1992 年和 Dayan 证明了其收敛性。

对于一个状态，执行了某个动作，由于很多动作的反馈并不是即时的，比如下棋是有输赢的，通常希望未来的期望奖励最高。

有时还会引入折扣因子 $\lambda$ , 如果在某个时刻获得了奖励 R，对 t 个单位时间前的 动作的期望奖励 贡献是 $R * \lambda ^ t$ ，这个概念在经济学中也有广泛应用。

我们试图得到一个 Q 函数，使得对于一个状态和动作，能够计算期望奖励。

如果状态空间和动作空间都是有限离散的，转移概率是可估计的，那么就很容易用期望动态规划来求解 Q 函数。

策略就是，每次选择 Q 最大的动作，一般要通过探索来迭代估计转移概率，这种方法称为 Q-Learning，值函数方法。

### Deep Q-learning

2013 年，DeepMind 在 NIPS workshop 上提出了 Deep Q-learning，主要工作是能让 AI 从像素输入学会完 Atari 游戏，后来进行了一些改进后上了 2015 年的 Nature 封面。

如果状态空间是连续的，动态规划的状态数就是无限的，所以我们用深度学习网络去拟合这个 Q 函数，这就是 DQN。

通常 DQN 的实现中，会把收集的（状态、动作、执行动作后的状态和奖励）存在内存中，训练的时候多次使用，称为 memory replay。

注意到每个（状态，动作）的 Q 值要拟合 （当前得到的奖励加上（执行动作后的新状态，新动作）的 Q 值），一个函数拟合自己可能引入额外的噪声，所以通常使用一个延迟更新的函数 Q' 来求新的 Q 值，称为 target network。

### DQN 的改进

2015 年，DQN 有三个主要改进，包括 Double DQN，Dueling Network 和 Prioritized Replay。

Double DQN 是在引入了 target network 后，改进了 Q 值的计算方法。

考虑到 Q 值和状态，动作都相关，但我们实际上更注重动作带来的奖励，Dueling Network 对网络结构做了改进。

Prioritized Replay 探讨在 replay memory 采样的优先级问题。

前两者都比较简单和有效，通常只要改三四行代码，但第三者需要维护数据结构，带来的额外时间消耗有时特别大。

### Policy Gradient （PG）

2000 年 Richard S. Sutton 在 NIPS 上提出了 Policy Gradient 方法，PG 看起来是一种更直接的做法，直接以状态作为输入，输出一个动作，根据获得的奖励来梯度下降更新一个动作出现的概率，但这种方法并不能证明收敛至最优策略。

### Deep Deterministic Policy Gradient （DDPG）

第一个 D 是神经网络，而 Deterministic Policy Gradient（DPG）确定性行为策略是 David Silver 等在 2014 年提出的，当概率策略的方差趋近于 0 的时候，就是确定性策略，其运用了演员-评论家框架，把 DQN 和 PG 混合了起来，显著提高样本利用率。

如果动作空间也是连续的，那么就无法直接取到最大的 Q 值，那么我们再用个深度学习网络，称为演员，演员的任务就是选 Q 值大的动作（确定性策略），演员的梯度来自值函数的估计网络（评论家），这个做法的最大优势是，能够离线地更新策略，即像 DQN 一样， 从 replay 采样出数据来训练。

[莫烦的强化学习课程](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/)

1. 确定型的 policy
   A = pi(S)
2. 在 S 下取 A 为一概率值

DFP(Direct Future Prediction)
[Direct Future Prediction - Supervised Learning for Reinforcement Learning](https://flyyufelix.github.io/2017/11/17/direct-future-prediction.html)
直接预测未来
强化学习和监督学习的结合。

### 组合在线学习(combinatorial online learning)

[组合在线学习：实时反馈玩转组合优化](https://mp.weixin.qq.com/s?__biz=MzAwMTA3MzM4Nw==&mid=2649441835&idx=1&sn=abf10e00dd2354a0f256620b9e1fcda9&chksm=82c0afafb5b726b9a4cdb4d9112deba1bfe72803b20fd5f10bd7dd00b798214fbce750d4503f#rd)

###

- [Policy Gradient Methods](http://www.scholarpedia.org/article/Policy_gradient_methods)

### 示教学习（Learning from Demonstration，LfD）

不同于从经验（experice）中学习，而是通过示教者（teacher）给出的示例（example）进行学习。

IRL(Inverse Reinforcement Learning) 逆向强化学习

### 深度强化学习

2013 年，在 DeepMind 发表的著名论文 `Playing Atari with Deep Reinforcement Learning`中，他们介绍了一种新算法，深度 Q 网络（DQN）。文章展示了 AI agent 如何在没有任何先验信息的情况下通过观察屏幕学习玩游戏。结果令人印象深刻。这篇文章开启了被我们成为“深度强化学习”的新时代。

在 Q 学习算法中，有一种函数被称为 Q 函数，它用来估计基于一个状态的回报。同样地，在 DQN 中，使用一个神经网络估计基于状态的回报函数。

[Deep Reinforcement Learning through policy optimization](http://people.eecs.berkeley.edu/~pabbeel/nips-tutorial-policy-optimization-Schulman-Abbeel.pdf)

[The Nuts and Bolts of Deep RL Reseach](http://rll.berkeley.edu/deeprlcourse/docs/nuts-and-bolts.pdf)
深度强化学习研究的基本要点

Ray RLLib: A Composable and Scalable Reinforcement Learning Library

可组合的强化学习并行训练，而不是将并行逻辑贯穿在整个程序中、内聚在所有模块中，从而获得更好的扩展性、组合性和重用性，并且不损失性能。
Ray RLlib 是 Ray 的一部分。
Ray 是一个基于 Python 的分布式执行框架，除了 Ray RLlib， 包括一个超参数优化框架 Ray tune. 17 年 11 月底，Ray 发布了 0.3 版本，因此，是一个相对较新的框架。网文中说 Ray 有望取代 Spark 。

Ray 是如何异步执行以实现并行的呢？如何用对象列表去表示远程对象

### MARDPG

### AmoebaNet(GPipe)

### 层次式强化学习（Hierarchical RL）

### 基于模型的强化学习（model-based RL）



## 分布式强化学习

Alpha 系列背后利器：分布式强化学习。

时间差分学习(temporal difference learning，_TD_)算法，可以说是强化学习的中心。

## 参考

- [深度强化学习介绍](https://zhuanlan.zhihu.com/p/36827710)
-
