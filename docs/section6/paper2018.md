### 2018年阅读的论文


- Deep Regression Forests for Age Estimation
三校联合文章（之一为南开）

目标： 从面部图片估计年龄，即学习一个图片到年龄的映射。

难点： 同龄人的面部表情差异非常大（想想郭德纲和林志颖）

方法： Deep Regression Forests (DRFs)

年龄估计，有两种：一种是估计年龄的真实值，一种是估计年龄的区间。文章解决的是第一种。

在每一个分裂节点上，定义一个软分割寒暑，以此让整个决策树可微


交替优化策略： first we fix the leaf nodes and optimize the data partitions at split nodes as well as the CNN parameters (feature learning) by Back-propagation; 
Then, we fix the split nodes and optimize the data abstractions at leaf nodes (local regressors) by Variational Bounding

采用 caffe 实现，用了 VGG-16 架构

困惑不解之处： 文章不断提到的 data abstraction at leaf nodes ，什么是 数据抽象？


- Adversarial Examples: Attacks and Defenses for Deep Learning
对抗样本： 深度学习中的攻与防

综述了近期在对抗样本上的研究进展，对对抗样本的生成方法做系统性分类，并调研其应用前景。可以说是对抗样本领域非常全面的一个综述了。


- Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods
对抗样本并不容易被检测出来。我有10种躲过检测的方法。


- Fast Threshold Tests for Detecting Discrimination
from 斯坦福大学
研究人类决策中的偏见：是否对不同群体采取了不同的标准。

分析 2700 万纽约警察在大街上拦截行人并搜查武器的数据，


对照方法：
1. benchmark test： 在考虑了一些变量的不同先验概率的情况下（例如，已经考虑了不同种族的人容易发动袭击的概率），如果比率还有显著的不同，则认为存在偏差。
这种方法的缺陷：遗漏变量带来的偏差，你不知道除了种族，是否还有其他变量在主宰差异，无法找出所有的关键变量。
2. outcome test：研究命中率（hit rate）. 就是从结果来看，警察在不同种族的群体中搜查出武器的概率是否存在差异？
这种方法的缺陷：infra-marginnality, 即，没有考虑到一些先验。例如，即使警察采取中立的标准，最后的结果也可能会显示说警察存在歧视。

threshold test 阈值检验。
基于贝叶斯潜因子模型，

Upon stopping a pedestrian, offi-cers observe the probability p the individual is carrying a weapon; this probability summarizes all the available information, such as the stopped individual’s age and gender, his criminal record, and behavioral indicators like nervousness and evasiveness. Because these probabilities vary from one individual to the next, p is modeled as being drawn from a risk distribution that depends on the stopped person’s race (r) and the location of the stop (d), where location might indicate the precinct in which the stop occurred.
Officers 
deterministically conduct a search if the probability p exceeds a race- and location-specific threshold (trd), and if a search is conducted, a weapon is found with probability p. 


本文的主要贡献在于提出了一种高效的阈值检验的计算方法。


- Understanding Career Progression in Baseball Through Machine Learning

- Query-efficient Black-box Adversarial Examples

- Focal Loss for Dense Object Detection


- Bidirectional LSTM-CRF Models for Sequence Tagging

来自百度2015年的工作。组合双向LSTM和CRF 的想法是创新点。

- How Well Can Generative Adversarial Networks (GAN) Learn Densities: A Nonparametric View
理论派的文章，各种公式哇。

- Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition
中文命名体识别，基于字粒度，使用了偏旁部首特征。
中科院和东芝联合出品，16年的工作。模型其实用的也是标准的 双向LSTM 加 CRF

论文中提到的数据来源为 SIGHAN （国际计算语言学会（ACL）中文语言处理小组的简称），其英文全称为“Special Interest Group for Chinese Language Processing of the Association for Computational Linguistics”，又可以理解为“SIG汉“或“SIG漢“。
而Bakeoff则是SIGHAN所主办的国际中文语言处理竞赛，第一届于2003年在日本札幌举行（Bakeoff 2003),第二届于2005年在韩国济州岛举行(Bakeoff 2005), 而2006年在悉尼举行的第三届（Bakeoff 2006）则在前两届的基础上加入了中文命名实体识别评测。目前SIGHAN Bakeoff已成功举办了6届，其中Bakeoff 2005的数据和结果在其主页上是完全免费和公开的，但是请注意使用的前提是非商业使用（non-commercial）:


- [Prefrontal cortex as a meta-reinforcement learning system](https://deepmind.com/blog/prefrontal-cortex-meta-reinforcement-learning-system/)
















































