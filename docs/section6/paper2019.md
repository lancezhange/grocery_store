# 2019 论文精读 {ignore=true}

<h3 style="color: inherit;line-height: inherit;margin-top: 1.6em;margin-bottom: 1.6em;font-weight: bold;border-bottom: 2px solid rgb(239, 112, 96);font-size: 1.3em;">
<span style="font-size: inherit;line-height: inherit;display: inline-block;font-weight: normal;background: rgb(239, 112, 96);color: rgb(255, 255, 255);padding: 3px 10px 1px;border-top-right-radius: 3px;border-top-left-radius: 3px;margin-right: 3px;">论文
</span>
</h3>

<section style="margin-bottom: -10px;margin-left: -8px;max-width: 100%;width: 18px;height: 18px;border-top: 8px solid rgb(54, 85, 173);border-left: 8px solid rgb(54, 65, 173);box-sizing: border-box !important;overflow-wrap: break-word !important;">
</section>

<section data-bgopacity="50%" style="max-width: 100%;background: rgb(247, 247, 247);box-sizing: border-box !important;overflow-wrap: break-word !important;">

<section style="padding: 1em;max-width: 100%;letter-spacing: 1.5px;line-height: 1.75em;box-sizing: border-box !important;overflow-wrap: break-word !important;">

<p style="color: rgb(63, 63, 63);font-size: 15px;max-width: 100%;min-height: 1em;box-sizing: border-box !important;overflow-wrap: break-word !important;">
<span style="color: rgb(11, 0, 34);font-size: 15px;">Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity
</span>
</p>
<span class="author-span">Hulu-陈拉明 •</span>
<span class="author-span">2018-NIPS</span>
<p style="color: rgb(63, 63, 63);font-size: 15px;max-width: 100%;min-height: 1em;box-sizing: border-box !important;overflow-wrap: break-word !important;">
<span style="color: rgb(11, 35, 234);font-size: 14px;">
</span>
</p>

</section>

</section><section data-width="100%" style="margin-top: -10px;margin-left: 8px;max-width: 100%;justify-content: flex-end;display: flex;box-sizing: border-box !important;overflow-wrap: break-word !important;">

<section style="max-width: 100%;width: 18px;height: 18px;border-bottom: 8px solid rgb(54, 65, 173);border-right: 8px solid rgb(54, 65, 173);box-sizing: border-box !important;overflow-wrap: break-word !important;"><br>
</section></section>

### 背景概念

#### 行列式点过程

DPP (`Determinantal Point Process`) 行列式点过程

> DPP 是一种性能较高的概率模型。DPP 将复杂的概率计算转换成简单的行列式计算，并通过核矩阵的行列式计算每一个子集的概率。DPP 不仅减少了计算量，而且提高了运行效率，在图片分割、文本摘要和商品推荐系统中均具有较成功的应用。

行列式计算简单吗？

DPP 通过最大后验概率估计，找到商品集中相关性和多样性最大的子集，从而作为推荐给用户的商品集。

DPP 可以理解为一种抽样方法:

两个元素作为子集被抽取的概率不仅和单一元素被抽取的概率相关,还和这两个元素的相关性有关。单一元素被选择的概率越大，同时元素之间的相似度越低，则这个集合被选择的概率越高。

#### 多样性的衡量

Exploration 主要有三个方面：

1. 覆盖度：被推荐给用户的内容占全部内容的比例应该较高，特别是新的内容能够有机会展现给用户。

2. 惊喜：推荐的内容并不与用户之前的行为明显相关，但又是用户所喜欢的。这能很大程度提升用户体验，但却难以给出衡量指标。

3. 多样性：在短时间内不要过多地向同一用户推荐同一类型的内容，而是混合各种类型的内容推荐给用户。

如何衡量多样性？

1. Temporal Diversity ( 时间的多样性 )
   在固定的时间间隔内推荐*不同类*的内容的个数
   策略： 跨类别推荐；时间衰减；Impression discount ( 印象折扣 ) ，统计所有推荐给用户的内容中哪些是用户没有观看
2. Spatial Diversity ( 空间的多样性 )
   单个推荐列表中物品之间的差异程度，可以通过计算在同一个推荐 list 中两两 Item 之间的相似度的平均值来进行衡量。

一个用户很喜欢看漫威的电影，也喜欢看一些文艺类的电影，其中用户观看漫威的电影比较多一些，看文艺类的电影少一些，那么推荐系统很容易造成推荐的时候只推荐漫威类的电影。

### 论文主要内容

直接优化求解哪个子集的行列式最大，这是 NP hard 的问题，因此，陈拉明团队则利用贪婪算法，提出了一种能加速行列式点过程推理过程的方法： 即将该问题转化为一种贪婪的形式。

在模型求解时，行列式的复杂度其实是非常高的（一般是行列式大小的三次方），现时不适用于于线上对性能有高要求的场景。论文中对此也做了改进：利用 Cholesky 分解，将行列式的计算转化为下三角行列式这种简单的形式（计算复杂度能降低到二次方）。进一步，用增量的方式更新，将每次迭代的复杂度进一步降低到一次方。

<h3 style="color: inherit;line-height: inherit;margin-top: 1.6em;margin-bottom: 1.6em;font-weight: bold;border-bottom: 2px solid rgb(239, 112, 96);font-size: 1.3em;">
<span style="font-size: inherit;line-height: inherit;display: inline-block;font-weight: normal;background: rgb(239, 112, 96);color: rgb(255, 255, 255);padding: 3px 10px 1px;border-top-right-radius: 3px;border-top-left-radius: 3px;margin-right: 3px;">论文
</span>
</h3>
<section style="margin-bottom: -10px;margin-left: -8px;max-width: 100%;width: 18px;height: 18px;border-top: 8px solid rgb(54, 85, 173);border-left: 8px solid rgb(54, 65, 173);box-sizing: border-box !important;overflow-wrap: break-word !important;">
</section>
<section data-bgopacity="50%" style = "max-width: 100%;background: rgb(247, 247, 247);box-sizing: border-box !important;overflow-wrap: break-word !important;">
<section style="padding: 1em;max-width: 100%;letter-spacing: 1.5px;line-height: 1.75em;box-sizing: border-box !important;overflow-wrap: break-word !important;">
<p style="color: rgb(63, 63, 63);font-size: 15px;max-width: 100%;min-height: 1em;box-sizing: border-box !important;overflow-wrap: break-word !important;">
<span style="color: rgb(11, 0, 34);font-size: 15px;">Focal Loss for Dense Object Detection</span></p>
<span class="author-span">ICCV2017 | paper-author</span>
<p style="color: rgb(63, 63, 63);font-size: 15px;max-width: 100%;min-height: 1em;box-sizing: border-box !important;overflow-wrap: break-word !important;">
<span style="color: rgb(11, 35, 234);font-size: 14px;"></span></p></section></section>
<section data-width="100%" style = "margin-top: -10px;margin-left: 8px;max-width: 100%;justify-content: flex-end;display: flex;box-sizing: border-box !important;overflow-wrap: break-word !important;">
<section style="max-width: 100%;width: 18px;height: 18px;border-bottom: 8px solid rgb(54, 65, 173);border-right: 8px solid rgb(54, 65, 173);box-sizing: border-box !important;overflow-wrap: break-word !important;"><br></section></section>

### 论文内容

Focal Loss 的引入主要是为了解决难易样本数量不平衡的问题。
注意，难易样本数量不平衡**区别于正负样本数量不平衡**

正负样本不平衡的问题，可以在 交叉熵损失的前面加上一个参数 $\alpha$

对难易问题，一个简单的思想：把高置信度(p)样本的损失再降低一些不就好了吗！

$$
F L=\left \\{\begin{array}{ccc}{-(1-p)^{\gamma} \log (p),} & {\text { if }} & {y=1} \\\ {-p^{\gamma} \log (1-p),} & {\text { if }} & {y=0}\end{array}\right.
$$

结合二者，最终的 Focal Loss 形式如下

$$
F L=\left\\{\begin{aligned}-\alpha(1-p)^{\gamma} \log (p), & \text { if } & y=1 \\\-(1-\alpha) p^{\gamma} \log (1-p), & \text { if } & y=0 \end{aligned}\right.
$$

实验表明$\gamma$ 取 2, $\alpha$ 取 0.25 的时候效果最佳。

### 后续延伸

GHM (gradient harmonizing mechanism) 发表于 “Gradient Harmonized Single-stage Detector", AAAI2019，是基于 Focal loss 的改进。
