# 广告系统技术 {ignore=true}

[TOC]

广告的计价方式

1. 按照展示计费
   CPM（cost per mail/ cost per thousand impressions） 千人成本,这种计量方式比较粗犷

CPTM (cost per targeted thousand impressions) 有效千人成本
排除无效的人群

2.按照点击计费
cpc(cost per click)

模型的评估
lift5

#### DSP

追踪用户行为
受众选择: low-level model 做初筛，high-level model 做细选

- [Ad Click Prediction: a View from the Trenches](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf) by H. Brendan McMahan, et al., google, 2013

  来自谷歌广告一线战壕的干货。FTRL-Proximal 在线学习算法

[实时竞价方面的研究文章汇总](https://github.com/wnzhang/rtb-papers)

### 流量预估

### 预算分配和投放控制

#### 2014-LinkedIn

`Budget Pacing for Targeted Online Advertisements at LinkedIn`

在二阶竞价下，有竞争力的广告消耗过快，会影响平台的收入。

算法的主要思想就是令每个 campaign（推广计划）的消耗趋势与其曝光变化趋势基本保持一致，以天为时间单位，campaign 为预算控制单位，首先为每个 campaign 预测出其在当天的曝光情况；然后基于其曝光情况，在当前时间片，假如 已消耗/当天预算 的比例大于 已曝光/预测的总曝光 的比例，则说明预算已经消耗过快，需要减小消耗的速度，反之则要加快消耗的速度。
