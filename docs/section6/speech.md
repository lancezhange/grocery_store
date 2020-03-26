# 语音识别专题 {ignore=True}

[TOC]

</br>

</br>
<section style="margin-bottom:-16px">
<section style="margin-top:0px;margin-right:0px;margin-bottom:0px;margin-left:2em;padding-top:2px;padding-right:1em;padding-bottom:2px;padding-left:1em;max-width:100%;display:inline-block;background-image:none;background-color:rgb(196, 212, 218);color:rgb(61, 88, 98);font-size:16px;text-align:center;letter-spacing:1.5px;line-height:1.75em;border-top-left-radius:16px;border-top-right-radius:16px;border-bottom-right-radius:16px;border-bottom-left-radius:16px;box-sizing:border-box;word-wrap:break-word;">
<strong>前言</strong>
</section>
</section>
<section style="margin-top:0px;margin-right:0px;margin-bottom:20px;margin-left:0px;padding-top:2.5em;padding-right:1em;padding-bottom:1em;padding-left:1em;max-width:100%;box-sizing:border-box;border-top-width:1px;border-right-width:1px;border-bottom-width:1px;border-left-width:1px;border-top-style:solid;border-right-style:solid;border-bottom-style:solid;border-left-style:solid;border-top-color:rgb(196, 212, 218);border-right-color:rgb(196, 212, 218);border-bottom-color:rgb(196, 212, 218);border-left-color:rgb(196, 212, 218);border-top-left-radius:10px;border-top-right-radius:10px;border-bottom-right-radius:10px;border-bottom-left-radius:10px;word-wrap:break-word;">语音处理于我而言，目前还是一个神秘的处女地。
</section>
</br>

## 基础知识

语音识别的一个难点在于语音和文字的对齐。

### 特征

#### MFCC

#### CTC(Connection Temporal Classification)

### 语音唤醒 keyword spotting

评价指标： 准召率，实时性，能耗

## 模型

## 工具

### KALDI

目前工业界最流行的语音识别框架,完整地包含隐马尔可夫模型，高斯混合模型，决策树聚类，深度神经网络，解码器，以及加权有限状态转换器等技术。

### Essential

[Essentia](https://github.com/MTG/essentia)

### Gentle

[Gentle](https://lowerquality.com/gentle/) 语音和文字的强制对齐

## Demo

<div class="c-callout c-callout–note">
<strong class="c-callout__title">Note</strong>
<p class="c-callout__paragraph">
   本例取自博文 <a href="https://wangkaisine.github.io/2019/06/25/fst-in-kaldi-and-its-visual/">Kaldi中的FST及其可视化 </a>
</p>
</div>

1. train.txt: 经过分词的中文语料

```
语音 识别 技术
语音 识别 算法 公式
作战 防御 工事
```

2. lexicon.txt：发音词典文件

```
!SIL SIL
<SPOKEN_NOISE> SPN
<SPOKEN_NOISE> sil
<UNK> SPN
语音 vv v3 ii in1
识别 sh ix2 b ie2
技术 j i4 sh u4
算法 s uan4 f a3
公式 g ong1 sh ix4
作战 z uo4 zh an4
防御 f ang2 vv v4
工事 g ong1 sh ix4
```

3.

## 参考资料

- [Speech Recognition with Neural Networks](http://andrew.gibiansky.com/blog/machine-learning/speech-recognition-neural-networks/)

- [语音识别技术的过去、现在和未来](file:///Users/zhangxisheng/Downloads/The-Past-Present-and-Future-of-Speech-Recognition-Technology.pdf)

- [Speech Recognition with Deep RNNs](http://arxiv.org/pdf/1303.5778.pdf)

- [Deep Speech: Scaling up end-to-end speech recognition](http://arxiv.org/abs/1412.5567)

  百度的 _Deep Speech_，无需对噪音和混响人工建模，在 Switchboard Hub5'00 基准数据上的错误率为 16%

- [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin by Dario Amodei, et al., 2015](http://arxiv.org/abs/1512.02595)

  Deep Speech ２代，百度出品。列出的作者有３４人之多！宣称在几个基准测试上比亚马逊的土耳其工人的准确率还要高。只需很少的修改就能用到其他语种，并且可以部署到生产环境（用到所谓的批量分发(batch dispatch)）。主要的三个方面：更好的模型结构（结合　 CTC 损失函数），更好的数据（11940 小时的英语和　 9400 小时的汉语），以及更快的计算（放弃使用参数服务器和异步更新，而采用同步的 SGD，利用高性能计算中的优化策略对 GPU 计算做优化）。

- [Towards End-to-end Speech Recognition with RNNs](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf)

  ICML 2014 论文，结合了双向 LSTM 和 CTC(connectionist temporal classification)目标函数
