# 异常检测

## 深度学习方法

### Autoencoders

思路：利用自编码器，学习正样本的稠密表示；当新样本的重建误差（the difference between the input data and the reconstructed outpu）较大的时候，就认为是异常（此误差阈值可以实验确定）。

参考 [这个例子](https://victordibia.github.io/anomagram/#/)

[paper: Anomaly Detection with Density Estimation](https://arxiv.org/abs/2001.04990)

## 资料

1. [A collection of anomaly detection methods](https://github.com/shubhomoydas/ad_examples)
