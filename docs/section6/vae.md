## VAE

变分自编码器 和 普通的自编码器 的联系和区别

### AE

变分推断(variational Inference)

- [Variational Inference for Machine Learning](http://shakirm.com/papers/VITutorial.pdf)

  出自 Shakir Mohamed from Google DeepMind.

变分方法
Variational bounds

### VAE

Denoising Autoencoder(DAE)是在 AE 的基础之上，对输入的训练数据加入噪声。所以 DAE 必须学习去除这些噪声而获得真正的没有被噪声污染过的输入数据。因此，这就迫使编码器去学习输入数据的更加鲁棒的表达，通常 DAE 的泛化能力比一般的 AE 强

### SDAE

Stacked Denoising Autoencoder(SDAE)是一个多层的 AE 组成的神经网络，其前一层自编码器的输出作为其后一层自编码器的输入。

### VQ_VAE

生成出的图像，号称比 BigGAN 更加高清逼真，而且更具有多样性！

## 神经变分推断
