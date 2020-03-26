## Deep Learning Tricks

1. 3e-4 号称是 Adam 最好的初始学习率

2. topk-loss(OHEM)

3. weighted loss
   ```python
   weights = [1.2, 1.2, 0.8]
   class_weights = torch.FloatTensor(weights).to(device)
   criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
   ```
4. knowledge distillation（知识蒸馏）
   此方法有助于提高小模型（student）的性能，将大模型（teacher）的预测作为 soft label（用于学习 teacher 的模型信息）与 truth（hard label）扔进去给小模型一起学习
5. Warmup Learning
6. Label Smoothing 标签平滑

7.
