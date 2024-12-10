# ownflow

# 记录
- 2024.11.29: 最小系统实现(layers, activations, optimizers, module), 实现了Linear层的前向推理、反向传播
- 2024.11.30: 实现pytorch中绝大部分激活函数(不包括可学习参数的激活函数、softmax、softmin和MultiheadAttention), 实现pytorch中对应的所有参数初始化方法
- 2024.12.1: 实现Conv2d、Maxpool2D、Avgpool2D、Embeeding层
- 2024.12.2: 给优化器都加上weight_decay选项并实现新优化器Adagrad
- 2024.12.3: 实现Adadelta、Adam、AdamW、Adamax优化器
- 2024.12.6: 实现NAdam优化器
- 2024.12.7: 根据pytorch官网公式修改SGD代码，添加nesterov选择，实现RMSprop优化器
- 2024.12.8: 添加dropout层，并且给model添加了train和test两种状态，可以正确的控制dropout在不同情况下的行为
- 2024.12.10: 重构train和test状态设置方式，变成显示设置，实现了RRelu激活函数和dropout在train和test状态下的不同行为

# 参考
- 框架: [**npnet**](https://github.com/MorvanZhou/npnet)
- pytorch: [**pytorch**](https://pytorch.org/)