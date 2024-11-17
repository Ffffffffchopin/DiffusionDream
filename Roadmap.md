# Roadmap
本项目处于早期开发阶段，以下是我们的规划和一些构想：

DiffusionDream的开发分为三个部分分别是训练，推理和交互。

## 训练

我们目前的训练基于[InstructPix2Pix](https://arxiv.org/abs/2211.09800)中的方法，为了在Kaggle平台的T4*2GPU上进行训练，优化器使用Adaw 8bit模式（可能存在精度损失），只训练了基于InstructPix2Pix上Huggingface已有的权重的微调的Unet。数据量50,000张，BatchSize为8，训练了4个Epoch，训练时间约为4小时。未来计划使用更大的数据集，

### 训练数据
