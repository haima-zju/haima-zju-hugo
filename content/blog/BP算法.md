+++
author = "shenchun"
categories = ["NN", "deep learning"]
date = "2017-02-08T15:03:24+08:00"
description = "神经网络中反向传播算法公式推导"
featured = ""
featuredalt = ""
featuredpath = ""
linktitle = ""
title = "BP算法"
type = "post"

+++
##说明
在这里首先定义一个三层的神经网络，为了方便起见，输入层、隐藏层、输出层的神经元节点个数均为两个，网络没有偏置，激活函数为sigmod函数，符号定义如下：

| 符号 | 含义 |
|:----|:-----|
| $W_{ab}$ |节点a到节点b的权重 |
| $Y_{a}$ | 节点a的输入 |
| $Z_{a}$ | 节点a的输出 |
| $\delta a$ | 节点a的误差（反向传播） |
| C | 损失函数 |
| $f(x) = \frac {1} {1+e^{-x}}$ | 激活函数 |
| $W_x$ | 第x层权重矩阵（从0开始）|
