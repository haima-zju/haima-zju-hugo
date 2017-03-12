+++
author = "shenchun"
categories = ["NN", "deep learning"]
date = "2017-02-08T15:03:24+08:00"
description = "神经网络中反向传播算法公式推导"
featured = ""
featuredalt = ""
featuredpath = ""
linktitle = ""
title = "通过简单实例说明神经网络中的反向传播算法"
type = "post"

+++
## 说明
在这里首先定义一个三层的神经网络，为了方便起见，输入层、隐藏层、输出层的神经元节点个数均为两个，网络没有偏置，激活函数为sigmod函数，符号定义如下：

| 符号 | 含义 |
|:----|:-----|
| $\hat x_{i}$ | 样本输入第i个节点（从1开始）|
| $\hat y_{i}$ | 样本输出第i个节点（从1开始）|
| $z_{ij}$ | 第i层第j个节点的输入（定义输入层为第0层） |
| $w_{ab}$ |节点a到节点b的权重 |
| $W_x$ | 第x层权重矩阵（从0开始）|
| $y_{ij}$ | 第i层第j个节点的输出 |
| $\delta a$ | 节点a的误差（反向传播） |
| C | 损失函数 |
| $f(x) = \frac {1} {1+e^{-x}}$ | sigmoid激活函数 |

网络结构图如下：
## 推导
根据符号的定义可以得到如下等式：

- `$$C = \frac{1}{2}((y_{21} - \hat y_{1} )^2 - (y_{22} - \hat y_{2})^2) $$`
- `$$ \begin{bmatrix}
        y_{21}  \\
        y_{22} 
	    \end{bmatrix} =  \begin{bmatrix}
							       f( z_{21} ) \\
							        f( z_{22} )
								    \end{bmatrix}$$`
- `$$ \begin{bmatrix}
        z_{21}  \\
        z_{22} 
		    \end{bmatrix} = \begin{bmatrix}
							        w_{35} & w_{45} \\
							        w_{36} & w_{46}
									    \end{bmatrix}  \times \begin{bmatrix}
															        y_{11}  \\
															        y_{12} 
																    \end{bmatrix}$$`
- `$$ \begin{bmatrix}
        y_{11}  \\
        y_{12} 
	    \end{bmatrix} =  \begin{bmatrix}
							       f( z_{11} ) \\
							        f( z_{12} )
								    \end{bmatrix}$$`
- `$$ \begin{bmatrix}
        z_{11}  \\
        z_{12} 
		    \end{bmatrix} = \begin{bmatrix}
							        w_{13} & w_{23} \\
							        w_{14} & w_{24}
									    \end{bmatrix}  \times \begin{bmatrix}
															        \hat x_{1}  \\
															        \hat x_{2} 
																    \end{bmatrix}$$`

通过以上公式对`$W_{1}$`中对权重值求导可得：
`$$ \frac{\partial C}{\partial w_{35}} = \frac{\partial C}{\partial y_{21}} * \frac{\partial y_{21}}{\partial z_{21}} * \frac{\partial z_{21}}{\partial w_{35}} = (y_{21}-\hat y_{1}) * f^{'}(z_{21})*y_{11} $$`
其中，`$f^{'}(z_{21}) = f(z_{21})*(1- f(z_{21}))$`

同理：
`$$ \frac{\partial C}{\partial w_{45}} 
= \frac{\partial C}{\partial y_{21}} * \frac{\partial y_{21}}{\partial z_{21}} * \frac{\partial z_{21}}{\partial w_{45}} 
= (y_{21}-\hat y_{1}) *f^{'}(z_{21}) *y_{12}$$`
`$$ \frac{\partial C}{\partial w_{36}} 
= \frac{\partial C}{\partial y_{22}} * \frac{\partial y_{22}}{\partial z_{22}} * \frac{\partial z_{22}}{\partial w_{36}} 
= (y_{22}-\hat y_{2}) * f^{'}(z_{22}) *y_{11}$$`
`$$ \frac{\partial C}{\partial w_{46}} 
= \frac{\partial C}{\partial y_{22}} * \frac{\partial y_{22}}{\partial z_{22}} * \frac{\partial z_{22}}{\partial w_{46}} 
= (y_{22}-\hat y_{2}) * f^{'}(z_{22}) *y_{12}$$`
其中，`$f^{'}(z_{22}) = f(z_{22})*(1- f(z_{22}))$`


将等式两两合并得到：
`$$ \begin{bmatrix}
	\frac{\partial C}{\partial w_{35}} \\
	\frac{\partial C}{\partial w_{45}}  
	\end{bmatrix} 
= (y_{21}-\hat y_{1}) * f^{'}(z_{21}) \times y_{1}$$`
`$$ \begin{bmatrix}
	\frac{\partial C}{\partial w_{36}} \\
	\frac{\partial C}{\partial w_{46}}  
	\end{bmatrix} 
	= (y_{22}-\hat y_{2}) * f^{'}(z_{22}) \times y_{1}$$`

因此：
`$$\frac{\partial C}{\partial W_{1}} 
= \begin{bmatrix}
	\frac{\partial C}{\partial w_{35}} & \frac{\partial C}{\partial w_{36}}\\
	\frac{\partial C}{\partial w_{45}} & \frac{\partial C}{\partial w_{46}}
	\end{bmatrix} 
= \begin{bmatrix}
	(y_{21}-\hat y_{1}) * f^{'}(z_{21}) \times y^{T}_{1}\\
	(y_{22}-\hat y_{2}) * f^{'}(z_{22}) \times y^{T}_{1}
   \end{bmatrix} 
= \begin{bmatrix}
	(y_{21}-\hat y_{1}) * f^{'}(z_{21}) \\
    (y_{22}-\hat y_{2}) * f^{'}(z_{22})
    \end{bmatrix}\times y^{T}_{1} 
= (y_{2}-\hat y) * f^{'}(z_{2})\times y^{T}_{1}$$`
其中，`$(y_{2}-\hat y) * f^{'}(z_{2})$`是`$2\times 1$`矩阵，`$y^{T}_{1}$`是`$1\times 2$`矩阵，`$*$`表示元素乘，`$\times$`表示矩阵乘

于是得到更新`$W_{1}$`的等式：`$W_{1} += -\eta \frac{\partial C}{\partial W_{1}}$`，`$\eta$`为学习速率

上面我们得到了`$W_{1}$`的更新等式，那下面就开始`$W_{0}$`的吧（嘿嘿……^_^）

对`$w_{13}$`的求导可以得到：
`$$ \frac{\partial C}{\partial w_{13}} 
= (\frac{\partial C}{\partial y_{21}} * \frac{\partial y_{21}}{\partial z_{21}} * \frac{\partial z_{21}}{\partial y_{11}}+\frac{\partial C}{\partial y_{22}} * \frac{\partial y_{22}}{\partial z_{22}} * \frac{\partial z_{22}}{\partial y_{11}}) *\frac{\partial y_{11}}{\partial z_{11}}*\frac{\partial z_{11}}{\partial w_{13}}
= ((y_{21}-\hat y_{1}) * f^{'}(z_{21})*w_{35}+(y_{22}-\hat y_{2})*f^{'}(z_{22})*w_{36})*f^{'}(z_{11})*\hat x_{1}$$`

同理：
`$$ \frac{\partial C}{\partial w_{14}} 
= (\frac{\partial C}{\partial y_{21}} * \frac{\partial y_{21}}{\partial z_{21}} * \frac{\partial z_{21}}{\partial y_{11}}+\frac{\partial C}{\partial y_{22}} * \frac{\partial y_{22}}{\partial z_{22}} * \frac{\partial z_{22}}{\partial y_{11}}) *\frac{\partial y_{11}}{\partial z_{11}}*\frac{\partial z_{11}}{\partial w_{14}}
= ((y_{21}-\hat y_{1}) * f^{'}(z_{21})*w_{35}+(y_{22}-\hat y_{2})*f^{'}(z_{22})*w_{36})*f^{'}(z_{11})*\hat x_{2}$$`

`$$ \frac{\partial C}{\partial w_{23}} 
= (\frac{\partial C}{\partial y_{21}} * \frac{\partial y_{21}}{\partial z_{21}} * \frac{\partial z_{21}}{\partial y_{12}}+\frac{\partial C}{\partial y_{22}} * \frac{\partial y_{22}}{\partial z_{22}} * \frac{\partial z_{22}}{\partial y_{12}}) *\frac{\partial y_{12}}{\partial z_{12}}*\frac{\partial z_{12}}{\partial w_{23}}
= ((y_{21}-\hat y_{1}) * f^{'}(z_{21})*w_{45}+(y_{22}-\hat y_{2})*f^{'}(z_{22})*w_{46})*f^{'}(z_{12})*\hat x_{1}$$`

`$$ \frac{\partial C}{\partial w_{24}} 
= (\frac{\partial C}{\partial y_{21}} * \frac{\partial y_{21}}{\partial z_{21}} * \frac{\partial z_{21}}{\partial y_{12}}+\frac{\partial C}{\partial y_{22}} * \frac{\partial y_{22}}{\partial z_{22}} * \frac{\partial z_{22}}{\partial y_{12}}) *\frac{\partial y_{12}}{\partial z_{12}}*\frac{\partial z_{12}}{\partial w_{24}}
= ((y_{21}-\hat y_{1}) * f^{'}(z_{21})*w_{45}+(y_{22}-\hat y_{2})*f^{'}(z_{22})*w_{46})*f^{'}(z_{12})*\hat x_{2}$$`

同样进行两两合并：
`$$\begin{bmatrix}
\frac{\partial C}{\partial w_{13}} \\
\frac{\partial C}{\partial w_{14}}
\end{bmatrix} 
= ((y_{21}-\hat y_{1}) * f^{'}(z_{21})*w_{35}+(y_{22}-\hat y_{2})*f^{'}(z_{22})*w_{36})*f^{'}(z_{11})\times \hat x$$`

`$$\begin{bmatrix}
\frac{\partial C}{\partial w_{23}} \\
\frac{\partial C}{\partial w_{24}}
\end{bmatrix} 
= ((y_{21}-\hat y_{1}) * f^{'}(z_{21})*w_{45}+(y_{22}-\hat y_{2})*f^{'}(z_{22})*w_{46})*f^{'}(z_{12})\times \hat x$$`

因此：
`$$\frac{\partial C}{\partial W_{0}} 
= \begin{bmatrix}
\frac{\partial C}{\partial w_{13}} & \frac{\partial C}{\partial w_{14}}\\
\frac{\partial C}{\partial w_{23}} & \frac{\partial C}{\partial w_{24}}
\end{bmatrix} 
= \begin{bmatrix}
	((y_{21}-\hat y_{1}) * f^{'}(z_{21})*w_{35}+(y_{22}-\hat y_{2})*f^{'}(z_{22})*w_{36})*f^{'}(z_{11})\times \hat x^{T} \\ 
	((y_{21}-\hat y_{1}) * f^{'}(z_{21})*w_{45}+(y_{22}-\hat y_{2})*f^{'}(z_{22})*w_{46})*f^{'}(z_{12})\times \hat x^{T}
	\end{bmatrix} $$`
`$$= \begin{bmatrix}
	((y_{21}-\hat y_{1}) * f^{'}(z_{21})*w_{35}+(y_{22}-\hat y_{2})*f^{'}(z_{22})*w_{36})*f^{'}(z_{11})\\ 
	((y_{21}-\hat y_{1}) * f^{'}(z_{21})*w_{45}+(y_{22}-\hat y_{2})*f^{'}(z_{22})*w_{46})*f^{'}(z_{12})
	\end{bmatrix} \times \hat x^{T}
= \begin{bmatrix}
	((y_{21}-\hat y_{1}) * f^{'}(z_{21})*w_{35}+(y_{22}-\hat y_{2})*f^{'}(z_{22})*w_{36}) \\ 
	((y_{21}-\hat y_{1}) * f^{'}(z_{21})*w_{45}+(y_{22}-\hat y_{2})*f^{'}(z_{22})*w_{46})*
	\end{bmatrix} * \begin{bmatrix}
								f^{'}(z_{11}) \\
								f^{'}(z_{12})
							\end{bmatrix} \times \hat x^{T}$$`
`$$= \begin{bmatrix}
	w_{35} & w_{36} \\
	w_{45} & w_{46}
	\end{bmatrix}\times 
	\begin{bmatrix}
	(y_{21}-\hat y_{1}) * f^{'}(z_{21}) \\
	(y_{22}-\hat y_{2})*f^{'}(z_{22})
	\end{bmatrix}	* f^{'}(z_{1}) \times 	\hat x^{T}
=\begin{bmatrix}
	w_{35} & w_{36} \\
	w_{45} & w_{46}
	\end{bmatrix}\times 
	((y_{2}-\hat y) * f^{'}(z_{2}))* f^{'}(z_{1})\times \hat x^{T}
= W_{1} \times ((y_{2}-\hat y) * f^{'}(z_{2}))* f^{'}(z_{1})\times \hat x^{T}$$`
其中`$W_{1}$`为`$2\times 2$`的第二个权重矩阵，`$(y_{2}-\hat y) * f^{'}(z_{2})$`是`$2\times 1$`矩阵，`$\hat x^{T}$`是`$1\times 2$`矩阵
于是得到更新`$W_{0}$`的等式：`$W_{0} += -\eta \frac{\partial C}{\partial W_{0}}$`，`$\eta$`为学习速率

好了，大功告成，到这里我们推到出了`$W_{0}$`、`$W_{1}$`的偏导以及它们的更新等式啦

## 总结
通过以上的推倒，得到`$W_{0}$`、`$W_{1}$`的偏导如下：
`$$\frac{\partial C}{\partial W_{1}} 
= (y_{2}-\hat y) * f^{'}(z_{2})\times y^{T}_{1}$$`
`$$\frac{\partial C}{\partial W_{0}} 
=  W_{1} \times ((y_{2}-\hat y) * f^{'}(z_{2}))* f^{'}(z_{1})\times \hat x^{T}$$`
这个结果是在输入层、隐藏层、输出层的神经元节点个数均为两个的情况下得到，但是对于其他结构的全联接三层网络同样适用，那么对于等式中具体的含义呢，可以这样认为

- `$(y_{2}-\hat y)$`为输出层误差，`$ (y_{2}-\hat y) * f^{'}(z_{2})$`为输出层输入误差`$\delta{_2}$`，`$\delta{_2}\times \hat y^{T}_{1}$`为`$W_{1}$`的梯度
- `$W_{1}\times \delta{_2}$`为中间层的输出误差，`$W_{1}\times \delta{_2}*f^{'}(z_{1})$`为中间层的输入误差，`$\delta{_1}$，$\delta{_1}\times \hat x^{T}$`为`$W_{0}$`的梯度

那么对于多层网络呢，同样适用

- 以此类推，`$W_{i-1}\times \delta{_i}$`为第i-1层的输出误差，`$W_{i-1}\times \delta{_i}*f^{'}(z_{i-1})$`为中间层的输入误差`$\delta{_{i-1}}$，$\delta{_{i-1}}\times \hat y^{T}_{i-1}$`为`$W_{i-1}$`的梯度，`$W_{i-1}$`为i-1至i的权重

需要注意的是以上的等式是在建立在特征值向量`$\hat x$`、标签值向量`$\hat y$`使用列向量和权重矩阵`$W$`左乘以及不考虑偏置的情况下得到