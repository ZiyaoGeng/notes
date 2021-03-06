# Normalization

学习大纲：

1. normalization需要解决的问题；
2. batch normalization在训练与推理时的区别；
3. batch与layer normalization的区别；



随着训练的进行，网络中的参数也随着梯度下降在不停更新。一方面，当底层网络中【参数发生微弱变化】时，由于【每一层中的线性变换与非线性激活映射】，这些微弱变化随着网络层数的加深而被放大（类似蝴蝶效应）；另一方面，【参数的变化导致每一层的输入分布会发生改变】，进而【上层的网络需要不停地去适应这些分布变化】，使得我们的【模型训练变得困难】。上述这一现象叫做Internal Covariate Shift。



## 1. Internal Covariate Shift

在深层网络训练的过程中，**由于网络中参数变化而引起内部结点数据分布发生变化**的这一过程被称作Internal Covariate Shift。

定义每一层的线性变换为$$Z^{[l]}=W^{[l]} \times i n p u t+b^{[l]}$$，其中$$l$$ 代表层数；非线性变换为$$A^{[l]}=g^{[l]}\left(Z^{[l]}\right)$$，其中$$g^{[l]}(\cdot)$$为第 $$l$$层的激活函数。

随着梯度下降的进行，每一层的参数$$W^{[l]}$$ 与$$b^{[l]}$$ 都会被更新，那么$$Z^{[l]}$$ 的分布也就发生了改变，进而 ![[公式]](https://www.zhihu.com/equation?tex=A%5E%7B%5Bl%5D%7D) 也同样出现分布的改变。而 ![[公式]](https://www.zhihu.com/equation?tex=A%5E%7B%5Bl%5D%7D) 作为第$$l+1$$ 层的输入，意味着$$l+1$$  层就需要去【不停适应这种数据分布的变化】，这一过程就被叫做Internal Covariate Shift。

Internal Covariate Shift带来的【问题】：

**（1）上层网络需要不停调整来【适应输入数据分布的变化】，导致网络【学习速度的降低】**

我们在上面提到了梯度下降的过程会让每一层的参数$$W^{[l]}$$ 和$$b^{[l]}$$ 发生变化，进而使得每一层的线性与非线性计算结果分布产生变化。**后层网络就要不停地去适应这种分布变化，这个时候就会使得整个网络的学习速率过慢**。

**（2）网络的训练过程容易陷入【梯度饱和】区，减缓网络收敛速度**

当我们在神经网络中采用**饱和激活函数**（saturated activation function）时，例如sigmoid，tanh激活函数，很容易使得模型训练陷入梯度饱和区（saturated regime）。随着模型训练的进行，我们的参数$$W^{[l]}$$ 会逐渐更新并变大，此时$$Z^{[l]}=W^{[l]} A^{[l-1]}+b^{[l]}$$ 就会随之变大，并且$$Z^{[l]}$$ 还受到更底层网络参数$$W^{[1]}, W^{[2]}, \cdots, W^{[l-1]}$$ 的影响，随着网络层数的加深，$$Z^{[l]}$$ 很【容易陷入梯度饱和区】，此时【**梯度会变得很小甚至接近于0**】，参数的更新速度就会减慢，进而就会放慢网络的收敛速度。

对于激活函数梯度饱和问题，有两种解决思路。第一种就是更为非饱和性激活函数，例如线性整流函数ReLU可以在一定程度上解决训练进入梯度饱和区的问题。另一种思路是，我们可以让【**激活函数的输入分布保持在一个稳定状态来尽可能避免它们陷入梯度饱和区**】，这也就是Normalization的思路。



### 1.1 如何缓解？

要缓解ICS的问题，就要明白它产生的原因。ICS产生的原因是由于参数更新带来的网络中每一层输入值分布的改变，并且随着网络层数的加深而变得更加严重，因此我们可以通过固定每一层网络输入值的分布来对减缓ICS问题。

**（1）白化（Whitening）**

白化（Whitening）是机器学习里面常用的一种规范化数据分布的方法，主要是PCA白化与ZCA白化。白化是对输入数据分布进行变换，进而达到以下两个目的：

- **使得输入特征分布具有相同的均值与方差。**其中PCA白化保证了所有特征分布均值为0，方差为1；而ZCA白化则保证了所有特征分布均值为0，方差相同；
- **去除特征之间的相关性。**

通过白化操作，我们可以减缓ICS的问题，进而固定了每一层网络输入分布，加速网络训练过程的收敛。

**（2）Batch Normalization提出**

既然白化可以解决这个问题，为什么我们还要提出别的解决办法？当然是现有的方法具有一定的缺陷，白化主要有以下两个问题：

- **白化过程计算成本太高，**并且在每一轮训练中的每一层我们都需要做如此高成本计算的白化操作；
- **白化过程由于改变了网络每一层的分布**，因而**改变了网络层中本身数据的表达能力**。底层网络学习到的参数信息会被白化操作丢失掉。

既然有了上面两个问题，那我们的解决思路就很简单，一方面，我们提出的normalization方法要能够**简化计算过程**；另一方面又需要**经过规范化处理后让数据尽可能保留原始的表达能力**。于是就有了【简化+改进版的白化】——Batch Normalization。



## 2. Normalization

既然白化计算过程比较复杂，那我们就简化一点，比如我们可以尝试单独对每个特征进行normalizaiton就可以了，让每个特征都有均值为0，方差为1的分布就OK。

另一个问题，既然白化操作减弱了网络中每一层输入数据表达能力，那我就再加个线性变换操作，让这些数据再能够尽可能恢复本身的表达能力就好了。

因此，Normalization的通用公式为：
$$
h=f\left(\mathbf{g} \cdot \frac{\mathbf{x}-\mu}{\sigma + \epsilon}+\mathbf{b}\right)
$$

### 2.1 Batch Normalization

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210321153529184.png" alt="image-20210321153529184" style="zoom:50%;" />

Batch Normalization 于2015年由 Google 提出，开 Normalization 之先河。其规范化针对单个神经元进行，利用网络训练时一个 mini-batch 的数据来计算该神经元 ![[公式]](https://www.zhihu.com/equation?tex=x_i) 的均值和方差,因而称为 Batch Normalization。
$$
\mu_{i}=\frac{1}{M} \sum x_{i}, \quad \sigma_{i}=\sqrt{\frac{1}{M} \sum\left(x_{i}-\mu_{i}\right)^{2}+\epsilon}
$$
其中$$M$$ 是 mini-batch 的大小。

按上图所示，相对于一层神经元的水平排列，BN 可以看做一种纵向的规范化。由于 BN 是针对单个维度定义的，因此标准公式中的计算均为 element-wise 的。

BN 独立地规范化每一个输入维度$$x_i$$ ，【但规范化的参数是一个 mini-batch 的一阶统计量和二阶统计量】。这就要求【 每一个 mini-batch 的统计量是整体统计量的近似估计】，或者说【每一个 mini-batch 】彼此之间，以及和整体数据，都应该是【近似同分布的】。分布差距较小的 mini-batch 可以看做是为规范化操作和模型训练引入了噪声，可以增加模型的鲁棒性；但如果每个 mini-batch的原始分布差别很大，那么不同 mini-batch 的数据将会进行不一样的数据变换，这就增加了模型训练的难度。

因此，BN 比较适用的场景是：每个 mini-batch 比较大，数据分布比较接近。在进行训练之前，要做【好充分的 shuffle. 否则效果会差很多】。

#### 2.1.1 优点

Batch Normalization在实际工程中被证明了能够缓解神经网络难以训练的问题，BN具有的优势可以总结为以下三点：

**（1）BN使得网络中每层输入数据的分布相对稳定，加速模型学习速度**

BN通过规范化与线性变换使得每一层网络的输入数据的均值与方差都在一定范围内，使得后一层网络不必不断去适应底层网络中输入的变化，从而实现了网络中层与层之间的解耦，允许每一层进行独立学习，有利于提高整个神经网络的学习速度。

**（2）BN使得模型对网络中的参数不那么敏感，简化调参过程，使得网络学习更加稳定**

在使用Batch Normalization之后，抑制了参数微小变化随着网络层数加深被放大的问题，使得网络对参数大小的适应能力更强，此时我们可以设置较大的学习率而不用过于担心模型divergence的风险。

**（3）BN允许网络使用饱和性激活函数（例如sigmoid，tanh等），缓解梯度消失问题**

在不使用BN层的时候，由于网络的深度与复杂性，很容易使得底层网络变化累积到上层网络中，导致模型的训练很容易进入到激活函数的梯度饱和区；**通过normalize操作可以让激活函数的输入数据落在梯度非饱和区**，缓解梯度消失的问题；另外通过自适应学习$$\gamma$$ 与$$\beta$$ 又让数据保留更多的原始信息。

**（4）BN具有一定的正则化效果**

在Batch Normalization中，由于我们使用mini-batch的均值与方差作为对整体训练样本均值与方差的估计，尽管每一个batch中的数据都是从总体样本中抽样得到，但不同mini-batch的均值与方差会有所不同，这就为网络的学习过程中增加了随机噪音，与Dropout通过关闭神经元给网络训练带来噪音类似，在一定程度上对模型起到了正则化的效果。

#### 2.1.2 Tensorflow

训练中，直接计算每个mini-batch中每个神经元的均值和方差；

在评估或测试时，均值和方差采用的是移动平均`(batch - self.moving_mean) / (self.moving_var + epsilon) * gamma + beta`

其中，【移动平均】的均值和方差并不是训练参数，而是在训练模式中进行更新的：

- `moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)`
- `moving_var = moving_var * momentum + var(batch) * (1 - momentum)`



### 2.2 Layer Normalization

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210321154115689.png" alt="image-20210321154115689" style="zoom:50%;" />

层规范化就是针对 BN 的上述不足而提出的。与 BN 不同，LN 是一种**横向的规范化**，如图所示。它综合考虑**一层所有维度的输入**，计算该层的平均输入值和输入方差，然后用同一个规范化操作来转换各个维度的输入。

![[公式]](https://www.zhihu.com/equation?tex=%5Cmu+%3D+%5Csum_i%7Bx_i%7D%2C+%5Cquad+%5Csigma%3D+%5Csqrt%7B%5Csum_i%7B%28x_i-%5Cmu%29%5E2%7D%2B%5Cepsilon+%7D%5C%5C)

其中$$i$$ 枚举了该层所有的输入神经元。对应到标准公式中，四大参数$$\mu, \sigma, g,b$$ 均为标量（BN中是向量），所有输入共享一个规范化变换。

**LN 针对单个训练样本进行，不依赖于其他数据，因此可以避免 BN 中受 mini-batch 数据分布影响的问题**，可以用于 小mini-batch场景。此外，**LN 不需要保存 mini-batch 的均值和方差，节省了额外的存储空间**。

但是，BN 的转换是针对单个神经元可训练的——不同神经元的输入经过再平移和再缩放后分布在不同的区间，而 LN 对于一整层的神经元训练得到同一个转换——所有的输入都在同一个区间范围内。**如果不同输入特征不属于相似的类别（比如颜色和大小），那么 LN 的处理可能会降低模型的表达能力**。



## 参考资料

[1] [详解深度学习中的Normalization，BN/LN/WN](https://zhuanlan.zhihu.com/p/33173246)

[2] [Batch Normalization原理与实战](https://zhuanlan.zhihu.com/p/34879333)

