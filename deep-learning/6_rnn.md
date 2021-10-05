# 循环神经网络



## 1. 基础

### 1.1 RNN

循环神经网络是一类**具有短期记忆能力**的神经网络。神经元不但可以接受其他神经元的信息，也可以接受自身的信息，形成具有环路的网络结构。

循环神经网络的参数学习可以通过随时间反向传播算法，即按照时间的逆序将错误信息一步步往前传递。当输入序列比较长时，会存在梯度爆炸和消失问题，也称长程依赖问题。---->解决问题，【引入门控机制】。

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210328082323494.png" alt="image-20210328082323494" style="zoom:50%;" />

RNN 的统一定义为：
$$
h_{t}=f\left(x_{t}, h_{t-1} ; \theta\right)
$$
其中$$h_t$$是每一步的输出，它由当前输入$$x_t$$和前一时刻输出$$h_{t-1}$$共同决定，而$$\theta$$则是可训练参数。在做最基本的分析时，我们可以假设$$h_t,x_t,\theta$$都是一维的。我们定义的模型有一个比较合理的梯度。我们可以求得：
$$
\frac{d h_{t}}{d \theta}=\frac{\partial h_{t}}{\partial h_{t-1}} \frac{d h_{t-1}}{d \theta}+\frac{\partial h_{t}}{\partial \theta}
$$
可以看到，其实 RNN 的梯度也是一个 RNN，当前时刻梯度$$\frac{d h_{t}}{d \theta}$$是前一时刻梯度$$\frac{d h_{t-1}}{d \theta}$$与当前运算梯度$$\frac{\partial h_{t}}{\partial \theta}$$的函数。同时，从上式我们就可以看出，其实梯度消失或者梯度爆炸现象几乎是必然存在的：

当$$\left|\frac{\partial h_{t}}{\partial h_{t-1}}\right|<1$$时，意味着历史的梯度信息是衰减的，因此步数多了梯度必然消失（好比$$\lim _{n \rightarrow \infty} 0.9^{n} \rightarrow 0$$）；当$$\left|\frac{\partial h_{t}}{\partial h_{t-1}}\right|>1$$ ，因为这历史的梯度信息逐步增强，因此步数多了梯度必然爆炸（好比$$\lim _{n \rightarrow \infty} 1.1^{n} \rightarrow \infty$$）。总不可能一直$$\left|\frac{\partial h_{t}}{\partial h_{t-1}}\right|=1$$吧？当然，也有可能有些时刻大于 1，有些时刻小于 1，最终稳定在 1 附近，但这样概率很小，需要很精巧地设计模型才行。

所以步数多了，**梯度消失或爆炸几乎都是不可避免的，我们只能对于有限的步数去缓解这个问题。**



### 1.2 LSTM

LSTM通过引入门控机制来控制信息传递的路径。控制内部状态应该遗忘多少历史信息，输入多少新信息，以及输出多少信息。

（1）遗忘门$$\boldsymbol{f}_t$$控制上一个时刻的内部状态$$\boldsymbol{c}_{t-1}$$需要遗忘多少信息。

（2）输入门$$\boldsymbol{i}_t$$控制当前时刻的候选状态$$\tilde{\boldsymbol{c}_t}$$有多少信息保存；
$$
\tilde{\boldsymbol{c}_t}=tanh(\boldsymbol{W}_c\boldsymbol{x}_t+\boldsymbol{U}_c\boldsymbol{h}_{t-1}+\boldsymbol{b}_c)
$$
（3）输出门$$\boldsymbol{o}_t$$控制当前时刻的内部状态$$\boldsymbol{c}_t$$有多少信息需要输出给外部状态$$\boldsymbol{h}_t$$；
$$
\boldsymbol{c}_t=\boldsymbol{f}_t \odot tanh(\boldsymbol{c}_{t-1})+\boldsymbol{i}_t \odot \tilde{\boldsymbol{c}_t} \\ \boldsymbol{h}_t= \boldsymbol{o}_t \odot tanh(\boldsymbol{c}_{t})
$$
门控的计算方式：
$$
\begin{array}{l}\boldsymbol{i}_{t}=\sigma\left(W_{i} \boldsymbol{x}_{t}+U_{i} \boldsymbol{h}_{t-1}+\boldsymbol{b}_{i}\right) \\ \boldsymbol{f}_{t}=\sigma\left(W_{f} \boldsymbol{x}_{t}+U_{f} \boldsymbol{h}_{t-1}+\boldsymbol{b}_{f}\right) \\ \boldsymbol{o}_{t}=\sigma\left(W_{o} \boldsymbol{x}_{t}+U_{o} \boldsymbol{h}_{t-1}+\boldsymbol{b}_{o}\right)\end{array}
$$
<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210328084424919.png" alt="image-20210328084424919" style="zoom:50%;" />

1）首先利用上一时刻的外部状态$$\boldsymbol{h}_{t-1}$$和当前时刻的输入$$\boldsymbol{x}_t$$，计算出三个门，以及候选状态$$\tilde{\boldsymbol{c}_t}$$；

2）结合遗忘门$$\boldsymbol{f}_t$$和输入门$$\boldsymbol{i}_t$$来更新记忆单元$$\boldsymbol{c}_t$$；

3）结合输出门$$\boldsymbol{o}_t$$将内部状态的信息传递给外部状态$$\boldsymbol{h}_t$$；



### 1.3 GRU

GRU 网络引入门控机制来控制信息更新的方式。和 LSTM 不同，GRU 不 引入额外的记忆单元，GRU 网络引入一个更新门 （Update Gate）来控制当前状态需要**从历史状态中保留多少信息（不经过非线性变换）**，以及需要**从候选状态中接受多少新信息**。
$$
\boldsymbol{h}_{t}=\boldsymbol{z}_{t} \odot \boldsymbol{h}_{t-1}+\left(1-\boldsymbol{z}_{t}\right) \odot \tilde{\boldsymbol{h}}_{t})
$$
其中$$\boldsymbol{z}_{t} \in[0,1]$$为更新门
$$
\boldsymbol{z}_{t}=\sigma\left(\boldsymbol{W}_{z} \boldsymbol{x}_{t}+\boldsymbol{U}_{z} \boldsymbol{h}_{t-1}+\boldsymbol{b}_{z}\right)
$$
在 LSTM 网络中，输入门和遗忘门是互补关系，具有一定的冗余性。**GRU网络直接使用一个门来控制输入和遗忘之间的平衡**。

函数$$\tilde{\boldsymbol{h}}_{t}$$的定义为：
$$
\tilde{\boldsymbol{h}}_{t}=\tanh \left(W_{h} \boldsymbol{x}_{t}+U_{h}\left(\boldsymbol{r}_{t} \odot \boldsymbol{h}_{t-1}\right)+\boldsymbol{b}_{h}\right)
$$
其中$$\tilde{\boldsymbol{h}}_{t}$$表示当前时刻的候选状态，$$\boldsymbol{r}_{t} \in[0,1]$$为**重置门**Reset Gate），用来控制候选状态 $$\tilde{\boldsymbol{h}}_{t}$$的计算是否依赖上一时刻的状态$$\boldsymbol{h}_{t-1}$$。
$$
\boldsymbol{r}_{t}=\sigma\left(W_{r} \boldsymbol{x}_{t}+U_{r} \boldsymbol{h}_{t-1}+\boldsymbol{b}_{r}\right)
$$
<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210328091736245.png" alt="image-20210328091736245" style="zoom:50%;" />



## 2. 面试

**1、为什么RNN 训练的时候Loss波动很大**

​	由于RNN特有的记忆会影响后期其他的RNN的特点，梯度时大时小，learning rate没法个性化的调整，导致RNN在train的过程中，Loss会震荡起伏，为了解决RNN的这个问题，在训练的时候，可以设置临界值，当梯度大于某个临界值，直接截断，用这个临界值作为梯度的大小，防止大幅震荡。



**2、RNN中为什么会出现梯度消失？**

tanh函数及导数图：

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/tanh.jpg" style="zoom: 33%;" />

sigmoid函数及导数图：

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/sigmoid.jpg" style="zoom:33%;" />

sigmoid函数的导数范围是(0,0.25]，tanh函数的导数范围是(0,1]，他们的导数最大都不大于1。

RNN中对参数$$W,U$$求偏导，结果：
$$
\frac{\partial L^{(t)}}{\partial W}=\sum_{k=0}^{t}\frac{\partial L^{(t)}}{\partial o^{(t)}}\frac{\partial o^{(t)}}{\partial h^{(t)}}(\prod_{j=k+1}^{t}\frac{\partial h^{(j)}}{\partial h^{(j-1)}})\frac{\partial h^{(k)}}{\partial W}
$$

$$
\frac{\partial L^{(t)}}{\partial U}=\sum_{k=0}^{t}\frac{\partial L^{(t)}}{\partial o^{(t)}}\frac{\partial o^{(t)}}{\partial h^{(t)}}(\prod_{j=k+1}^{t}\frac{\partial h^{(j)}}{\partial h^{(j-1)}})\frac{\partial h^{(k)}}{\partial U}
$$

其中如果选择激活函数为$$tanh$$或$$sigmoid$$，则：
$$
\prod_{j=k+1}^{t}{\frac{\partial{h^{j}}}{\partial{h^{j-1}}}} = \prod_{j=k+1}^{t}{tanh^{'}}\cdot W_{s}
$$

$$
\prod_{j=k+1}^{t}{\frac{\partial{h^{j}}}{\partial{h^{j-1}}}} = \prod_{j=k+1}^{t}{sigmoid^{'}}\cdot W_{s}
$$

**梯度消失现象**：基于上式，会发现**累乘会导致激活函数导数的累乘**，如果取tanh或sigmoid函数作为激活函数的话，那么必然是一堆小数在做乘法，结果就是越乘越小。随着时间序列的不断深入，小数的累乘就会导致梯度越来越小直到接近于0，这就是“梯度消失“现象。

​	实际使用中，会优先选择tanh函数，原因是tanh函数相对于sigmoid函数来说梯度较大，**收敛速度更快且引起梯度消失更慢**。



**3、如何解决RNN中的梯度消失问题？**

梯度消失是在无限的利用历史数据而造成，但是RNN的特点本来就是能利用历史数据获取更多的可利用信息，解决RNN中的梯度消失方法主要有：

1. 选取更好的激活函数，如Relu激活函数。ReLU函数的左侧导数为0，右侧导数恒为1，这就避免了“梯度消失“的发生。但恒为1的导数容易导致“梯度爆炸“，但设定合适的阈值可以解决这个问题；
2. 加入BN层，其优点包括可加速收敛、控制过拟合，可以少用或不用Dropout和正则、降低网络对初始化权重不敏感，且能允许使用较大的学习率等。
3. 改变传播结构，LSTM结构可以有效解决这个问题。



**4、LSTMs与GRUs的区别？**

LSTMs与GRUs的区别如图所示：

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/figure_6.6.6_2.png" style="zoom:10%;" />

从上图可以看出，二者结构十分相似，**不同在于**：

1. new memory都是根据之前state及input进行计算，但是GRUs中有一个reset gate控制之前state的进入量，而在LSTMs里没有类似gate；
2. 产生新的state的方式不同，LSTMs有两个不同的gate，分别是forget gate (f gate)和input gate(i gate)，而GRUs只有一种update gate(z gate)；
3. LSTMs对新产生的state可以通过output gate(o gate)进行调节，而GRUs对输出无任何调节。



## 参考资料

[1] [也来谈谈RNN的梯度消失/爆炸问题](https://mp.weixin.qq.com/s/-X77bR-G_OTudzUqCMOF5w)