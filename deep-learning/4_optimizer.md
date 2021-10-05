# 优化器



## 梯度下降变体

梯度下降有三种变体，它们在计算目标函数的梯度时使用的数据量上有所不同。根据数据量的不同，我们需要在参数更新的准确性和执行更新所需的时间之间进行权衡。

### Batch gradient descent

批量梯度下降法，用整个数据集来计算损失函数的梯度更新参数$\theta$：
$$
\theta=\theta-\eta \cdot \nabla_{\theta} J(\theta)
$$
然后在梯度的相反方向更新参数，并根据学习速率决定执行多大的更新。对于凸误差曲面，批量梯度下降法保证收敛到全局最小值，对于非凸误差曲面保证收敛到局部最小值。

缺点：

批处理梯度下降可能会非常缓慢，对于不适合内存的数据集来说是很难处理的。批量梯度下降也不允许在线更新模型，即使用新的动态示例。

### Stochastic gradient descent

随机梯度（SGD）是根据单个样本更新梯度：
$$
\theta=\theta-\eta \cdot \nabla_{\theta} J\left(\theta ; x^{(i)} ; y^{(i)}\right)
$$
SGD通过一次执行一个更新来消除这种冗余。因此，它通常要快得多，也可以用来在线学习。

SGD 的缺点在于收敛速度慢，而且可能会在沟壑的两边持续震荡，停留在一个局部最优点。

### Mini-batch gradient descent

小批量梯度下降是取$n$个样本进行梯度的计算、更新。
$$
\theta=\theta-\eta \cdot \nabla_{\theta} J\left(\theta ; x^{(i: i+n)} ; y^{(i: i+n)}\right)
$$
优点：

- 减小了参数更新的方差，使收敛更加稳定；
- 可以利用最先进的深度学习库中常见的高度优化的矩阵优化，使小批量计算梯度非常高效。

常见的mini-batch范围在50到256之间。小批量梯度下降法是训练神经网络时通常选择的算法，当使用小批量时也通常使用术语SGD。

Mini-batch梯度下降也存在着一些挑战：

- 选择一个合适的学习速度是很困难的。学习速率过小会导致收敛速度缓慢，而学习速率过大则会阻碍收敛，并导致损失函数在最小值附近波动甚至发散。

- 学习速率尝试在训练过程中调整学习速率；
- 此外，相同的学习率适用于所有参数更新。如果我们的数据是稀疏的，而我们的特征有非常不同的频率，我们可能不想更新所有的特征到相同的程度，而是对很少出现的特征执行更大的更新。
- 最小化神经网络常见的高度非凸误差函数的另一个关键挑战是避免陷入它们无数次最优的局部极小。Dauphin等人[3]认为，困难实际上并非来自局部极小值，而是来自鞍点，即一维向上倾斜而另一维向下倾斜的点。这些鞍点通常被相同误差的平台所包围，这使得SGD很难逃脱，因为梯度在所有维度上都接近于零。



## 梯度下降优化算法

### SGD with Momentum

SGD：

<img src="https://ruder.io/content/images/2015/12/without_momentum.gif" style="zoom:50%;" />

带有动量的SGD：

<img src="https://ruder.io/content/images/2015/12/with_momentum.gif" style="zoom:50%;" />

动量是一种帮助加速SGD在相关的方向的方法，抑制不断的震荡，
$$
\begin{aligned} v_{t} &=\gamma v_{t-1}+\eta \nabla_{\theta} J(\theta) \\ \theta &=\theta-v_{t} \end{aligned}
$$
从上述公式可以看到，梯度下降方向不仅由当前点的梯度方向决定，而且由此**前累积的下降方向决定**【有一个初始的速度】。动量系数$\gamma$ 一般取0.9。

一阶动量是各个时刻梯度方向的指数移动平均值

【对于梯度指向相同方向的维度，动量项增加，而对于梯度改变方向的维度，动量项减少更新】。结果得到了更快的收敛和减小了振荡。



### Nesterov accelerated gradient

然而，非常不令人满意的是，一个球从山上滚下来，盲目地跟随斜坡。我们希望有一个更聪明的球，这个球具有要去往何处的概念，以便在山坡再次变高之前知道它会慢下来。

我们使用【动量项$】\gamma v_{t-1}$来【移动参数$\theta$】，计算$\theta-\gamma v_{t-1}$可以【给出参数下一个位置的近似，参数将会在哪的粗略想法】：
$$
\begin{aligned} v_{t} &=\gamma v_{t-1}+\eta \nabla_{\theta} J\left(\theta-\gamma v_{t-1}\right) \\ \theta &=\theta-v_{t} \end{aligned}
$$
同样，我们设置动量项$\gamma$值约为0.9。动量首先计算当前梯度（蓝色小矢量），然后在更新的累积梯度（蓝色矢量）的方向上发生较大的跃迁，【而NAG首先在先前的累积梯度的方向上进行较大的跃迁（棕色矢量），测量梯度，然后进行校正（红色矢量），从而完成NAG更新（绿色矢量）】。此预期更新可防止我们过快地执行并导致响应速度增加，从而显着提高了RNN在许多任务上的性能。

<img src="https://ruder.io/content/images/2016/09/nesterov_update_vector.png" style="zoom:50%;" />





### Adagrad

Adagrad根据**参数调整学习速率**，进行较小的更新(低学习率)与频繁发生的特征相关联的参数，和较大的更新(高学习率)与不频繁的特征相关联的参数。所以它非常适合处理稀疏数据。

之前，我们更新参数$\theta$时都是以相同的学习率，但是Adagrad却在【每个时间步以不同的学习率来更新参数】。假设$g_t$为$t$时刻的梯度，$g_{ti}$表示$t$时刻参数$\theta_i$的梯度：
$$
g_{t, i}=\nabla_{\theta} J\left(\theta_{t, i}\right)
$$
SGD更新的方式：
$$
\theta_{t+1, i}=\theta_{t, i}-\eta \cdot g_{t, i}
$$
Adagrad【基于过去的梯度】来更新学习率：
$$
\theta_{t+1, i}=\theta_{t, i}-\frac{\eta}{\sqrt{G_{t, i i}+\epsilon}} \cdot g_{t, i}
$$
其中$G_t \in \mathbb{R}^{d\times d}$是一个对角矩阵，对角矩阵上的元素（坐标$i,i$）是在$t$时刻的过去梯度平方和，$\epsilon$是一个平滑系数，防止分母为0（一般取1e-8）。

我们可以向量化实现：
$$
\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{G_{t}+\epsilon}} \odot g_{t}
$$
Adagrad的【主要优点之一是它无需手动调优学习速率】。大多数实现都使用默认值0.01，并保持不变。

Adagrad的主要缺点是其在分母上的平方梯度的积累:【由于每一项相加都是正的，累积的总和在训练过程中不断增长。这反过来导致学习速度收缩，最终变得无限小，此时算法不再能够获得额外的知识】。



### Adadelta

Adadelta是对Adagrad的改进，【Adagrad选择过去所有的梯度平方和来更新学习率，而Adadelta则采用了一个**窗口（范围）**选择计算梯度平方和】。

当然并不是低效的存储过去的$w$个梯度平方和，梯度的总和被【递归】定义为【所有过去的平方梯度的衰减平均值】。$E[g^2]_t$只依赖于【过去平均】和【当前的梯度】（$\gamma$类似于动量项，0.9）：
$$
E\left[g^{2}\right]_{t}=\gamma E\left[g^{2}\right]_{t-1}+(1-\gamma) g_{t}^{2}
$$
因此对于SGD来说：
$$
\begin{array}{l}\Delta \theta_{t}=-\eta \cdot g_{t, i} \\ \theta_{t+1}=\theta_{t}+\Delta \theta_{t}\end{array}
$$
Adagrad：
$$
\Delta \theta_{t}=-\frac{\eta}{\sqrt{G_{t}+\epsilon}} \odot g_{t}
$$
Adadelta：
$$
\Delta \theta_{t}=-\frac{\eta}{\sqrt{E\left[g^{2}\right]_{t}+\epsilon}} g_{t}
$$

### RMSprop

与Adadelta基本类似【同一时间独立开发】
$$
\begin{aligned} E\left[g^{2}\right]_{t} &=0.9 E\left[g^{2}\right]_{t-1}+0.1 g_{t}^{2} \\ \theta_{t+1} &=\theta_{t}-\frac{\eta}{\sqrt{E\left[g^{2}\right]_{t}+\epsilon}} g_{t} \end{aligned}
$$
$\gamma=0.9,\eta=0.001$。



### Adam

自适应动量估计（Adaptive Moment Estimation，Adam）也是采用【自适应的学习率】。除了存储【过去平方梯度的指数衰减平均值$v_t$】，Adam还保留了【过去梯度的衰减平均值$m_t$】。动量可以被看作是一个从斜坡上跑下来的球，而Adam则表现得像一个有摩擦力的重球，因此在误差面中倾向于平坦的最小值。计算方式：
$$
\begin{aligned} m_{t} &=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\ v_{t} &=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2} \end{aligned}
$$
$m_t,v_t$分别是梯度的第一动量（均值）和第二个动量（偏移方差）的估计。

在迭代初始阶段，![[公式]](https://www.zhihu.com/equation?tex=m_t) 和 ![[公式]](https://www.zhihu.com/equation?tex=v_t) 有一个向初值的偏移（过多的偏向了 0）。因此，可以对一阶和二阶动量做偏置校正 (bias correction)，Z
$$
\begin{aligned} \hat{m}_{t} &=\frac{m_{t}}{1-\beta_{1}^{t}} \\ \hat{v}_{t} &=\frac{v_{t}}{1-\beta_{2}^{t}} \end{aligned}
$$
Adam更新规则：
$$
\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}
$$
其中$\beta_1=0.9,\beta_2=0.999,\epsilon=1e-8$。



## 面试题

1、SGD,Momentum,Adagard,Adam 原理

SGD为随机梯度下降,每一次迭代计算数据集的mini-batch的梯度,然后对参数进行跟新。

Momentum参考了物理中动量的概念,前几次的梯度也会参与到当前的计算中,但是前几轮的梯度叠加在当前计算中会有一定的衰减。

Adagard在训练的过程中可以自动变更学习的速率,设置一个全局的学习率,而实际的学习率与以往的参数模和的开方成反比。

Adam利用梯度的一阶动量估计和二阶动量估计动态调整每个参数的学习率,在经过偏置的校正后,每一次迭代后的学习率都有个确定的范围,使得参数较为平稳。



## 参考资料

[1] https://ruder.io/optimizing-gradient-descent/index.html#gradientdescentvariants

[2] https://zhuanlan.zhihu.com/p/32230623

