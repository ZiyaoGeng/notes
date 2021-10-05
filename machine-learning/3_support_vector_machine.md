# 支持向量机

学习大纲：

1. 三种支持向量机：线性可分支持向量机，线性支持向量机，非线性支持向量机；
2. 感知机向SVM的过渡；
3. SVM的公式推导过程，间隔最大化—>原始问题（凸二次规划问题）—>拉格朗日函数—>对偶问题—>利用KKT条件求解；
4. hinge loss，支持向量，核函数等；
5. 为什么要将原问题转为对偶问题，为什么要引入核函数；



## 1. 感知机与SVM

> 感知机--->缺陷--->加入约束/优化策略（为什么可行）--->支持向量机

首先给出支持向量机的定义：

> 支持向量机（SVM）是一个**二分类模型**，是定义在特征空间上的**间隔最大**的分类器。

可以发现，与感知机的区别在于【间隔最大】这个概念。首先，从几何角度来说，感知机就是在特征空间$$\mathbf{R}^n$$上找到一个超平面，这个超平面能刚好将分布在特种空间上的正负两类点给区分开来。那么**不做任何的约束**，这个超平面就会有无穷多个（线性可分的前提下）。那么就存在一个**关键问题**：如何挑选一个最优的超平面（也可以理解为如何让模型更加的健壮）？这就需要找到找到一个新的学习策略/约束。

将其公式化，对于感知机来说，定义一个超平面：
$$
w^*\cdot x + b^* = 0
$$
相对应的决策函数为：
$$
f(x) = sign(w^* \cdot x + b^*)
$$
通过对**误分类点到超平面的距离的总和**这一损失函数进行建模，通过SGD等方法进行优化。

以上是**经验风险最小化**，存在的缺陷：

- 解不唯一（超平面不唯一）；
- 容易受数据集影响；

那么就需要加入一个新的学习策略/约束，即**间隔最大化**，可以使其**结构风险最小**。学习的目标就是来找到一个最优超平面。

再将间隔最大化给剥离出来，存在两个问题：

1. 最大间隔化指的是什么？
2. 为什么最大间隔化可以找到一个唯一的超平面？

对于第1个问题，一个点（样本）到超平面的远近可以表示分类预测的确信程度（离超平面越远越准确，超平面附近的点不确定性更大）。点到平面距离的公式为：$$\frac{1}{\|w\|}|w\cdot x+b|$$，假设两种分类点的标签为$$\{-1,+1\}$$，那么每个样本的确信程度可以表示为：$$\frac{1}{\|w\|}y_i(w\cdot x+b)$$，对其进行简化，如果某个点的确信程度大于等于0的话，则就分类正确，$$y_i(w\cdot x+b)\ge0$$。

以上就是**函数间隔**的定义，

> 对于给定的训练集$$T$$和超平面$$(w,b)$$【超平面由法向量$$w$$和截距$$b$$所确定，因此可以用$$(w,b)$$来表示超平面】，定义超平面$$(w, b)$$关于样本点$$(x_i,y_i)$$的函数间隔为：
> $$
> \hat \gamma_i=y_i(w \cdot x_i+b)
> $$
> 定义**超平面$$(w,b)$$关于训练集T的函数间隔**为超平面关于$$T$$中所有样本点的函数间隔的最小值，即
> $$
> \hat \gamma=\min_{i=1,...,N}\hat \gamma_i
> $$

我们知道，如果等比例改变$$w,b$$，超平面不变，但函数间隔会变化，所以又定义了一个**几何间隔**（添加规范化约束）。



> 对于给定的训练集$$T$$和超平面$$(w,b)$$，定义超平面$$(w, b)$$关于样本点$$(x_i,y_i)$$的几何间隔为：
> $$
> \gamma_i=y_i(\frac{w}{\|w\|} \cdot x_i+\frac{b}{\|w\|})
> $$
> 定义**超平面$$(w,b)$$关于训练集T的几何间隔**为超平面关于$$T$$中所有样本点的几何间隔的最小值，即
> $$
> \gamma=\min_{i=1,...,N} \gamma_i
> $$

所以最大间隔化指的就是**在能够正确划分训练数据集条件下，使得几何间隔最大化**，这就是支持向量机的学习策略/思想。

因此，可以公式化为：
$$
\begin{array}{cl} \displaystyle\max_{w, b} & \gamma \\ \text { s.t. } & y_{i}\left(\frac{w}{\|w\|} \cdot x_{i}+\frac{b}{\|w\|}\right) \geqslant \gamma, \quad i=1,2, \cdots, N\end{array}
$$
通过一系列简化（参考《统计学习方法》），最终得到：
$$
\begin{array}{cl} \displaystyle\min_{w, b} &\frac{1}{2}{\|w\|}^2 \\ \text { s.t. } & y_{i}\left(w \cdot x_{i}+b\right) -1\geqslant 0, \quad i=1,2, \cdots, N\end{array}
$$
这个公式，就是整个支持向量机模型的构建。

【注】这里是线性可分支持向量机的定义。

对于第2个问题，存在定理（这里先考虑线性可分的情况）：

> 最大间隔分离超平面的存在唯一性：若训练数据集线性可分，则可将训练数据集中的样本点完全正确分开的最大间隔分离超平面是存在且唯一的。

证明参考《统计学习方法》P117-P118。





## 2. 线性可分支持向量机与硬间隔最大化

接下来具体梳理下SVM（以线性可分为例）的推导过程。



### 2.1 将硬间隔最大化问题转化为凸二次优化问题

考虑一个二分类问题，给定特征空间上的训练数据集$$T=\{(x_1,y_1),(x_2,y_2),\dots,(x_n,y_n)\}$$，其中，$$x_i\in \mathcal{X}=\mathbf{R}^n,y_i\in \mathcal{Y}=\{+1,-1\},i=1,2,\dots,N$$。

一般地，当训练数据集线性可分时，存在无穷多个分离超平面可将两类数据正确分类。感知机是利用误分类最小策略，求得分离超平面，解有无穷多个。我们希望在这些超平面中选一个最好的，即找到一个“最中间”的超平面$$w^*\cdot x+b^*=0$$，它离所有点的距离最远，称之为**最大间隔分离超平面**，或**最大间隔分类器**。

若定义$$\text{margin} (w,b)$$为超平面$$(w,b)$$关于训练数据集$$T$$的间隔，要使间隔最大，且所有点分类正确，即：
$$
\begin{align*}
&\max \quad\text{margin(w,b)} \\
&s.t.\quad\begin{cases}
w^Tx_i+b>0,y_i=+1 \\
w^Tx_i+b<0,y_i=-1
\end{cases}

\end{align*}
$$
简化一下，即：
$$
\begin{align*}
&\max \quad\text{margin(w,b)} \\
&s.t. \quad
y_i(w^Tx_i+b)>0,i=1,2,\dots,N



\end{align*} \tag{1.1}
$$

对于式$$(1.1)$$，令其中的$$\text{margin}(w,b)$$为几何间隔$$\gamma$$，因为$$\gamma$$是所有点几何间隔的最小值，所以约束条件改为超平面关于每个样本点的几何间隔大于等于$$\gamma$$，即求：
$$
\begin{align*}
&\max\limits_{w,b} \quad \gamma \\
&s.t. \quad
\frac{1}{||w||}y_i(w^Tx_i+b) \geqslant \gamma,\quad i=1,2,\dots,N
\end{align*} \tag{1.2}
$$
考虑函数间隔$$\hat{\gamma}$$和几何间隔$$\gamma$$的关系：$$\hat{\gamma}=\gamma \cdot ||w||$$，可以把式$$(1.2)$$转化为下式：
$$
\begin{align*} &\max\limits_{w,b} \quad 

\frac{\hat{\gamma}}{||w||} \\ &s.t. \quad y_i(w^Tx_i+b) \geqslant \hat{\gamma},\quad i=1,2,\dots,N \end{align*} \tag{1.3}
$$
上式是求关于$$w,b$$的最值，**$$\hat{\gamma}$$的取值并不影响最优解的结果**（若$$w$$和$$b$$变为$$\lambda w$$和$$\lambda b$$，那么$$\hat \gamma$$也变为$$\lambda \hat \gamma$$，代入上式中发现式子并没有发生改变），那可以**令$$\hat{\gamma}=1$$**。同时，**最大化$$\frac{1}{||w||}$$和最小化$$\frac{1}{2}||w||^2$$是等价的**，因此式$$(1.3)$$可转化为下式：
$$
\begin{align*} &\min\limits_{w,b} \quad 

\frac{1}{2}{\|w\|}^2 \\ &s.t. \quad y_{i}\left(w \cdot x_{i}+b\right) -1\geqslant 0,\quad i=1,2,\dots,N \end{align*} \tag{1.4}
$$
式$$(1.4)$$满足**凸二次优化问题**的形式。

> 引出概念：支持向量、间隔
>
> * **支持向量**：
>
> 在线性可分情况下，**训练数据集的样本点中与分离超平面距离最近的样本点的实例**称为**支持向量**。【靠这些点来支撑这个超平面】，如下图$$H_1,H_2$$上的点就是支持向量。
>
> <img src="https://gitee.com/nekomoon404/blog-img/raw/master/img/26.png" style="zoom:50%;" />
>
> * **间隔边界**，**间隔**：
>
> $$H_1$$与$$H_2$$称为**间隔边界**，$$H_1$$与$$H_2$$的距离称为**间隔**，间隔依赖于分离超平面的法向量$$w$$，距离为$$\frac{2}{\|w\|}$$。
>
> 所以**在决定分离超平面时只有支持向量起作用。**这就是**这种分类模型叫支持向量机的原因**。



### 1.2 对偶问题

将式$$(1.4)$$作为我们要求解的**原问题**（primal problem），：
$$
\begin{align*} &\min\limits_{w,b} \quad 

\frac{1}{2}{\|w\|}^2 \\ &s.t. \quad 1-y_{i}\left(w \cdot x_{i}+b\right)\leqslant 0,\quad i=1,2,\dots,N \end{align*}  \tag{1.5}
$$
原问题有个不好的地方是，我们要求关于$$w,b$$的最值，但约束条件中也带有$$w,b$$，会造成求解的不便。因此就希望将**关于$$w,b$$的带约束问题转化成无约束问题**。

可使用**拉格朗日乘子法**，对于约束$$y_{i}\left(w \cdot x_{i}+b\right) -1\geqslant 0,\quad i=1,2,\dots,N$$，即共有$$N$$个约束不等式，对于每一个约束引入一个拉格朗日乘子$$\alpha_i \geqslant 0$$，构造**拉格朗日函数**：
$$
\begin{align*}
L(w,b,a) &= \frac{1}{2}{\|w\|}^2+\sum_{i=1}^N\alpha_i(1-y_{i}\left(w \cdot x_{i}+b\right)) \\
&=\frac{1}{2}\|w\|^2-\sum_{i=1}^{N}\alpha_iy_i(w\cdot x_i+b)+\sum_{i=1}^{N}\alpha_i

\end{align*} \tag{1.6}
$$
则关于$$w,b$$的无约束问题为：
$$
\begin{align*} &\min\limits_{w,b} \max_{\alpha} L(w,b,\alpha) \\ &s.t. \quad \alpha_i \geqslant 0,\quad i=1,2,\dots,N \end{align*}  \tag{1.7}
$$

> 可以较直观地证明式$$(1.7)$$与式$$(1.5)$$等价：
> $$
> \begin{cases}
> \text{如果}\quad 1-y_{i}\left(w \cdot x_{i}+b\right)> 0，则 \max\limits_{\alpha} L(w,b,\alpha)=\frac{1}{2}{\|w\|}^2 + \infty=\infty
> 
> \\\text{如果}\quad 1-y_{i}\left(w \cdot x_{i}+b\right)\leqslant 0，则\max\limits_{\alpha} L(w,b,\alpha)=\frac{1}{2}{\|w\|}^2 + 0=\frac{1}{2}{\|w\|}^2
> 
> \end{cases}
> $$
> 则有：
> $$
> \min\limits_{w,b} \max_{\alpha} L(w,b,\alpha)=\min\limits_{w,b}(\infty,\quad\frac{1}{2}{\|w\|}^2)=\min\limits_{w,b}\frac{1}{2}{\|w\|}^2
> $$
> 可见虽然式$$(1.7)$$中没有显式地表达关于$$w,b$$的约束，但在求解过程中已将不满足原问题约束的情况，即$$1-y_{i}\left(w \cdot x_{i}+b\right)> 0$$的情况丢掉，即原问题与关于$$w,b$$的无约束问题是等价的。

进一步地，由问题$$(1.7)$$得到其**对偶问题**（dual problem）：
$$
\begin{align*} & \max_{\alpha} \min\limits_{w,b}L(w,b,\alpha) \\ &s.t. \quad \alpha_i \geqslant 0,\quad i=1,2,\dots,N \end{align*}  \tag{1.8}
$$
先求$$\min\limits_{w,b}L(w,b,\alpha)$$的最优解$$w^*,b^*$$，令$$L$$关于$$w,b$$的梯度为0即可：
$$
\begin{array}{l}\nabla_{w} L(w, b, \alpha)=w-\displaystyle\sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}=0 \\ \nabla_{b} L(w, b, \alpha)=-\displaystyle\sum_{i=1}^{N} \alpha_{i} y_{i}=0\end{array}
$$
则有：
$$
w=\sum_{i=1}^{N}\alpha_iy_ix_i \\
\sum_{i=1}^{N}\alpha_iy_i=0
$$
代入拉格朗日函数中，得到：
$$
\begin{array}\\
\min\limits_{w,b}L(w,b,\alpha) &= \frac{1}{2}\displaystyle\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^{N}\alpha_iy_i(\sum_{j=1}^{N}\alpha_jy_jx_j\cdot x_i+b)+\sum_{i=1}^{N}\alpha_i \\
&=\frac{1}{2}\displaystyle\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^{N}\alpha_iy_ib+\sum_{i=1}^{N}\alpha_i \\
&=-\frac{1}{2}\displaystyle\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)+\sum_{i=1}^{N}\alpha_i
\end{array}
$$
因此对偶问题的目标函数变为：
$$
\max_{\alpha} \min\limits_{w,b}L(w,b,\alpha)=\max_{\alpha}-\frac{1}{2}\displaystyle\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)+\sum_{i=1}^{N}\alpha_i
$$
再将上式中的目标函数求极大转换为求极小，得到对偶问题的最终优化形式：
$$
\begin{aligned} \min _{\alpha} &\ \  \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \\ \text { s.t. } & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\ & \alpha_{i} \geqslant 0, \quad i=1,2, \cdots, N \end{aligned}  \tag{1.9}
$$

> 可见这里引入了内积$$(x_i,x_j)$$，方便后续引入核函数。



### 1.3 硬间隔SVM模型求解

上一小节用利用拉格朗日对偶性，由原始问题推导出了其对偶问题。硬间隔SVM模型属于凸二次优化问题，**对于凸二次优化问题，原问题和对偶问题是等价的**，即满足强对偶关系。

>引出概念：弱对偶关系，强对偶关系，KKT条件
>
>* **弱对偶关系**：$$\min\limits_{w,b} \max\limits_{\alpha} L  \geqslant\max\limits_{\alpha} \min\limits_{w,b}L$$    （凤尾 $$\geqslant $$ 鸡头，我最大的里面取最小  也比 你最小的里面取最大  还要大），**弱对偶关系是所有对偶问题都满足的性质**。
>* **强对偶关系**：$$\min\limits_{w,b} \max\limits_{\alpha} L  =\max\limits_{\alpha} \min\limits_{w,b}L$$
>
>* **KKT条件**：原问题和对偶问题等价（强对偶关系） $$\Leftrightarrow$$  最优解$$w,b,\alpha$$满足KKT条件
>
> （有前提条件：目标函数是凸函数，约束不等式是仿射函数，具体推导见《统计学习方法》附录C）
>
> 对于凸二次优化问题，是天生满足KKT条件的，因此可以通过求解KKT条件得到其原始问题的最优解。

设对于对偶最优化问题$$(1.9)$$对$$\alpha$$的解为$$\alpha^*=(\alpha_1^*,\alpha_2^*,...,\alpha_N^*)^T$$，可由$$\alpha^*$$得到原始最优化问题$$(1.5)$$对$$(w,b)$$的解$$w^*,b^*$$。由KKT条件可得：
$$
\begin{align*}
& \nabla_{w} L\left(w^{*}, b^{*}, \alpha^{*}\right)=w^{*}-\displaystyle\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} x_{i}=0 \\ 
& \nabla_{b} L\left(w^{*}, b^{*}, \alpha^{*}\right)=-\displaystyle\sum_{i=1}^{N} \alpha_{i}^{*} y_{i}=0 \\ 
& \alpha_{i}^{*}\left(y_{i}\left(w^{*} \cdot x_{i}+b^{*}\right)-1\right)=0, \quad i=1,2, \cdots, N \\
& y_{i}\left(w^{*} \cdot x_{i}+b^{*}\right)-1 \geqslant 0, \quad i=1,2, \cdots, N \\
& \alpha_{i}^{*} \geqslant 0, \quad i=1,2, \cdots, N\end{align*}  \tag{1.10}
$$

> KKT条件由三部分组成：梯度为0（上式中第1,2条），**对偶互补条件**（上式中第3条），可行条件，满足可行域（上式中第4,5条）

由梯度为0可得：
$$
w^{*}=\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} x_{i} \tag{1.11}
$$
其中至少有一个$$\alpha_j>0$$（反证法可得），对于这个$$j$$，代入对偶互补条件，有：
$$
y_{i}\left(w^{*} \cdot x_{i}+b^{*}\right)-1=0
$$
又$$y_i=\pm 1$$，则$$y_i^2=1$$，上式左右同乘$$y_i$$，再移项得：
$$
b^*=y_j-(\sum_{i=1}^{N}\alpha_i^*y_ix_i)*x_j \tag{1.12}
$$
因此硬间隔SVM的分离超平面为：
$$
\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} (x \cdot x_{i})+b^*=0 \tag{1.13}
$$
分类决策函数为：
$$
f(x)=\text{sign} \left(\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} (x \cdot x_{i})+b^* \right) \tag{1.14}
$$
可见分类决策函数只依赖于输入$$x$$和训练样本输入的内积，上式称为**线性可分支持向量机的对偶形式**。

由式$$(1.11)$$和$$(1.12)$$知，**原问题最优解$$w^*,b^*$$只依赖于$$a_i^*>0$$的样本点$$(x_i,y_i)$$，因此将对于训练数据中对应$$\alpha_i^*>0$$的实例点$$x_i$$称为支持向量**。

> 也就是说**间隔边界上的点满足$$a_i^*>0$$**，证明也很显然：
>
> 将$$a_i^*>0$$代入对偶互补条件：
> $$
> y_i(w^*\cdot x_i+b^*)-1=0
> $$
> 所以$$x_i$$肯定在间隔边界上，即是训练样本中离超平面最近的点。



## 2. 线性支持向量机与软间隔最大化



### 2.1 软间隔SVM的原问题

上一节讨论的线性可分支持向量机是针对线性可分的数据集的，但在现实问题中，训练数据集往往是线性不可分的，样本中会出现噪声或特异点（outlier）。

考虑线性不可分数据集中较简单的情况，即训练数据集中有一些特异点，将这些特异点除去后，剩下大部分的样本点组成的集合是线性可分的。

回顾式$$(1.5)$$，线性不可分意味着，某些样本点$$(x_i,y_i)$$不能满足函数间隔大于等于1的约束条件。**软间隔（Soft margin）**的思想就是允许SVM模型“犯一点点错误”，我们要选一个犯错最小的超平面。表现在数学表达式上，可以给式$$(1.5)$$的目标函数加上一个loss：
$$
\min\limits_{w,b} \frac{1}{2}{\|w\|}^2 + \text{loss}
$$
如何表示loss，有两种自然的想法：

* 用犯错误的点的个数表示loss，loss为0-1损失，$$\text{loss}=\sum\limits_{i=1}^N I(y_i(w^Tx_i+b)<1)$$

  令$$z_i=y_i(w^Tx_i+b)$$，则有：
  $$
  \text{loss}_i=\begin{cases}
  1,\quad z < 1 \\
  0, \quad z \geqslant 1
  \end{cases}
  $$
  但0-1损失函数不是连续可导的，采用这种方法并不好。

* 用犯错误的点离间隔边界的距离表示loss，被称为**合页损失函数**（hinge loss）
  $$
  \text{loss}_i=
  \begin{cases}
  1-y_i(w^Tx_i+b),\quad &y_i(w^Tx_i+b) < 1 \\
  0,\quad &y_i(w^Tx_i+b) \geqslant 1 
  \end{cases}
  $$
  用$$[1-y_i(w^Tx_i+b)]_+$$表示上式，则loss为：
  $$
  \text{loss}=\sum\limits_{i=1}^N [1-y_i(w^Tx_i+b)]_+  \tag{2.1}
  $$
  <img src="https://gitee.com/nekomoon404/blog-img/raw/master/img/微信图片_20210130181223.png" style="zoom:67%;" />

  再给loss乘上一个**惩罚参数**$$C$$，$$C$$越大对犯错误的惩罚越大。令$$\xi_i=1-y_i(w^Tx_i+b),\quad \xi_i \geqslant 0$$，$$\xi_i$$被称为**松弛变量**，则约束条件变为$$y_{i}\left(w \cdot x_{i}+b\right) \geqslant 1-\xi_{i}$$，软间隔SVM变成如下的凸二次规划问题（原问题）：
  $$
  \begin{array}{ll}\min\limits _{w, b, \xi} & \frac{1}{2}\|w\|^{2}+C \displaystyle\sum_{i=1}^{N} \xi_{i} \\ \text { s.t. } & y_{i}\left(w \cdot x_{i}+b\right) \geqslant 1-\xi_{i}, \quad i=1,2, \cdots, N \\ & \xi_{i} \geqslant 0, \quad i=1,2, \cdots, N\end{array}  \tag{2.2}
  $$

### 2.2 对偶问题

构建软间隔SVM的原问题$$(2.2)$$的拉格朗日函数：
$$
L(w, b, \xi, \alpha, \mu) \equiv \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i}-\sum_{i=1}^{N} \alpha_{i}\left(y_{i}\left(w_i \cdot x_{i}+b\right)-1+\xi_{i}\right)-\sum_{i=1}^{N} \mu_{i} \xi_{i} \tag{2.3}
$$
其中$$\alpha_i\geqslant 0,\mu_i\geqslant 0$$。

则原始问题变为拉格朗日函数的极小极大问题：$$\min\limits_{w,b,\xi,}\max\limits_{\alpha,\mu}L(w,b,\xi,\alpha,\mu)$$，对偶问题是拉格朗日函数的极大极小问题：$$\max\limits_{\alpha,\mu}\min\limits_{w,b,\xi,}L(w,b,\xi,\alpha,\mu)$$。

首先求$$\displaystyle\min_{w,b,\xi,}L(w,b,\xi,\alpha,\mu)$$：
$$
\begin{aligned} \nabla_{w} L(w, b, \xi, \alpha, \mu) &=w-\sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}=0 \\ \nabla_{b} L(w, b, \xi, \alpha, \mu) &=-\sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\ \nabla_{\xi_{i}} L(w, b, \xi, \alpha, \mu) &=C-\alpha_{i}-\mu_{i}=0 \end{aligned}
$$
得：
$$
\begin{array}{c}w=\displaystyle\sum_{i=1}^{N} \alpha_{i} y_{i} x_{i} \\ \displaystyle\sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\ C-\alpha_{i}-\mu_{i}=0\end{array}
$$
代入式$$(2.3)$$中，得：
$$
\min _{w, b, \xi} L(w, b, \xi, \alpha, \mu)=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i} \tag{2.4}
$$
再对上式求$$\alpha$$的极大，得到对偶问题：
$$
\begin{array}{l}\displaystyle\max _{\alpha}-\frac{1}{2} \displaystyle\sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i} \\ \text { s.t. } \displaystyle\sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\ C-\alpha_{i}-\mu_{i}=0 \\ \qquad \begin{array}{l}\alpha_{i} \geqslant 0 \\ \mu_{i} \geqslant 0, \quad i=1,2, \cdots, N\end{array}\end{array} \tag{2.5}
$$
利用等式约束消去$$\mu_i$$，从而只留下变量$$\alpha_i$$，上式中的后三条约束变为：
$$
0 \leqslant \alpha_{i} \leqslant C
$$
因此得到软间隔SVM的对偶问题：
$$
\begin{array}{ll}\displaystyle\min _{\alpha} & \frac{1}{2} \displaystyle\sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \\ \text { s.t. } & \displaystyle\sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\ & 0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \cdots, N\end{array} \tag{2.6}
$$

### 2.3 软间隔SVM模型求解

设$$\alpha^*=(\alpha_1^*,\alpha_2^*,...,\alpha_N^*)^T$$是对偶问题的一个解，若存在$$\alpha^*$$的一个分量$$\alpha_j^*,0<\alpha_j^*<C$$，则**原始问题的解**为：



原始问题是凸二次规划问题，满足KKT条件，因此有：
$$
\begin{align*}
&\nabla_{w} L(w^*, b^*, \xi^*, \alpha^*, \mu^*) =w^*-\sum_{i=1}^{N} \alpha_{i}^* y_{i}^* x_{i}^*=0  \tag{2.7}\\ 
&\nabla_{b} L(w^*, b^*, \xi^*, \alpha^*, \mu^*) =-\sum_{i=1}^{N} \alpha_{i}^* y_{i}^*=0 \\ 
&\nabla_{\xi_{i}} L(w^*, b^*, \xi^*, \alpha^*, \mu^*) =C-\alpha_{i}^*-\mu_{i}^*=0 \tag{2.8}\\
\\
&\alpha_{i}^*\left(y_{i}\left(w_i^* \cdot x_{i}+b^*\right)-1+\xi_{i}^*\right)=0 \tag{2.9}\\
&\mu^*\xi^*=0  \tag{2.10}\\
\\
&y_i(w^* \cdot x_i + b^*)-1+\xi_i^* \geqslant 0 \\
&\xi_i^* \geqslant 0 \\
&\alpha_i^* \geqslant 0 \\
& \mu_i^* \geqslant 0, \quad i=1,2,\dots,N

\end{align*}
$$
（梯度为0：第1-3条，对偶互补条件：第4-5条；可行域：第6-9条）

由式$$(2.7)$$得：
$$
w^{*}=\displaystyle\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} x_{i} \tag{2.11}
$$
若存在$$\alpha_j$$，$$0<\alpha_j<C$$，由式$$(2.8)$$知$$\mu_j \ne 0$$，由式$$(2.10)$$知$$\xi_j=0$$，代入式$$(2.9)$$得：$$y_j(w^* \cdot x_j + b^*)-1=0$$，又$$y_j^2=1$$，等式左右乘$$y_j$$得：
$$
\begin{align*}
b^*=&y_j-w^*\cdot x_j=y_j-\displaystyle\sum_{i=1}^{N} y_{i} \alpha_{i}^{*}\left(x_{i} \cdot x_{j}\right)
\end{align*} \tag{2.12}
$$
因此软间隔SVM的分离超平面可以写成：
$$
\sum_{i=1}^{N}\alpha_i^*y_i(x\cdot x_i)+b^*=0 \tag{2.13}
$$
分类决策函数为：
$$
f(x)=sign(\sum_{i=1}^{N}\alpha_i^*y_i(x\cdot x_i)+b^*) \tag{2.14}
$$

### 2.4 软间隔的支持向量

对偶问题的解$$\alpha^*=(\alpha_1^*,\alpha_2^*,...,\alpha_N^*)^T$$中对应于$$\alpha_i^*>0$$的样本点$$(x_i,y_i)$$的实例$$x_i$$称为支持向量（软间隔的支持向量）。由【KKT的对偶互补条件】可知：

$$
\alpha^*_i(y_i(w^*\cdot x_i+b^*)-1+\xi_i)=0,\ \ i=1,2,...,N
$$

且
$$
\begin{array}{l}C-\alpha_{i}-\mu_{i}=0 \\ \alpha_{i} \geq 0 \\ \mu_{i} \geq 0, i=1,2, \ldots, N\end{array}
$$

软间隔的支持向量$$x_i$$或者在间隔边界上，或者在间隔边界与分离超平面之间，或者在分离超平面误分一侧。

<img src="https://gitee.com/nekomoon404/blog-img/raw/master/img/微信图片_20210130185529.png" style="zoom:50%;" />

- $$\alpha_i^*<C,$$则$$\xi_i=0$$，支持向量$$x_i$$恰好落在间隔边界上 【不需要松弛】；
- $$\alpha_i^*=C,0<\xi_i<1$$，则分类正确，$$x_i$$在间隔边界与分离超平面之间；
- $$\alpha_i^*=C,\xi_i=1$$，则$$x_i$$在分离超平面上；
- $$\alpha_i^*=C,\xi_i>1$$，则分类错误，$$x_i$$位于分离超平面误分一侧；



## 3. 非线性支持向量机与核函数

对于非线性问题，不能用一个超平面对数据进行分割，需要非线性模型（超曲面）才能完成分类任务。求解非线性问题的常用方法是：进行一个非线性变换，通过解变换后的线性问题来求解原来的非线性问题。

<img src="https://gitee.com/nekomoon404/blog-img/raw/master/img/微信图片_20210130190354.png" style="zoom: 50%;" />

以上图为例，假设原空间为$$\mathcal{X} \subset \mathbf{R}^2,x=(x^{(1)},x^{(2)})^T \in \mathcal{X}$$，映射后的新空间为$$\mathcal{H} \subset \mathbf{R}^2,z=(z^{(1)},z^{(2)})^T \in \mathcal{Z}$$，定义映射函数：
$$
z=\phi(x)=((x^{(1)})^2,(x^{(2)})^2)^T
$$
则原空间的椭圆曲线：
$$
w_1(x^{(1)})^2+w_2(x^{(2)})^2+b=0
$$
变换为新空间下的直线：
$$
w_1z^{(1)}+w_2z^{(2)}+b=0
$$
在变换后的新空间里，上式直线可以将变换后的正负实例点正确分开。这样，原空间的新非线性可分问题变成了新空间下的线性可分问题。

**核技巧**的基本思想是：通过一个**映射函数$$\phi(x)$$**将输入空间$$\mathcal{X}$$下的样本点映射到特征空间$$\mathcal{H}$$下的样本点，使其能够用线性模型进行求解。

但是如何去寻找合适的映射函数$$\phi(x)$$是个问题，回顾上一节式$$(2.13)(2.14)$$，线性支持向量机的超平面和决策函数都只依赖于输入$$x$$和训练样本输入的内积$$(x\cdot x_i)$$，通过映射函数$$\phi(x)$$后映射为$$\phi(x) \cdot \phi(x_i)$$，如果我们能直接计算内积，就能使问题大大化简，这就引出了核函数。



### 3.1 核函数

**核函数**的定义：

设$$\mathcal{X}$$是输入空间（欧氏空间$$\mathbf{R}^n$$的子集或离散集合），又设$$\mathcal{H}$$为特征空间（希尔伯特空间），如果存在一个从$$\mathcal{X}$$到$$\mathcal{H}$$的映射：
$$
\phi(x):\mathcal{X} \rightarrow \mathcal{H}
$$
使得对所有$$x,z\in \mathcal{X}$$，函数$$K(x, z)$$满足条件：
$$
K(x, z) = \phi(x)\cdot\phi(z)
$$
则称$$K(x,z)$$为核函数，$$\phi(x)$$为映射函数，$$\phi(x)\cdot\phi(z)$$为内积。

**核技巧的想法是在学习和预测中只定义核函数$$K(x,z)$$，而不显示地定义映射函数$$\phi(x)$$**。

**好处**是：直接计算$$K(x,z)$$比较容易，$$\phi$$是输入空间$$\mathcal X$$到特征空间$$\mathcal H$$的映射，特征空间一般是高维的，甚至是无穷维。且对于给定核$$K(x,z)$$，特征空间$$\mathcal H$$和映射函数$$\phi$$的取法并不唯一。



### 3.2 核技巧在支持向量机中的应用

对于线性可分支持向量机的对偶问题，目标函数、决策函数都**只涉及输入实例与实例之间的内积**。因此内积可以用核函数代替目标函数为：
$$
\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}K\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}
$$
决策函数为：
$$
f(x)=sign(\sum_{i=1}^{N}\alpha_i^*y_i(\phi(x_i) \cdot \phi(x))+b^*)\\
=sign(\sum_{i=1}^{N}\alpha_i^*y_iK(x_i \cdot x)+b^*)
$$
在新的特征空间里训练线性支持向量机，当映射函数是非线性函数时，学习到的含有核函数的支持向量机就是**非线性模型**。

通常所说的核函数就是正定核函数（positive definite kernel function），那么**函数$$K(x,z)$$满足什么条件才能称为核函数**？

> 对称函数$$K(x,z)$$为正定核的充要条件如下：对任意$$x_i \in \mathcal{X},i=1,2,...,m$$，任意正整数$$m$$，对称函数$$K(x,z)$$对应的Gram矩阵$$K=[K(x_i,x_j)]_{m \times m}$$是半正定的。---->$$K_{ij} \geq 0 $$

**常用的核函数**有：

1. 多项式核函数（polynomial kernel function）
   $$
   K(x,z)=(x\cdot z+1)^p
   $$
   对应的支持向量机是一个$$p$$次多项式分类器。决策函数为：
   $$
   f(x)=sign(\sum_{i=1}^{N}\alpha_i^*y_i(x_i\cdot x+1)^p+b^*)
   $$

2. 高斯核函数（Gaussian kernel function）
   $$
   K(x,z)=exp(-\frac{\|x-z\|^2}{2\sigma^2})
   $$
   对应的支持向量机时高斯径向基函数分类器，决策函数为：
   $$
   f(x)=sign(\sum_{i=1}^{N}\alpha_i^*y_iexp(-\frac{\|x-z\|^2}{2\sigma^2})+b^*)
   $$

3. 字符串核函数（string kernel function）

   定义在离散的数据集合上



## 4. 序列最小最优化算法SMO

支持向量机的学习问题可以化为求解凸二次规划问题，具有全局最优解，但当训练样本容量很大时，如何高效地实现支持向量机学习？**序列最下最优化算法**（sequential minimal optimization, SMO）就是一种**快速实现算法**。

SVM的最终形式：解如下凸二次规划的对偶问题
$$
\begin{array}{ll}\min _{\alpha} & \frac{1}{2} \displaystyle\sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j} K\left(x_{i}, x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \\ \text { s.t. } & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\ & 0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \cdots, N\end{array}
$$
SMO的基本思想是将对偶问题转换为一系列子问题：

1. 如果所有变量的解都满足此最优化问题的KKT条件，那么最优化问题的解就得到了；

2. 否则，**选择两个变量**，固定其他变量，针对这两个变量构建一个二次规划问题。

   （因为有约束条件，如果只选一个变量，固定其他变量，那约束必然被打破，因此至少要选择两个变量，**两个变量中只有一个是自由变量**。）

3. 这个二次规划问题关于这两个变量的解应该更接近原始二次规划问题的解，因为这会使得原始二次规划问题的目标函数值变得更小。**子问题可以通过解析方法**求解，提高整个算法的计算速度。

**两个变量的二次规划求解方法**以及**如何选择两个变量**（一个是违反KKT条件最严重的那个，另一个由约束条件求得），具体推导见《统计学习方法》P143-149。



## 参考资料

[1] 李航.统计学习方法[M].北京: 清华大学出版社, 2019

【注】本篇笔记大部分是我一位朋友整理的，我加入了一些推导的理解。