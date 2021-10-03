

# 集成学习

Boosting主要关注与降低偏差，Bagging主要关注与降低方差。

## Boosting

Boosting 训练过程为阶梯状，基模型的训练是有顺序的，每个基模型都会在前一个基模型学习的基础上进行学习，最终综合所有基模型的预测值产生最终的预测结果，用的比较多的综合方式为加权法。

- AdaBoost；
- GBDT；



## Bagging

Bagging是并行式集成学习方法，它直接基于自助采样法（bootstrap sampling）。

> 给定包含$m$个样本的数据集，先随机取出一个样本放入采样集中，再把该样本放入初始数据集。经过$m$次随机采样操作，得到$m$个样本采样集，初始训练集中约有63.2%的样本出现在采样集中
> $$
> \lim_{m \rightarrow \infty}(1-\frac{1}{m})^m=\frac{1}{e} \approx0.368
> $$

每个基学习器基于不同子训练集进行训练，并综合所有基学习器的预测值得到最终的预测结果。Bagging 常用的综合方法是投票法，票数最多的类别为预测类别。



## Bagging与Boosting的区别

1. 样本选择：Bagging的训练集是在原始集中有放回的选取，而Boosting每轮的训练集不变，只是训练集中的每个样本在分类器中的权重都会发生变化，此权值会根据上一轮的结果进行调整。
2. 样例权重：：Bagging：使用均匀取样，每个样例的权重相等。Boosting：根据错误率不断调整样例的权值，错误率越大则权重越大。
3. 预测函数：bagging：所有预测函数的权重相等。Boosting：每个弱分类器都有相应的权重，对于分类误差小的分类器会有更大的权重。
4. 并行计算：Bagging的各个预测函数可以并行生成，Boosting的各预测函数只能顺序生成。
5. 模型强度：Bagging中整体模型的期望近似于基模型的期望，所以整体模型的偏差相似于基模型的偏差，因此Bagging中的基模型为强模型（强模型拥有低偏差高方差）。Boosting中的基模型为弱模型，若不是弱模型会导致整体模型的方差很大。



# AdaBoost

对于Boosting方法，有两个问题：

1. 每一轮如何改变训练数据的权值或概率分布；；
2. 弱分类器如何组成一个强分类器；

**AdaBoost的做法是**：

（1）提高那些被前一轮弱分类器**错误分类样本的权值**，降低被正确分类样本的权值。

（2）加权多数表决，即**加大分类误差率小的弱分类器的权值**，使其在表决中起较大的作用；减小分类误差率大的弱分类器的权值，使其在表决中起较小的作用。



## 算法描述

AdaBoost的算法描述具体如下：

输入：训练数据集$T={(x_1,y_1),(x_2, y_2),...,(x_N,y_N)}$，其中$x_i \in \mathcal{X} \subseteq \mathbf{R}^n, y_i \in \mathcal{Y}=\{=1,+1\}$；

输出：最终分类器$G(x)$。

（1）初始化训练数据的权值分布（**平均**）：
$$
D_{1}=\left(w_{11}, \cdots, w_{1 i}, \cdots, w_{1 N}\right), \quad w_{1 i}=\frac{1}{N}, \quad i=1,2, \cdots, N
$$
（2）对$m=1,2,...,M$（弱学习器的数量）

​		（a）使用具有权值分布的$D_m$训练集学习，得到基本分类器【决策树等】
$$
G_m(x):\mathcal{X}\rightarrow\{-1, +1\}
$$


​		（b）计算$G_m(x)$（弱学习器）在训练集上的分类误差率：
$$
e_{m}=\sum_{i=1}^{N} P\left(G_{m}\left(x_{i}\right) \neq y_{i}\right)=\sum_{i=1}^{N} w_{m i} I\left(G_{m}\left(x_{i}\right) \neq y_{i}\right)
$$
​		【注】$e_m$为当前弱学习器误分类率，$w_{mi}$即第$m$轮中第$i$个样本的权值，$\displaystyle\sum_{i=1}^Nw_{mi}=1$。

​		（c）计算$G_m(x)$的系数
$$
\alpha_m=\frac{1}{2}log\frac{1-e_m}{e_m}
$$
​		【注】$\alpha$表示弱学习器$G_m(x)$在最终分类器的重要性，且当$e_m \leq \frac{1}{2}$时，$\alpha_m$会随着$e_m$的减小而增大（即误分率越低的弱学习器权值越大，越能起到较大的作用）。log为自然对数

​		（d）更新训练数据集的权值分布
$$
\begin{array}{c}D_{m+1}=\left(w_{m+1,1}, \cdots, w_{m+1, i}, \cdots, w_{m+1, N}\right) \\ w_{m+1, i}=\frac{w_{m i}}{Z_{m}} \exp \left(-\alpha_{m} y_{i} G_{m}\left(x_{i}\right)\right), \quad i=1,2, \cdots, N\end{array}
$$
​		其中$Z_m$为规范化因子，**使得$D_{m+1}$为一个概率分布**：
$$
Z_{m}=\sum_{i=1}^{N} w_{m i} \exp \left(-\alpha_{m} y_{i} G_{m}\left(x_{i}\right)\right)
$$
​		【注】关于$w_{m+1,i}$的计算可以写成：
$$
w_{m+1, i}=\left\{\begin{array}{ll}\frac{w_{m i}}{Z_{m}} \mathrm{e}^{-\alpha_{m}}, & G_{m}\left(x_{i}\right)=y_{i} \\ \frac{w_{m i}}{Z_{m}} \mathrm{e}^{\alpha_{m}}, & G_{m}\left(x_{i}\right) \neq y_{i}\end{array}\right.
$$
​		由此我们可以看到**被基学习器$G_m(x)$误分样本的权值得到增大，而被正确分类样本的权值得以缩小**。

（3）构建基本的线性组合
$$
f(x)=\sum_{m=1}^M\alpha_mG_m(x)
$$
【注】$\alpha_m$之和并不为1。

得到最终分类器
$$
G(x)=sign(f(x))=sign(\sum_{m=1}^M\alpha_mG_m(x))
$$



## 加法模型

AdaBoost算法是模型为加法模型，**损失函数为指数函数**，学习算法为前向分布算法的二分类学习方法。

#### 前向分布算法

加法模型：
$$
f(x)=\sum_{m=1}^M\beta_mb(x;\gamma_m)
$$
其中$b(x;\gamma_m)$为基函数，$\gamma_m$为基函数的参数，$\beta_m$为基函数的系数。**因此对于AdaBoost最终的线性组合也是一个加法模型。**

给定损失函数$L(y,f(x))$，学习加法模型，最小化损失函数：
$$
\min _{\beta_{m}, \gamma_{m}} \sum_{i=1}^{N} L\left(y_{i}, \sum_{m=1}^{M} \beta_{m} b\left(x_{i} ; \gamma_{m}\right)\right)
$$
这是一个**复杂的优化问题**。**前向分布算法求解这一优化问题的想法是：因为学习的是加法模型，如果能从前向后，每一个只学习一个基函数及其系数，逐步逼近优化目标函数，那么就可以简化优化复杂度。**每步只需优化如下损失：
$$
\min _{\beta, \gamma} \sum_{i=1}^{N} L\left(y_{i}, \beta b\left(x_{i} ; \gamma\right)\right)
$$

前向分布算法学习的是加法模型，当基函数为基本分类器时，该加法模型等价于AdaBoost的最终分类器
$$
f(x)=\sum_{m=1}^M\alpha_mG_m(x)
$$
损失函数为：
$$
L(y,f(x))=exp[-yf(x)]
$$


**（1）$\alpha_m=\frac{1}{2}log\frac{1-em}{e_m}$是如何得到的？**

假设经过$m-1$轮迭代前向分布算法得到$f_{m-1}(x)$：
$$
f_{m-1}(x)=f_{m-2}(x)+\alpha_{m-1}G_{m-1}(x)\\
=\alpha_1G_1(x)+...+\alpha_{m-1}G_{m-1}(x)
$$
在第$m$轮迭代得到$\alpha_m,G_m(x),f_m(x)$
$$
f_m(x)=f_{m-1}(x)+\alpha_mG_m(x)
$$
目标是使前向分布算法得到的$\alpha_m,G_m(x)$使$f_m(x)$在训练集上的**指数损失最小**：
$$
\left(\alpha_{m}, G_{m}(x)\right)=\arg \min _{\alpha, G} \sum_{i=1}^{N} \exp \left[-y_{i}\left(f_{m-1}\left(x_{i}\right)+\alpha G\left(x_{i}\right)\right)\right]
$$
可以表示为：
$$
\left(\alpha_{m}, G_{m}(x)\right)=\arg \min _{\alpha, G} \sum_{i=1}^N \bar{w}_{m i} \exp \left[-y_{i} \alpha G\left(x_{i}\right)\right]
$$
其中$\bar{w}_{m i}=\exp \left[-y_{i} f_{m-1}\left(x_{i}\right)\right]$。$\bar w_{mi}$与$\alpha,G$无关，所以与最小化无关。但依赖于$f_{m-1}(x)$，随着每一轮迭代而发生改变。

对于使上式最小化的最优基学习器为$G^*_m(x)$，由下式得到：
$$
G^*_m(x)=arg\ \min_G\sum_{i=1}^N\bar w_{mi}I(y_i \neq G(x_i))
$$
关于$\alpha^*_m$：
$$
\begin{aligned} \sum_{i=1}^{N} \bar{w}_{m i} \exp \left[-y_{i} \alpha G\left(x_{i}\right)\right] &=\sum_{y_{i}=G_{m}\left(x_{i}\right)} \bar{w}_{m i} \mathrm{e}^{-\alpha}+\sum_{y_{i} \neq G_{m}\left(x_{i}\right)} \bar{w}_{m i} \mathrm{e}^{\alpha} \\ &=\left(\mathrm{e}^{\alpha}-\mathrm{e}^{-\alpha}\right) \sum_{i=1}^{N} \bar{w}_{m i} I\left(y_{i} \neq G\left(x_{i}\right)\right)+\mathrm{e}^{-\alpha} \sum_{i=1}^{N} \bar{w}_{m i} \end{aligned}
$$
对$\alpha$求导并使导数为0，得到最小的$\alpha$：
$$
\alpha^*_m=\frac{1}{2}log\frac{1-e_m}{e_m}
$$
其中$e_m$为分类误差率：
$$
\begin{aligned} e_{m}=& \frac{\displaystyle\sum_{i=1}^{N} \bar{w}_{m i} I\left(y_{i} \neq G_{m}\left(x_{i}\right)\right)}{\sum_{i=1}^{N} \bar{w}_{m i}} \\=& \sum^{N}_{i=1} w_{m i} I\left(y_{i} \neq G_{m}\left(x_{i}\right)\right) \end{aligned}
$$
**（2）权值$w_{mi}$如何得到？**
$$
f_m(x)=f_{m-1}(x)+\alpha_mG_m(x)
$$
且$$\bar w_{mi}=exp[-y_if_{m-1}(x_i)]$$，可得
$$
\bar w_{m+1,i}=exp[-y_i(f_{m-1}(x)+\alpha_mG_m(x)]\\
=\bar w_{m,i}exp[-y_i\alpha_mG_m(x)]
$$


![image-20210313133536166](https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210313133536166.png)

