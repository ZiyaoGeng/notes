# Word2vec

Word2vec是一个生成对“词”的向量表达的模型。

模型结构：

- Continuous Bag-of-Word（CBOW）；
- skip-gram；

优化技术：

- hierarchical softmax；
- negative sampling；



## 模型

- 如果是用一个词语作为输入，来预测它周围的上下文，那这个模型叫做『Skip-gram 模型』
- 而如果是拿一个词语的上下文作为输入，来预测这个词语本身，则是 『CBOW 模型』

<img src="https://mmbiz.qpic.cn/mmbiz_png/xicQMWhcBia0JNhic0TqqiaPoyu4gVM0m9j8jliaQBgeFSp1GUZtPvG2bESJ3kKE2I8lSL5rcU09jnuiaDub29RwnOBw/640?wx_fmt=png&amp;wxfrom=5&amp;wx_lazy=1&amp;wx_co=1" alt="图片" style="zoom:50%;" />

### CBOW

CBOW的简单模型【单个词，也可以看作只预测一个上下文词的Skip-gram】如下所示：

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210315132218881.png" alt="image-20210315132218881" style="zoom:50%;" />

其中$V$表示词典的大小，$N$表示隐藏单元的数量，输入是one-hot编码向量。对于【输入层和隐藏层之间的权重】$\mathbf{W}_{V\times N}$，的每一行都是输入层关联词的$n$维向量表示$\mathbf{v}_w$。假设$x_k=1$，因此有输入词$w_I$的向量表示：
$$
\mathbf{h}=\mathbf{W}^{T} \mathbf{x}=\mathbf{W}_{(k, \cdot)}^{T}:=\mathbf{v}_{w_{I}}^{T} \tag{1}
$$
从【隐藏层到输出层】，存在一个不同的权值矩阵$\mathbf{W}'_{N\times V}$。使用这些权重，可以计算词汇表中【每个单词的得分】$u_j$，
$$
u_{j}=\mathbf{v}_{w_{j}}^{\prime T} \mathbf{h} \tag{2}
$$
接下来，可以【使用softmax函数】，得到对于词$w_I$的后验分布：
$$
p\left(w_{j} \mid w_{I}\right)=y_{j}=\frac{\exp \left(u_{j}\right)}{\sum_{j^{\prime}=1}^{V} \exp \left(u_{j^{\prime}}\right)} \tag{3}
$$
带入（1）（2），得到：
$$
p\left(w_{j} \mid w_{I}\right)=\frac{\exp \left({\mathbf{v}_{w_{j}}^{\prime}}^ T{\mathbf{v}_{w_I}}\right)}{\sum_{j^{\prime}=1}^{V} \exp \left({\mathbf{v}_{w_{j'}}^{\prime}}^ T{\mathbf{v}_{w_I}}\right)}
$$
其中$\mathbf{v}_w$与$\mathbf{v}'_w$都是词$w_I$的表示，$\mathbf{v}_w$称为输入向量，$\mathbf{v}'_w$称为输出向量。



下图时带有多单词上下文设置的CBOW模型。在计算隐含层输出时，**CBOW模型不是直接复制输入上下文单词的输入向量，而是取输入上下文单词的向量的平均值**，并使用输入→隐权矩阵与平均值向量的乘积作为输出，即：
$$
\begin{aligned} \mathbf{h} &=\frac{1}{C} \mathbf{W}^{T}\left(\mathbf{x}_{1}+\mathbf{x}_{2}+\cdots+\mathbf{x}_{C}\right) \\ &=\frac{1}{C}\left(\mathbf{v}_{w_{1}}+\mathbf{v}_{w_{2}}+\cdots+\mathbf{v}_{w_{C}}\right)^{T} \end{aligned}
$$
其中$C$表示上下文的数量，$x_1,...,x_C$表示上下文的词，对应的$v_{w1},...,v_{wc}$表示输入向量。

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210315142306891.png" alt="image-20210315142306891" style="zoom: 50%;" />



### Skip-Gram

Skip-Gram输出的并不是单个多项式分布，而是C个多项式分布（预测C个词）

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210315143149784.png" alt="image-20210315143149784" style="zoom:50%;" />

损失函数【预测每个上下文词向量的概率相乘】：
$$
E=-\log \left(p\left(w_{1, j}, w_{2, j}, \ldots, w_{C, j} \mid w_{k}\right)\right)=-\log \prod_{c=i}^{C} \frac{\exp \left(v_{w_{c}}^{\prime}{ }^{T} v_{w_{k}}\right)}{\sum_{i=1}^{V} \exp \left(v_{w_{i}}^{\prime}{ }^{T} v_{w_{k}}\right)}
$$


## 优化方法

为了更新输出向量的参数，我们需要先计算误差，然后通过反向传播更新参数。在计算误差是我们需要遍历词向量的所有维度，这相当于遍历了一遍单词表，碰到大型语料库时计算代价非常昂贵。要解决这个问题，有三种方式：

- **Hierarchical Softmax**：通过 Hierarchical Softmax 将复杂度从 O(n) 降为 O(log n)；
- **Sub-Sampling Frequent Words**：通过采样函数一定概率过滤高频单词；
- **Negative Sampling**：直接通过采样的方式减少负样本。



### 层次Softmax

基于Hierarchical Softmax的CBOW模型的目标函数通常为以下对数似然函数：
$$
\mathcal{L}=\sum_{w \in \mathcal{C}} \log p(w \mid \operatorname{Context}(w))
$$
Skip-gram的优化目标函数为：
$$
\mathcal{L}=\sum_{w \in \mathcal{C}} \log p(\operatorname{Context}(w) \mid w)
$$
重点是**如何去构造上述两个目标函数**。

改变：

1. 去掉了隐藏层【加速训练】
2. 输出层改用了Huffman树；【复杂度变成了二叉树的高度，之前为词典中所有的词】



#### CBOW模型

网络结构如下所示：

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210315165901196.png" alt="image-20210315165901196" style="zoom: 33%;" />

- 输入层：包含$Context(w)$中$2c$个词向量【选择了$w$左右各$c$个词】，设$m$为词向量的长度；
- 投影层：输入层的$2c$个词向量，累加求和，$\mathbf{x}_{w}=\sum_{i=1}^{2 c} \mathbf{v}\left(\operatorname{Context}(w)_{i}\right) \in \mathbb{R}^{m}$；
- 输出层：对应一棵二叉树，**以语料库中出现过的词当叶结点**，**以各词在语料中出现的次数当权值**构造的Huffman树；叶子结点共$N=｜\mathcal{D}｜$个，那么非叶子结点有$N-1$个。对于每一个结点，存在一条唯一的路径，从根结点到当前叶结点，路径用于估计叶子结点所代表的单词的概率。

**问题：**如何通过向量$x_w$以及Huffman树来定义函数$p(w \mid \operatorname{Context}(w))$？

【将每一次分枝视作二分类】，将一个结点进行分类时，分到左边就是负类，分到右边就是正例。

定义非叶结点（黄色）的向量为$\theta^w_i$，对于二分类来说，一个结点被分为正类的概率：
$$
\sigma\left(\mathbf{x}_{w}^{\top} \theta\right)=\frac{1}{1+e^{-\mathbf{x}_{w}^{\top} \theta}}
$$
被分负类的概率：
$$
1-\sigma\left(\mathbf{x}_{w}^{\top} \theta\right)
$$
每个非叶结点就对应着学习参数$\theta$。以下图为例，

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210331212738777.png" alt="image-20210331212738777" style="zoom:50%;" />

从根结点出发，4次二分类的结果概率为：【左为负类】

- $$p\left(d_{2}^{w} \mid \mathbf{x}_{w}, \theta_{1}^{w}\right)=1-\sigma\left(\mathbf{x}_{w}^{\top} \theta_{1}^{w}\right)$$
- $p\left(d_{3}^{w} \mid \mathbf{x}_{w}, \theta_{2}^{w}\right)=\sigma\left(\mathbf{x}_{w}^{\top} \theta_{2}^{w}\right)$
- $p\left(d_{4}^{w} \mid \mathbf{x}_{w}, \theta_{3}^{w}\right)=\sigma\left(\mathbf{x}_{w}^{\top} \theta_{3}^{w}\right)$
- $p\left(d_{5}^{w} \mid \mathbf{x}_{w}, \theta_{4}^{w}\right)=1-\sigma\left(\mathbf{x}_{w}^{\top} \theta_{4}^{w}\right)$

因此，对于最终足球的概率的估计为：
$$
p(\text { 足球 } \mid \operatorname{Contex}(\text { 足球 }))=\prod_{j=2}^{5} p\left(d_{j}^{w} \mid \mathbf{x}_{w}, \theta_{j-1}^{w}\right)
$$
Hierarchical Softmax的**基本思想**：对于词典$\mathcal{D}$中的任意词$w$，Huffman树必定存在一条从根结点到词$w$的完整路径（唯一），存在的的每个分支看作一次二分类，每一次分类就产生一个概率，将概率乘起来，就是我们所需的$p(w \mid \operatorname{Context}(w))$。

一般公式写为：
$$
p(w \mid \operatorname{Context}(w))=\prod_{j=2}^{l^{w}} p\left(d_{j}^{w} \mid \mathbf{x}_{w}, \theta_{j-1}^{w}\right)
$$
其中：【规定左为负类，值为1】
$$
p\left(d_{j}^{w} \mid \mathbf{x}_{w}, \theta_{j-1}^{w}\right)=\left\{\begin{array}{ll}\sigma\left(\mathbf{x}_{w}^{\top} \theta_{j-1}^{w}\right), & d_{j}^{w}=0 \\ 1-\sigma\left(\mathbf{x}_{w}^{\top} \theta_{j-1}^{w}\right), & d_{j}^{w}=1\end{array}\right.
$$
整体形式：
$$
p\left(d_{j}^{w} \mid \mathbf{x}_{w}, \theta_{j-1}^{w}\right)=\left[\sigma\left(\mathbf{x}_{w}^{\top} \theta_{j-1}^{w}\right)\right]^{1-d_{j}^{w}} \cdot\left[1-\sigma\left(\mathbf{x}_{w}^{\top} \theta_{j-1}^{w}\right)\right]^{d_{j}^{w}}
$$
带入似然函数：
$$
\begin{aligned} \mathcal{L} &=\sum_{w \in \mathcal{C}} \log \prod_{j=2}^{l^{w}}\left\{\left[\sigma\left(\mathbf{x}_{w}^{\top} \theta_{j-1}^{w}\right)\right]^{1-d_{j}^{w}} \cdot\left[1-\sigma\left(\mathbf{x}_{w}^{\top} \theta_{j-1}^{w}\right)\right]^{d_{j}^{w}}\right\} \\ &=\sum_{w \in \mathcal{C}} \sum_{j=2}^{l^{w}}\left\{\left(1-d_{j}^{w}\right) \cdot \log \left[\sigma\left(\mathbf{x}_{w}^{\top} \theta_{j-1}^{w}\right)\right]+d_{j}^{w} \cdot \log \left[1-\sigma\left(\mathbf{x}_{w}^{\top} \theta_{j-1}^{w}\right)\right]\right\} \end{aligned}
$$
**梯度上升**（上升是强调求最大值）更新参数，伪代码：

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210331214418102.png" alt="image-20210331214418102" style="zoom:50%;" />

更新方法：每取一个样本，就对目标函数中的所有相关参数进行一次更新。

更新的参数为：$\mathbf{x}_{w}, \theta_{j-1}^{w}, w \in \mathcal{C},j=2,...,l^w$【对应位置】

【注】$x_w$是所有上下文向量的累加和，而我们需要更新的是上下文向量，word2vec的做法是通过将$x_w$的梯度贡献$e$到每一个上下文向量。



#### Skip-gram

对比CBOW：

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210331214639780.png" alt="image-20210331214639780" style="zoom:50%;" />

Skip-gram模型可以定义为：
$$
p(\operatorname{Context}(w) \mid w)=\prod_{u \in \operatorname{Context}(w)} p(u \mid w)
$$
对于每一个上下文词，参照CBOW的计算方式，有
$$
p(u \mid w)=\prod_{j=2}^{l^{u}} p\left(d_{j}^{u} \mid \mathbf{v}(w), \theta_{j-1}^{u}\right)
$$
其中：
$$
p\left(d_{j}^{u} \mid \mathbf{v}(w), \theta_{j-1}^{u}\right)=\left[\sigma\left(\mathbf{v}(w)^{\top} \theta_{j-1}^{u}\right)\right]^{1-d_{j}^{u}} \cdot\left[1-\sigma\left(\mathbf{v}(w)^{\top} \theta_{j-1}^{u}\right)\right]^{d_{j}^{u}}
$$
带回似然函数：
$$
\begin{aligned} \mathcal{L} &=\sum_{w \in \mathcal{C}} \log \prod_{u \in \text { Context }(w)} \prod_{j=2}^{l^{u}}\left\{\left[\sigma\left(\mathbf{v}(w)^{\top} \theta_{j-1}^{u}\right)\right]^{1-d_{j}^{u}} \cdot\left[1-\sigma\left(\mathbf{v}(w)^{\top} \theta_{j-1}^{u}\right)\right]^{d_{j}^{u}}\right\} \\ &=\sum_{w \in \mathcal{C}} \sum_{u \in \operatorname{Context}(w)} \sum_{j=2}^{l^{u}}\left\{\left(1-d_{j}^{u}\right) \cdot \log \left[\sigma\left(\mathbf{v}(w)^{\top} \theta_{j-1}^{u}\right)\right]+d_{j}^{u} \cdot \log \left[1-\sigma\left(\mathbf{v}(w)^{\top} \theta_{j-1}^{u}\right)\right]\right\} \end{aligned}
$$

更新的是中心词向量以及参数$\theta$。



### 负采样

负采样不再使用复杂的Huffman树，而是利用相对简单的随机负采样，大幅度提高性能，作为层级softmax的一种替代。

#### 负采样算法

问题：给定一个词$w$，如何生成$NEG(w)$？

词典$\mathcal{D}$在语料$C$中出现的次数有高有低，对于那些高频词，被选为负样本的概率应该比较大，对于低频词，被选为负样本的概率比较小。【带权采样问题】

具体做法：记$l_0=0,l_k=\sum_{j=1}^{k}len(w_j),k=1,...,N$，$I_{i}=\left(l_{i-1}, l_{i}\right], i=1,2, \cdots, N$为$N$个剖分区间。进一步引入一个$[0, 1]$的等分区间，剖分节点为$\left\{m_{j}\right\}_{j=0}^{M}$，其中$M>>N$。

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210401104943530.png" alt="image-20210401104943530" style="zoom:50%;" />

因此，可以建立$\left\{m_{j}\right\}_{j=0}^{M}$与区间$\left\{I_{j}\right\}_{j=1}^{N}$的关系
$$
\operatorname{Table}(i)=w_{k}, \quad \text { where } m_{i} \in I_{k}, \quad i=1,2, \cdots, M-1
$$
所以采样的方法为：每生成一个$[1, M-1]$间的随机整数$r$，$Table(r)$就是一个样本。

word2vec源码中采用了$\alpha$次幂，因此：
$$
\operatorname{len}(w)=\frac{[\operatorname{counter}(w)]^{0.75}}{\sum_{u \in \mathcal{D}}[\operatorname{counter}(u)]^{0.75}}
$$
代码中$M=10^8$.



### 源码细节

#### 词典存储

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210401110101916.png" alt="image-20210401110101916" style="zoom:50%;" />

#### 低频词与高频词的处理

低频词：

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210401110227493.png" alt="image-20210401110227493" style="zoom:50%;" />

高频词：

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210401110426479.png" alt="image-20210401110426479" style="zoom:50%;" />

自适应学习率：

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/image-20210401110740543.png" alt="image-20210401110740543" style="zoom:50%;" />



## 面试

### 1、Word2Vec中skip-gram是什么,Negative Sampling怎么做

Word2Vec通过【学习文本然后用词向量的方式表征词的语义信息】，然后【使得语义相似的单词在嵌入式空间中的距离很近】。

在Word2Vec模型中有【Skip-Gram】和【CBOW】两种模式，Skip-Gram是给定输入单词来预测上下文，而CBOW与之相反,是给定上下文来预测输入单词。

Negative Sampling是对于给定的词，并生成其负采样词集合的一种策略，已知有一个词，这个词可以看做一个正例，而它的上下文词集可以看做是负例，但是负例的样本太多。而在语料库中，各个词出现的频率是不一样的。所以在采样时可以要求高频词选中的概率较大，低频词选中的概率较小，这样就转化为一个带权采样问题，大幅度提高了模型的性能。



### 2、Word2vec的原理，层级softmax 怎么训练？



### 3、Word2vec的负采样

- 统计每个词出现对概率，丢弃词频过低对词
- 每次选择softmax的负样本的时候，从丢弃之后的词库里选择（选择是需要参考出现概率的）
- 负采样的核心思想是：利用负采样后的输出分布来模拟真实的输出分布

### 4、两种方式的优势

Skip-gram 在处理少量数据时效果很好，可以很好地表示低频单词。而 CBOW 的学习速度更快，对高频单词有更好的表示



### 5、Word2vec的负采样和分层softmax介绍，负采样的负样本是如何采样的，分层softmax的细节以及树的节点是什么