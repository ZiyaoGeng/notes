# XGBoost

学习大纲：

1. XGBoost的推导过程（建议看原论文）；
2. XGBoost的一些优化方法；



XGBoost是一个可扩展的提升树模型，论文“XGBoost: A Scalable Tree Boosting System”发表在2016年的KDD会议上。文章包括了XGBoost的原理以及对其的优化。



## XGBoost原理

XGBoost最关键的思想就是采用二阶导来近似取代一般损失函数。整个推导过程分为以下几个步骤（问题）：

1. 目标函数的构造；
2. 目标函数难以优化，如何近似？
3. 将树的结构（参数化，因为模型的学习参数在树中）融入到目标函数；
4. 如何构造最优（局部）二叉树？采用贪心算法；



### 目标函数定义

首先我们假设一个数据集中包含$$n$$个样本以及每个样本有$$m$$个特征，因此数据集可以定义为：
$$
\mathcal{D}=\left\{\left(\mathbf{x}_{i}, y_{i}\right)\right\}\left(|\mathcal{D}|=n, \mathbf{x}_{i} \in \mathbb{R}^{m}, y_{i} \in \mathbb{R}\right)
$$
对于提升树模型来说，我们假设共有$$K$$个叠加函数（additive functions，即决策树），那么整个模型可以表示为：
$$
\hat{y}_{i}=\phi\left(\mathbf{x}_{i}\right)=\sum_{k=1}^{K} f_{k}\left(\mathbf{x}_{i}\right), \quad f_{k} \in \mathcal{F}
$$
其中：

- $$\hat{y}_i$$：表示模型对样本$$x_i$$的预测值；
- $$\phi(\cdot)$$：模型函数；
- $$x_i$$：表示单个样本；
- $$f_k(\cdot)$$：表示第$$k$$决策树；
- $$\mathcal{F}$$；表示决策树的空间集合；

我们要**学习上述集成模型函数（也称加法模型），则需要最小化正则化后的损失函数**（即目标函数，正则化是对复杂决策树的惩罚）：
$$
min \ \mathcal{L}(\phi)=\sum_{i} l\left(\hat{y}_{i}, y_{i}\right)+\sum_{k} \Omega\left(f_{k}\right)
$$
$$\Omega(f_k)$$表示第$$k$$个决策树的复杂度，这里我们先不去参数化$$f_k(\cdot)$$和$$\Omega(\cdot)$$。



### 梯度提升算法

**问题1：**对于上述的目标函数，其实很难在欧氏空间中使用传统的优化方法。

因此，提升树模型采用【前向分步的学习方法】。假设$$\hat{y}_i^{(t)}$$表示在第$$t$$次迭代时对第$$i$$个样本的预测，那么我们可以将目标函数转化为以下形式（这里假设你已掌握提升树算法的知识）：
$$
\mathcal{L}^{(t)}=\sum_{i=1} l\left(y_{i}, \hat{y}_{i}^{(t-1)}+f_{t}\left(\mathbf{x}_{i}\right)\right)+\Omega\left(f_{t}\right)
$$
其中，$$\hat{y}_i^{(t-1)}$$表示第$$t-1$$次迭代时，集成模型对样本的预测，$$f_t(\cdot)$$表示第$$t$$个决策树。$$\hat{y}_i^{(t-1)}$$为常数，所以我们只需要学习$$f_t(x_i)$$，当然对于决策树复杂度的刻画，前$$t-1$$的决策树的复杂度此时为常数，所以目标函数并没有包括。

**问题2：**此时，目标函数变得更为简单，但是仍然无法描述损失函数$$l$$，因为并不知道其类型，一般的复杂函数也很难刻画。

所以，我们需要一个近似函数来表示损失函数。论文中提到，采用在一般设定下，二阶近似法（Second-order approximation）可用于快速优化目标。论文这里引用了一篇参考文献，其实就是通过泰勒公式用函数在某点的信息描述其附近取值。

首先可以了解微分（微分是函数在一点附近的最佳线性近似）来近似表示一般函数【从几何的角度讲，就是在某点的切线来近似于原函数附近的曲线，二阶近似则是采用抛物线】：
$$
f(x+\Delta x)=f(x)+f^{\prime}(x)\Delta x+o(\Delta x)
$$
$$o(\Delta x)$$为高阶无穷小，所以：
$$
f(x+\Delta x) \approx f(x)+f^{\prime}(x)\Delta x
$$
所以$$f(x)$$附近的值可以用上述线性近似来表示，**而GBDT就是采用线性近似（微分）来描述一般的损失函数。**对于XGBoost来说，采用二阶近似（二阶泰勒公式）来描述损失函数：
$$
f(x+\Delta x)=f(x)+f^{\prime}(x)\Delta x+ \frac{1}{2}f''(x)\Delta x^2+o(\Delta x)
$$
那么如何应用在我们定义的损失函数？其实$$ \hat{y}_{i}^{(t-1)}$$可以对应于$$x$$，$$f_t(x_i)$$对应于$$\Delta x$$，所以可以得到近似的损失函数：
$$
\mathcal{L}^{(t)} \simeq \sum_{i=1}^{n}\left[l\left(y_{i}, \hat{y}^{(t-1)}\right)+\partial_{\hat{y}(t-1)} l\left(y_{i}, \hat{y}^{(t-1)}\right) f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2}\partial_{\hat{y}^{(t-1)}}^{2} l\left(y_{i}, \hat{y}^{(t-1)}\right)f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{t}\right)
$$
我们令$$g_{i}=\partial_{\hat{y}^{(t-1)}} l\left(y_{i}, \hat{y}^{(t-1)}\right)，h_{i}=\partial_{\hat{y}^{(t-1)}}^{2} l\left(y_{i}, \hat{y}^{(t-1)}\right)$$，因此近似目标函数化简为
$$
\mathcal{L}^{(t)} \simeq \sum_{i=1}^{n}\left[l\left(y_{i}, \hat{y}^{(t-1)}\right)+g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{t}\right)
$$
并且$$l\left(y_{i}, \hat{y}^{(t-1)}\right)$$为一个常数，所以可以去掉：
$$
\tilde{\mathcal{L}}^{(t)}=\sum_{i=1}^{n}\left[g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{t}\right)
$$
【注】：一阶导$$g_i$$和二阶导$$h_i$$是已知的，因此需要学习的参数包含在$$f_t(x_i)$$中。

**问题3**：如何参数化【将需要学习的参数在目标函数中显示】表示$$f_t(x_i)$$和决策树的复杂度$$\Omega\left(f_{t}\right)$$？

对于$$f(x)$$，文章给出定义：
$$
f(\mathbf{x})=w_{q(\mathbf{x})}
$$
其中$$q(x)$$表示表示单个树的结构，即样本对应的叶子结点的索引。$$w$$表示叶子结点的值。【某个样本的预测值对应于某个叶子结点的值，这样就能描述清楚决策树上对应每个样本的预测值】。但是下标依旧带有参数，所以最终作者用：
$$
I_{j}=\left\{i \mid q\left(\mathbf{x}_{i}\right)=j\right\}
$$
表示**决策树中分配到第$$j$$个叶子结点中的样本集合**，这样，所有的叶子结点集合就能描述整个决策树。

而决策树的复杂度与叶子结点的数量和叶子结点的值（学习参数）相关，所以一般表示为：
$$
\Omega(f)=\gamma T+\frac{1}{2} \lambda\|w\|^{2}
$$
所以，我们将目标函数表示为：
$$
\begin{aligned} \tilde{\mathcal{L}}^{(t)} &=\sum_{i=1}^{n}\left[g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\gamma T+\frac{1}{2} \lambda \sum_{j=1}^{T} w_{j}^{2} \\ &=\sum_{j=1}^{T}\left[\left(\sum_{i \in I_{j}} g_{i}\right) w_{j}+\frac{1}{2}\left(\sum_{i \in I_{j}} h_{i}+\lambda\right) w_{j}^{2}\right]+\gamma T \end{aligned}
$$
可以看到，这其实是关于$$w_j$$的一个二次函数，我们可以使用中学的知识，最优解$$w^*_j$$为$$-\frac{b}{2a}$$，即：
$$
w_{j}^{*}=-\frac{\sum_{i \in I_{j}} g_{i}}{\sum_{i \in I_{j}} h_{i}+\lambda}
$$
计算对应的目标函数为：
$$
\tilde{\mathcal{L}}^{(t)}(q)=-\frac{1}{2} \sum_{j=1}^{T} \frac{\left(\sum_{i \in I_{j}} g_{i}\right)^{2}}{\sum_{i \in I_{j}} h_{i}+\lambda}+\gamma T
$$
以上便是XGBoost的梯度提升算法的推导过程。



### 寻找分割点

上述近似目标函数可以作为一个决策树$$q$$的评价分数，但是我们之前对于最小化目标函数来优化决策树的值$$w$$是**假定决策树的结构是已知的**。简单来说，就是目前我们可以做到给定一个决策树的结构，我们可以将它的叶子结点的值进行优化，并且可以达到一个评价它性能的分数$$\tilde{\mathcal{L}}^{(t)}(q)$$。

**问题4**：如何构造决策树？

最简单的方法是将所有决策树的结构全部罗列出来，然后优化，计算他们的目标函数值，进行比较，选择最小目标函数值的决策树。但是如果决策树深度过深，那么所有决策树的数量太多，很难完成上述步骤。

所以，文章采用了一种贪心算法（greedy algorithm）的策略，从单个叶子结点开始，迭代的增加分支进行比较。

假设，当前有样本$$\{1,2,3,4,5,6\}$$，目前决策树为：

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/1.png" alt="1" style="zoom:50%;" />

其目标函数值可以表示为：
$$
obj^*_{old}=-\frac{1}{2}[\frac{(g_1+g_2)^2}{h_1+h_2+\lambda}+\frac{(g_3+g_4+g_5+g_6)^2}{h_3+h_4+h_5+h_6+\lambda}]+2\cdot \gamma
$$
我们对右下叶子结点增加新的分支：

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/2.png" alt="2" style="zoom:50%;" />

此时目标函数值为：
$$
obj^*_{new}=-\frac{1}{2}[\frac{(g_1+g_2)^2}{h_1+h_2+\lambda}+\frac{(g_3+g_4)^2}{h_3+h_4+\lambda}]+\frac{(g_5+g_6)^2}{h_5+h_6+\lambda}]+3\cdot \gamma
$$
我们将两者相减，得到：
$$
obj^*_{old}-obj^*_{new}=\frac{1}{2}[\frac{(g_3+g_4)^2}{h_3+h_4+\lambda}]+\frac{(g_5+g_6)^2}{h_5+h_6+\lambda} -\frac{(g_3+g_4+g_5+g_6)^2}{h_3+h_4+h_5+h_6+\lambda}]- \gamma
$$
如果$$obj^*_{old}-obj^*_{new}>0$$，则说明可以进行分支，所以我们可以推导出，
$$
\mathcal{L}_{s p l i t}=\frac{1}{2}\left[\frac{\left(\sum_{i \in I_{L}} g_{i}\right)^{2}}{\sum_{i \in I_{L}} h_{i}+\lambda}+\frac{\left(\sum_{i \in I_{R}} g_{i}\right)^{2}}{\sum_{i \in I_{R}} h_{i}+\lambda}-\frac{\left(\sum_{i \in I} g_{i}\right)^{2}}{\sum_{i \in I} h_{i}+\lambda}\right]-\gamma
$$
其中$$L$$与$$R$$指代分割叶子结点后的新的左右叶子结点。通过以上分割方法，就可以分步的找到基于贪心的局部最优决策树。

贪心算法寻找分割点的算法步骤如下【将所有特征的所有可能的分割点给罗列一遍】：

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/3.png" alt="3" style="zoom:50%;" />

解释：

- $$I$$：当前结点的样本；
- $$d$$：特征维度；
- $$G,H$$：$$G$$表示当前叶结点的1阶导，$$H$$表示当前叶结点的2阶导；
- $$k,m$$：$$k$$表示样本的第$$k$$个特征，$$m$$为特征数量；
- 首先需要初始化分割左右的一阶导、二阶导为0；
- **由于为连续的特征列举出所有可能的分割在计算上是很困难的**。为了有效地做到这一点，算法必须首先根据特征值对**数据进行排序**，并按照排序后的数据进行访问。这里将需要分割的叶结点的样本按照$$k$$维进行排序，按照分割点（首先是前两个的均值），先将小的值的样本分到左边叶结点，其他在右边叶结点，按照公式计算$$G_L,H_L,G_R,H_R$$，得到评估当前决策树的分数；【然后递归，选择下一个分割点】
- 最终选择最大的分数的特征$$k$$，以及左右叶结点的样本分布情况；



## 优化

### Shrinkage and Column Subsampling

XGBoost除了在损失函数后面添加正则项用于防止过拟合外，还引入了`Shrinkage and Column Subsampling`两个trick。`Shrinkage`（缩放）会在XGBoost每一步迭代求解决策树的时候将新加入的`W`通过一个因子`η`进行缩放，即乘以$$\eta$$，即
$$
\hat{y}_{i}^{t}=\hat{y}_{i}^{t-1}+\eta f_{t}\left(x_{i}\right)
$$
与随机优化中的`learning rate`类似，对于提升模型的新增树（future trees），使用这种方法就是不在每一步进行充分优化，而是保留未来继续优化的可能性，达到防止过拟合的目的。

第二个技术是列特征子抽样，类似于随机森林的处理思路，生成树模型的时候通过抽样部分特征来进行实现，在防止过拟合的同时还可以减少训练时间。

### Weighted Quantile Sketch

算法1将所有的特征列的所有可能的分割点给罗列了一遍，在小数据集下当然可行，但是当数据不能完全装入内存时，就不可能有效地这样做。文章中提出了一种近拟处理算法，引入了percentiles(百分比分位数）的概念，也可以理解为“分桶”的思路。在原来Greedy算法时间复杂度的重要影响因素即**特征的取值范围较广时**，直接将s缩减至特定的百分比区间（例如10个），而不是精确取每一个数值进行切分，复杂度将大大关系少。因此，便提出了近似算法：根据特征分布的百分位数，提出$$n$$个候选切分点，然后将样本映射到对应的两个相邻的切分点组成的桶中，整合统计值，计算最佳分割点。该算法有两种形式：**全局近似**和**局部近似**，其差别是全局近似是在生成一棵树之前，对各个特征计算其分位点并划分样本；局部近似是在每个节点进行分裂时采用percentiles划分。

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/4.png" alt="4" style="zoom:50%;" />

- $$S_k$$表示特征$$k$$的分位点的集合，共有$$l$$个分位点；
- 计算的$$G_{kv},H_{kv}$$是在一个区间内的分类样本点的一阶导、二阶导；

但是针对**具体不同类型的特征值**时，具体怎么执行划分percentiles操作，是需要引入Weight Quantile Sketch的算法。 XGBoost 不是简单地按照样本个数进行分位，而是以二阶导数值 ![[公式]](https://www.zhihu.com/equation?tex=h_i+) 作为样本的权重进行划分。
$$
\begin{aligned} \mathcal{L}^{(t)} & \simeq \sum_{i=1}^{n}\left[l\left(y_{i}, \hat{y}_{i}^{(t-1)}\right)+g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{t}\right) \\ &=\sum_{i=1}^{n}\left[\frac{1}{2} h_{i} \cdot \frac{2 \cdot g_{i} f_{t}\left(\mathbf{x}_{i}\right)}{h_{i}}+\frac{1}{2} h_{i} \cdot f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{t}\right) \\ &=\sum_{i=1}^{n} \frac{1}{2} h_{i}\left[2 \cdot \frac{g_{i}}{h_{i}} \cdot f_{t}\left(\mathbf{x}_{i}\right)+f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{t}\right) \\ &=\sum_{i=1}^{n} \frac{1}{2} h_{i}\left[\left(2 \cdot \frac{g_{i}}{h_{i}} \cdot f_{t}\left(\mathbf{x}_{i}\right)+f_{t}^{2}\left(\mathbf{x}_{i}\right)+\left(\frac{g_{i}}{h_{i}}\right)^{2}\right)-\left(\frac{g_{i}}{h_{i}}\right)^{2}\right]+\Omega\left(f_{t}\right) \\ &=\sum_{i=1}^{n} \frac{1}{2} h_{i}\left[\left(f_{t}\left(\mathbf{x}_{i}\right)+\frac{g_{i}}{h_{i}}\right)^{2}\right]+\Omega\left(f_{t}\right) \\ &=\sum_{i=1}^{n} \frac{1}{2} h_{i}\left(f_{t}\left(\mathbf{x}_{i}\right)-\left(-g_{i} / h_{i}\right)\right)^{2}+\Omega\left(f_{t}\right) \end{aligned}
$$
第$$t$$颗树的损失函数可以视为去拟合标签$$-g_i/h_i$$ 的权重为$$h_i$$的平方损失函数。因此，定义一个数据集$$\mathcal{D}_{k}=\left\{\left(\mathbf{x}_{i k}, h_{i}\right)\right\}$$，其为第$$k$$个特征及其对应的二阶梯度，定义一个函数$$r_k(z)$$ 为：
$$
r_{k}(z)=\frac{1}{\sum_{(x, h) \in \mathcal{D}_{k}} h} \sum_{(x, h) \in \mathcal{D}_{k}, x<z} h
$$
通过给定一个超参数$$\epsilon$$ ，同时寻找一个相对准确的【候选切分点】 $$\left\{s_{k 1}, s_{k 2}, \ldots, s_{k l}\right\}$$ ，即通过不等式即可选出合适的切分点。【桶之间间距不能太大】
$$
\left|r_{k}\left(s_{k, j}\right)-r_{k}\left(s_{k, j+1}\right)\right|<\epsilon, \quad s_{k 1}=\min _{i} \mathbf{x}_{i k}, s_{k l}=\max _{i} \mathbf{x}_{i k}
$$
其直观理解为，划分为“桶”与“桶”之间**通过损失函数贡献进行等分切分**[正确的理解】，【最终选出合理的$$1/\epsilon$$个分裂点】。

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/7.png" alt="7" style="zoom:50%;" />

### Sparsit-aware Split Finding

在实际模型的训练过程中，数据往往会出现**比较稀疏**的情况，而且还会**有空值**的情况。XGBoost算法提出的**Sparsity-aware分割方法就是让算法根据当前特征把空值划入在左右子树分别计算score，然后取最优的划分，从而达到算法自己学会对空值的处理方式的目的**，文章中主要出现了两种优化：

（1）对缺失值指定默认的切分方向，如将特征对应的Nan值默认放到左子树；

（2）在搜索切分点的时候，直接跳空所有空值，便于提升整体的计算速度。

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/5.png" alt="5" style="zoom:50%;" />

Xgboost采取的策略是【先不处理那些值缺失的样本，先依据有值的特征计算特征的分割点，然后在遍历每个分割点的时候】，尝试将缺失样本划入左子树和右子树，选择使损失最优的情况。



### Column Block for Parallel Learning【并行学习】

在XGBoost模型计算过程中，【特征值的排序与切分点的选择是最耗时的部分】，文章中提出了一种**划分块**的优化方法【提前排序】，具体表现为如下流程：

每一个**“Block”**内部采用一种**Compress Sparse Column**的稀疏短阵格式，**每一列特征分别做好升序排列**，便于搜索切分点，整体的时间复杂度有效降低。对应示意图：

- 每一列特征进行排序，并且存储了指向对应梯度值的索引；
- 稀疏值不进行排序；

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/6.png" alt="6" style="zoom:50%;" />

这种**块结构存储的特征之间相互独立**，方便计算机进行并行计算。在对节点进行分裂时需要选择增益最大的特征作为分裂，这时**各个特征的增益计算可以同时进行**。



### Cache-aware Access【CPU Cache 优化】

针对一个具体的块(block)，其中（1）存储了排序好的特征值，以及（2）指向特征值所属样本的索引指针，算法需要间接地利用索引指针来获得样本的梯度值。由于块中数据是按特征值来排序的，【当索引指针指向内存中不连续的样本时】（因为排过序），无法充分利用CPU缓存来提速。文章中作者提出来了两种优化思路。

（1）提前取数（Prefetching）

对于精确搜索，【利用多线程的方式】，给**每个线程划分一个连续的缓存空间**，当training线程在按特征值的顺序计算梯度的累加时，prefetching线程可以提前将接下来的一批特征值对应的梯度加载到CPU缓存中。**非连续空间到连续空间的转换**

（2）合理设置分块大小

对于近似分桶搜索，按行分块时需要准确地选择块的大小。块太小会导致每个线程的工作量太少，切换线程的成本过高，不利于并行计算；块太大导致缓存命中率低，需要花费更多时间在读取数据上。经过反复实验，作者找到一个合理的`block_size`为$$2^{16}$$。



### Blocks for Out-of-core Computation【IO优化】

1. Block Compression：Block压缩；原始数据**在磁盘上是以压缩格式存取的**，**读取的时候，现场解压** (decompress on-the-fly)相当于牺牲一部分CPU时间，换取对磁盘IO的读取时间损耗。
2. Block Sharding：**将Block放到多个磁盘**，每个磁盘开一个prefetching线程分别读取数据到各自的缓存，提供给一个training线程使用，增加磁盘的带宽；

<img src="https://gzy-gallery.oss-cn-shanghai.aliyuncs.com/img/8.png" alt="8" style="zoom:50%;" />



## 优缺点

在分析XGBooting优缺点的时候，通过比较该算法与GBDT的差异，即可有较清楚的描述，具体表现在如下方面。

**（1）基分类器的差异**

- GBDT算法只能利用CART树作为基学习器，满足分类应用；
- XGBoost算法除了回归树之外还**支持线性的基学习器**，因此其一方面可以解决带L1与L2正则化项的逻辑回归分类问题，也可以解决线性回问题。

**（2）节点分类方法的差异**

- GBDT算法主要是利用【Gini impurity】针对特征进行节点划分；
- XGBoost经过公式推导，提出的weighted quantile sketch划分方法，【依据影响Loss的程度来确定连续特征的切分值】。

**（3）模型损失函数的差异**

- 传统GBDT在优化时只用到一阶导数信息；
- xgboost则对代价函数进行了二阶泰勒展开，二阶导数有利于梯度下降的更快更准。

**（4）模型防止过拟合的差异**

- GBDT算法无正则项，可能出现过拟合；
- Xgboost在代价函数里【加入了正则项】，用于控制模型的复杂度，降低了过拟合的可能性。

**（5）模型实现上的差异**

决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点）。xgboost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量。其能够实现在特征粒度的并行。
