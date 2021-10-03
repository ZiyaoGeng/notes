# 逻辑回归

学习大纲：

1. 最重要的就是公式推导，即（1.1）-（1.16）部分；
2. 了解与线性回归、SVM、决策树的区别；



## 1. 建立假设函数

假设某个事件发生的概率为$$p$$，那么该事件发生的几率是发生的概率与不发生的概率的比值，即$$\frac{p}{1-p}$$，该事件的对数几率（logit）函数为：
$$
logit(p)=log\frac{p}{1-p}
$$
因此【假设函数】，
$$
log\frac{P(Y=1|x)}{1-P(Y=1|x)}=w\cdot x
$$
因此**逻辑斯谛回归模型**的条件概率如下：
$$
\begin{eqnarray}
P(Y=1|x) &= \frac{exp(w\cdot x)}{1+ exp(w\cdot x)} &= \pi{x} \tag{1.1}\\
 P(Y=0|x) &= \frac{1}{1+ exp(w\cdot x)} &= 1 - \pi{x} \tag{1.2}
\end{eqnarray}
$$

其中$$w=(w^{(1)},...b)^T,x=(x^{(1)},...1)^T$$。



## 2. 参数估计

给定训练集$$T=\{(x_1,y_1),...,(x_m,y_m)\}$$，采用**极大似然估计**来估计模型参数，似然函数为：
$$
L(w)=\prod_{i=1}^{m}[\pi(x_i)]^{y_i}[1-\pi(x_i)]^{1-y_i} \tag{1.3}
$$
对数似然函数为：
$$
L(w)=\sum^{m}_{i=1}y_ilog(\pi(x_i))+(1-y_i)log(1-\pi(x_i)) \tag{1.4}
$$
对其求最大值，估计参数$$w$$：
$$
w^*=\underset{w}{argmax}\sum^{m}_{i=1}y_ilog(\pi(x_i))+(1-y_i)log(1-\pi(x_i)) \tag{1.5}
$$
再将其改为最小化负的对数似然函数：
$$
w^*=\underset{w}{argmin}\sum^{m}_{i=1}-y_ilog(\pi(x_i))-(1-y_i)log(1-\pi(x_i))\tag{1.6}
$$
如此，就得到了Logistic回归的损失函数，即机器学习中的「二元交叉熵」（Binary crossentropy）：
$$
J(w)=-\frac{1}{m}\sum^{m}_{i=1}y_ilog(\pi(x_i))+(1-y_i)log(1-\pi(x_i)) \tag{1.7}
$$
此时转变为以负对数似然函数为目标函数的最优化问题，采用**梯度下降法**进行优化。



## 3. 梯度下降

$$
\begin{eqnarray}
\frac{\partial{}}{\partial{w}}J(w)&=\frac{\partial{}}{\partial{w}}[-\frac{1}{m}\sum^{m}_{i=1}-y_ilog(1+e^{-wx_i})-(1-y_i)log(1+e^{wx_i})] \tag{1.8}\\&=-\frac{1}{m}\sum\limits_{i=1}^{m}{[-{y}_i\frac{-x_i{{e}^{-wx_i}}}{1+e^{-wx_i}}-( 1-y_{i} )\frac{x_i{{e}^{{w}{{x}_i}}}}{1+{e^{wx_i}}}}] \tag{1.9}\\
&=-\frac{1}{m}\sum_{i=1}^{m}[y_i\frac{x_i}{1+e^{wx_i}}-(1-y_i)\frac{x_ie^{wx_i}}{1+e^{wx_i}}] \tag{1.10}\\
&=-\frac{1}{m}\sum_{i=1}^{m}\frac{y_ix_i-x_ie^{wx_i}+y_ix_ie^{wx_i}}{1+e^{wx_i}} \tag{1.11}\\
&=-\frac{1}{m}\sum_{i=1}^{m}\frac{y_i(1+e^{wx_i})-e^{wx_i}}{1+e^{wx_i}}x_i \tag{1.12}\\
&=-\frac{1}{m}\sum_{i=1}^{m}(y_i-\frac{e^{wx_i}}{1+e^{wx_i}})x_i \tag{1.13}\\
&=\frac{1}{m}\sum_{i=1}^{m}(\pi(x_i)-y_i)x_i \tag{1.14}
\end{eqnarray}
$$

【注】\(1.8\)式提取了负号；

故参数更新为：
$$
\begin{eqnarray}
w :=& w-\alpha\frac{\partial }{\partial w}J(w) \tag{1.15}\\
:=& w-\alpha\frac{1}{m}\sum_{i=1}^{m}(\pi(x_i)-y_i)x_i \tag{1.16}
\end{eqnarray}
$$
其中$$\alpha$$为学习率。



## 4. 加入正则化

当然，在损失函数中，我们也可以加入正则化来抑制过拟合：
$$
J(w)=-\frac{1}{m}\sum^{m}_{i=1}y_ilog(\pi(x_i))+(1-y_i)log(1-\pi(x_i))+\frac{\lambda}{2m}\|w\|_2
$$
其中，$$\lambda$$为正则化参数，$$\|\cdot\|_2为L_2$$正则化项。





## 5. 面试真题

1、逻辑回归推导？

（1.1）-（1.16）



2、逻辑回归如何实现多分类？

（1）【softmax回归】修改逻辑回归的损失函数；使用**softmax函数**构造模型解决多分类问题，softmax分类模型会有相同于类别数的输出，输出的值为对于样本属于各个类别的概率，最后对于样本进行预测的类型为概率值最高的那个类别；

> **Softmax函数**，或称**归一化指数函数**，是Sigmoid函数的一种推广，它能将一个含任意实数的K维向量$$\mathbf{z}$$“压缩”到另一个K维实向量$$\sigma(\mathbf{z})$$中，使得每一个元素的范围都在$$(0,1)$$之间，并且所有元素的和为1：
> $$
> \sigma(\mathbf{z})_{j}=\frac{e^{z_{j}}}{\sum_{k=1}^{K} e^{z_{k}}} \quad \text { for } j=1, \ldots, K
> $$
> 样本$$x$$属于第$$j$$个分类的概率为：
> $$
> P(y=j \mid \mathbf{x})=\frac{e^{\mathbf{x}^{\top} \mathbf{w}}}{\sum_{k=1}^{K} e^{\mathbf{x}^{\top} \mathbf{w}}}
> $$

$$
h_{\theta}\left(x_{i}\right)=\left[\begin{array}{c}p\left(y_{i}=1 \mid x_{i} ; \theta\right) \\ p\left(y_{i}=2 \mid x_{i} ; \theta\right) \\ \vdots \\ p\left(y_{i}=k \mid x_{i} ; \theta\right)\end{array}\right]=\frac{1}{\sum_{j=1}^{k} e^{\theta_{j}^{T} x_{i}}}\left[\begin{array}{c}e^{\theta_{1}^{T} x_{i}} \\ e^{\theta_{2}^{T} x_{i}} \\ \vdots \\ e^{\theta_{k}^{T} x_{i}}\end{array}\right]
$$

<img src="https://pic3.zhimg.com/80/v2-fe260e1336f1c75a51445e06b8e6d9a2_1440w.jpg" alt="img" style="zoom:50%;" />

代价函数：
$$
L(\theta)=-\frac{1}{m}\left[\sum_{i=1}^{m} \sum_{j=1}^{k} 1\left\{y_{i}=j\right\} \log \frac{e^{\theta_{j}^{T} x_{i}}}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_{i}}}\right]
$$


（2）根据每个类别都建立一个二分类器，本类别的样本标签定义为0，其它分类样本标签定义为1，则有多少个类别就构造多少个逻辑回归分类器；

【注】若所有类别之间有明显的互斥则使用softmax分类器，若所有类别不互斥有交叉的情况则构造相应类别个数的逻辑回归分类器。



4、LR与线性回归的比较？

联系：

- 线性回归和逻辑回归都是**广义线性回归模型的特例**；

区别：

- 线性回归用于**回归问题**，逻辑回归用于**分类问题**；
- 线性回归用最小二乘法（**最小化预测和实际之间的欧氏距离**）来估计参数，逻辑回归使用极大似然法估计参数（**最大化预测属于实际的概率**来最小化预测和实际之间的“距离”）；
- 线性回归更容易受到异常值的影响，而LR 对异常值有较好的稳定性；



6、逻辑回归反向传播伪代码？

```python
class LogisticReressionClassifier:
    def __init__(self, max_iter=200, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def sigmoid(self, x):  # 定义sigmoid
        return 1 / (1 + exp(-x))

    def data_matrix(self, X):
        data_mat = []
        for d in X:
            data_mat.append([1.0, *d])
        return data_mat

    def fit(self, X, y):  # 拟合，更新权重参数
        data_mat = self.data_matrix(X)  # 矩阵的转化
        self.weights = np.zeros((len(data_mat[0]), 1), dtype=np.float32)  # 建立初始权重参数

        for iter_ in range(self.max_iter):  # 迭代的次数
            for i in range(len(X)):
                result = self.sigmoid(np.dot(data_mat[i], self.weights))
                error = y[i] - result  # 误差
                self.weights += self.learning_rate * error * np.transpose(
                    [data_mat[i]])  # error * np.transpose([data_mat[i]]) 表示梯度
```



7、为什么逻辑回归损失函数是交叉熵？

【1】从极大似然估计的角度可以推导出交叉熵；（1.3-1.7）

【2】从KL散度（熵的角度）去理解；

逻辑回归最后的计算结果是各个分类的概率，可以看做是各个分类的概率分布，假设估计得到的概率分布是Q(x)， 真实的概率分布是P(x)，那么这两个概率分布的距离怎么衡量呢？可以用KL散度来衡量两个概率分布的差别：
$$
D_{KL}(p||q)=\displaystyle\sum_xp(x)log\frac{p(x)}{q(x)}=\displaystyle\sum_xp(x)(logp(x)-logq(x))
$$
且$$D_{KL}(p||q)=-\displaystyle\sum_xp(x)logq(x)-(-\displaystyle\sum_xp(x)logp(x))=H(p,q)-H(p)$$，**KL散度=交叉熵-真实概率分布的熵**，因为交叉熵越大，KL散度越大，所以也可以用交叉熵来衡量两个概率分布之间的距离，所以可以用交叉熵作为逻辑回归的损失函数。
