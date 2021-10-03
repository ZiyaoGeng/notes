## 朴素贝叶斯

学习大纲：

1. 在分类问题中，用贝叶斯决策论推导出期望风险最小等价于最大化后验概率；
2. 如何得到后验概率？（学习联合概率，贝叶斯定理）
3. 朴素贝叶斯的由来，如何学习朴素贝叶斯模型，以及如何用朴素贝叶斯进行预测？
4. 贝叶斯估计与贝叶斯定理的区别？
5. 实现代码（在贝叶斯模型中，到底保存了什么内容）



## 1. 贝叶斯决策论

贝叶斯决策论是概率框架下实施决策的基本方法，即**利用贝叶斯理论进行决策分类**。以分类问题为例，在**所有相关概率已知的理想情况下**（主观假设），对部分未知的状态（即分类类别）进行估计，**结合期望风险**选择最优的类别标记。

假设一个0/1损失：
$$
L(Y, f(X))=\left\{\begin{array}{ll}1, & Y \neq f(X) \\ 0, & Y=f(X)\end{array}\right.
$$
其中$f(X)$为**分类决策函数**，那么期望风险函数：
$$
R_{\exp }(f)=E[L(Y, f(X))]
$$
条件期望风险：
$$
R_{\exp }(f)=E_{X} \sum_{k=1}^{K}\left[L\left(c_{k}, f(X)\right)\right] P\left(c_{k} \mid X\right)
$$
那么我们的目的是找到一个分类决策函数，使得**条件期望风险最小**，根据**贝叶斯判定准则**，对于每个样本$x$，分类决策函数$f(x)$能够最小化风险，那么整体的条件期望风险也将最小化。因此为了条件期望风险最小化，对$X=x$逐个极小化：
$$
\begin{aligned} f(x)^* &=\arg \min _{y \in \mathcal{Y}} \sum_{k=1}^{K} L\left(c_{k}, y\right) P\left(c_{k} \mid X=x \right) \\ &=\arg \min _{y \in \mathcal{Y}} \sum_{k=1}^{K} P\left(y \neq c_{k} \mid X=x\right) \\ &=\arg \min _{y \in \mathcal{Y}}\left(1-P\left(y=c_{k} \mid X=x\right)\right) \\ &=\arg \max _{y \in \mathcal{Y}} P\left(y=c_{k} \mid X=x\right) \end{aligned}
$$
$f(x)^*$被称为**贝叶斯最优分类器**，对应的风险称为**贝叶斯风险**。所以，对于贝叶斯判定准则来说，**期望风险最小化就是等价于后验概率最大化**。那么对于后验概率$P(Y|X)$ 如何得到？有两种方式：

1. 直接建模$P(Y|X)$来预测$y$（判别模型）；

2. 对**联合概率分布$P(X,Y)$建模**，然后得到后验概率$P(Y|X)$（**生成模型**）；
   $$
   P(Y|X)=\frac{P(X,Y)}{P(X)}
   $$

对于生成模型来说，根据**贝叶斯定理**，可以将其写成：
$$
P(Y|X)=\frac{P(X|Y)P(Y)}{P(X)}
$$

- $P(Y|X)$称为$Y$的后验概率；
- $P(X|Y)$是称为已知$Y$，$X$的似然性/可能性（在贝叶斯理论中的规定），也称为类条件概率；
- $P(Y)$称为$Y$的先验概率；
- $P(X)$为$X$的概率（规范化因子，与$Y$无关）；

得到后验概率后，通过**最大化后验概率（MAP）**来选择的最优决策函数。

> 事情还没有发生，要求这件事情发生的可能性的大小，是先验概率。事情已经发生，要求这件事情发生的原因是由某个因素引起的可能性的大小，是后验概率。



## 2. 朴素贝叶斯

朴素贝叶斯是一种适用于二分类和多分类分类问题的分类算法。由于通过贝叶斯定理估计后验概率，很难来计算类条件概率，因此**朴素贝叶斯方法**提出一个特别强的假设：**特征独立性假设**，即各个特征是独立同分布的，不存在交互，这也是朴素贝叶斯名称的来由（对条件概率的计算作出了简化）。

朴素贝叶斯方法主要分为两步：

1. 学习联合概率模型（先验概率、类条件概率，**不需要拟合系数**）；
2. 后验概率最大化；



### 2.1 学习联合概率分布模型

首先是学习输入输出的联合概率分布模型，根据联合概率分布：
$$
P(X, Y)=P(Y)P(X|Y)
$$
需要求得**先验概率**和**类条件概率**。

#### 2.1.1 先验概率

假设先验概率分布：
$$
P(Y=c_k), k=1,2,...,K
$$
如何去**估计先验概率**？采用**极大似然估计法**。首先我们假设随机变量（这里指$Y$）均服从参数$p$的伯努利分布。因此得到似然函数：
$$
\mathrm{L}(\mathrm{p})=\mathrm{f}_{\mathrm{D}}\left(\mathrm{y}_{1}, \mathrm{y}_{2}, \cdots, \mathrm{y}_{\mathrm{n}} \right)=\left(\begin{array}{l}\mathrm{N} \\ \mathrm{m}\end{array}\right) \mathrm{p}^{\mathrm{m}}(1-\mathrm{p})^{(\mathrm{N}-\mathrm{m})}
$$
其中$N$为所有样本数量，$m={\displaystyle\sum_{i=1}^{N} I(y_i=c_k)},p=P(Y=c_k)$。
我们对$p$微分，求极值：
$$
\begin{aligned} 0 &=\left(\begin{array}{l}\mathrm{N} \\ \mathrm{m}\end{array}\right)\left[\mathrm{mp}^{(\mathrm{m}-1)}(1-\mathrm{p})^{(\mathrm{N}-\mathrm{m})}-(\mathrm{N}-\mathrm{m}) \mathrm{p}^{\mathrm{m}}(1-\mathrm{p})^{(\mathrm{N}-\mathrm{m}-1)}\right] \\ &=\left(\begin{array}{c}\mathrm{N} \\ \mathrm{m}\end{array}\right)\left[\mathrm{p}^{(\mathrm{m}-1)}(1-\mathrm{p})^{(\mathrm{N}-\mathrm{m}-1)}(\mathrm{m}-\mathrm{Np})\right] \end{aligned}
$$
解得$p=0,p=1,p=\frac{m}{N}$

因此，
$$
P(\mathrm{Y}=\mathrm{c}_{\mathrm{k}})=\mathrm{p}=\frac{\mathrm{m}}{\mathrm{N}}=\frac{\displaystyle\sum_{\mathrm{i}=1}^{\mathrm{N}} \mathrm{I}(\mathrm{y}_{\mathrm{i}}=\mathrm{c}_{\mathrm{k}})}{\mathrm{N}}
$$

【注】在给定训练集后，可以直接计算出先验概率，然后将其保存在模型中。另外，当训练集足够大的时候，那么先验概率也就越可靠。



#### 2.1.2 类条件概率

假设类条件概率为：
$$
 P(X=x|Y=c_k)=P(X^{(1)}=x^{(1)},...,X^{(n)}=x^{(n)}|Y=c_k),k=1,2,...,K
$$
上述有**指数级的参数**，即假设每个$x^{(j)}$共有$S_j$个取值，$Y$有$K$个取值，那么参数的个数为$K\prod_{j=1}^nS_j$个，难以估计。
因此朴素贝叶斯方法作出了一个重要的假设：**条件独立性假设**，即用于分类的特征在类确定的条件下都是条件独立的。
所以：
$$
    \begin{aligned} P\left(X=x \mid Y=c_{k}\right) &=P\left(X^{(1)}=x^{(1)}, \cdots, X^{(n)}=x^{(n)} \mid Y=c_{k}\right) \\ &=\prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right) \end{aligned}  
$$
所以此时估计条件概率的方法依旧采用**极大似然法**。

【注】采用极大似然估计，**首先最重要的是需要假设随机变量（特征）$X$服从什么分布**，对于不同的假设，也对应着不同的朴素贝叶斯，例如伯努利朴素贝叶斯、高斯朴素贝叶斯、多项分布朴素贝叶斯。

**伯努利朴素贝叶斯**：

假设随机变量$Y$服从参数为$p$的**伯努利分布**【对应的朴素贝叶称为伯努利朴素贝叶斯】。  似然函数：
$$
L(\mathrm{p})=\left(\begin{array}{l}\mathrm{m} \\ \mathrm{q}\end{array}\right) \mathrm{p}^{\mathrm{q}}(\mathrm{1}-\mathrm{p})^{\mathrm{m}-\mathrm{q}}
$$
其中$\mathrm{p}=\mathrm{P}\left(\mathrm{X}^{(\mathrm{j})}=\mathrm{a}_{\mathrm{j} l} \mid \mathrm{Y}=\mathrm{c}_{\mathrm{k}}\right)$，$\mathrm{m}=\displaystyle\sum_{i=1}^{\mathrm{N}} \mathrm{I}\left(\mathrm{y}_{\mathrm{i}}=\mathrm{c}_{\mathrm{k}}\right)$，$\mathrm{q}=\displaystyle\sum_{\mathrm{i}=1}^{\mathrm{N}} \mathrm{I}\left(\mathrm{x}_{\mathrm{i}}^{(\mathrm{j})}=\mathrm{a}_{\mathrm{jl}}, \mathrm{y}_{\mathrm{i}}=\mathrm{c}_{\mathrm{k}}\right)$，我们两边对$p$求微分：
$$
\begin{aligned} 0 &=\left(\begin{array}{c}\mathrm{m} \\ \mathrm{q}\end{array}\right)\left[\mathrm{qp}^{(\mathrm{q}-1)}(1-\mathrm{p})^{(\mathrm{m}-\mathrm{q})}-(\mathrm{m}-\mathrm{q}) \mathrm{p}^{\mathrm{q}}(1-\mathrm{p})^{(\mathrm{m}-\mathrm{q}-1)}\right] \\ &=\left(\begin{array}{c}\mathrm{m} \\ \mathrm{q}\end{array}\right)\left[\mathrm{p}^{(\mathrm{q}-1)}(1-\mathrm{p})^{(\mathrm{m}-\mathrm{q}-1)}(\mathrm{q}-\mathrm{mp})\right] \end{aligned}
$$
解得$p=0,p=1,p=\frac{q}{m}$，所以条件概率$ P\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)$的极大似然估计：
$$
\begin{array}{c}  P\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)=p=\frac{q}{m}=\frac{\displaystyle\sum_{i=1}^{N} I\left(x_{i}^{(j)}=a_{j l}, y_{i}=c_{k}\right)}{\displaystyle\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)} \\ j=1,2, \cdots, n ; \quad l=1,2, \cdots, S_{j} ; \quad k=1,2, \cdots, K\end{array}
$$
其中$x_{i}^{(j)}$是第$i$个样本的第$j$个特征；$a_{jl}$是第$j$个特征可能取的第$l$个值。



**高斯朴素贝叶斯：**

朴素贝叶斯可以扩展到实值属性，最常见的是通过假设高斯分布，这种朴素贝叶斯的扩展称为高斯朴素贝叶斯。其他函数可以用来估计数据的分布，但高斯分布(或正态分布)是最容易处理的，因为**只需要估计训练数据的平均值和标准偏差**。

假设随机变量$Y$服从参数为$\mu,\sigma^2$的**高斯分布**，似然函数：
$$
L\left(\mu, \sigma^{2}\right)=\left(\frac{1}{\sqrt{2 \pi \sigma^{2}}}\right)^{N} e^{-\displaystyle\sum_{i=1}^{N} \frac{\left(x_{i}^{(j)}-\mu\right)^{2}}{2 \sigma^{2}}}
$$
两边分别取ln，
$$
\ln L\left(\mu, \sigma^{2}\right)=N \ln \frac{1}{\sqrt{2 \pi}}-\frac{N}{2} \ln \sigma^{2}-\sum_{i=1}^{N} \frac{\left(x_{i}^{(j)}-\mu\right)^{2}}{2 \sigma^{2}}
$$
其中分别对$\mu,\sigma^2$求偏导：
$$
\begin{array}{l}\frac{\partial}{\partial \mu} \ln L\left(\mu, \sigma^{2}\right)=\frac{1}{\sigma^{2}} \displaystyle\sum_{i=1}^{N}\left(x_{i}^{(j)}-\mu\right)=0 \\ \frac{\partial}{\partial \sigma^{2}} \ln L\left(\mu, \sigma^{2}\right)=-\frac{N}{2 \sigma^{2}}+\frac{1}{2 \sigma^{4}} \displaystyle\sum_{i=1}^{N}\left(x_{i}^{(j)}-\mu\right)^{2}=0\end{array}
$$
因此求得：
$$
\mu=\frac{1}{N}\sum_{i=1}^Nx_i^{(j)}\\
\sigma^2=\frac{1}{N}\sum_{i=1}^N(x_i^{(j)}-\mu)^2
$$
因此：
$$
P\left(X^{(j)} \mid Y=c_k\right)=\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left(-\frac{\left(x_{i}^{(j)}-\mu\right)^{2}}{2 \sigma^{2}}\right)
$$
【注】这里是针对每个类别$c_k$、每个特征计算得到对应的$\mu,\sigma^2 $。（结合代码理解）



### 2.2 后验概率最大化

通过得到联合概率分布模型的目的是对于给定一个输入特征$x$，通过**贝叶斯定理**，来计算后验概率$P(Y=c_k|X=x)$。因此，
$$
P\left(Y=c_{k} \mid X=x\right)=\frac{P\left(X=x \mid Y=c_{k}\right) P\left(Y=c_{k}\right)}{\displaystyle\sum_{k} P\left(X=x \mid Y=c_{k}\right) P\left(Y=c_{k}\right)}
$$
将类条件概率（基于独立性假设）的估计结果代入：
$$
P\left(Y=c_{k} \mid X=x\right)=\frac{P\left(Y=c_{k}\right) \displaystyle\prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}{\displaystyle\sum_{k} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}, \quad k=1,2, \cdots, K
$$
以上就是朴素贝叶斯分类的基本内容。那么，求出所有的对类$c_k$的估计，概率最大的即为预测的分类【最大后验概率】，这称为**贝叶斯分类器**：
$$
y=f(x)=arg \ \max_{c_k}\frac{P\left(Y=c_{k}\right) \displaystyle\prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}{\displaystyle\sum_{k} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}
$$
由于分母对于所有的$c_k$都是相同的，所以：
$$
y=\arg \max _{c_{k}} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)
$$


## 3. 贝叶斯估计

朴素贝叶斯与贝叶斯估计并不是一个概念。由于**极大似然估计可能会出现所估计的概率值为0的情况**，这会影响到后验概率的计算结果（即当前类的预测结果为0）。解决该方法可以采用**贝叶斯估计**，条件概率的贝叶斯估计为：
$$
P_{\lambda}\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)=\frac{\displaystyle\sum_{i=1}^{N} I\left(x_{i}^{(j)}=a_{j l}, y_{i}=c_{k}\right)+\lambda }{\displaystyle\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)+S_{j} \lambda}
$$
其中，当$\lambda=1$，称为**拉普拉斯平滑**。

先验概率的贝叶斯估计是：
$$
P_{\lambda}\left(Y=c_{k}\right)=\frac{\displaystyle\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)+\lambda}{N+K \lambda}
$$



## 4. 实现代码

高斯朴素贝叶斯实现代码：

```python
class Gaussian_NaiveBayes:
	"""
	Gaussian Naive Bayes Model
	"""
	def __init__(self):
		self.model = None
		self.total_rows = None

	# Calculate the mean of a list of numbers
	@staticmethod
	def mean(X):
		return sum(X) / float(len(X))

	# Calculate the standard deviation of a list of numbers
	def stdev(self, numbers):
		avg = self.mean(numbers)
		return math.sqrt(sum([pow(x-avg, 2) for x in numbers]) / float(len(numbers)-1))

	# Calculate mean, stdev for each column in a dataset
	def summarize(self, train_data):
		# separates the dataset into separate lists for each row.
		summaries = [(self.mean(column), self.stdev(column), len(column)) for column in zip(*train_data)]
		return summaries

	# Build model
	# Split dataset by class then calculate statistics for each row
	def fit(self, X, y):
		self.total_rows = len(X)
		labels = list(set(y))
		data = {label: [] for label in labels}
		for f, label in zip(X, y):
			data[label].append(f)
		# Save list of mean and stdev for each class in train dataset
		self.model = {
		label: self.summarize(value) for label, value in data.items()
		}

	# Calculate the Gaussian probability distribution function for x
	def gaussian_probabality(self, x, mean, stdev):
		exponent = math.exp(-math.pow(x - mean, 2) / 
			(2 * math.pow(stdev, 2)))
		return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

	# Calculate the probabilities of predicting each class for a given row
	def calculate_probabalities(self, input_data):
		probalalities = {}
		for label, summaries in self.model.items():
			probalalities[label] = summaries[0][2] / self.total_rows
			for i in range(len(summaries)):
				mean, stdev, _ = summaries[i]
				probalalities[label] *= self.gaussian_probabality(
					input_data[i], mean, stdev)
		return probalalities

	# Predict the class for a given row
	def predict(self, X_test):
		labels = []
		for i in X_test:
			label = sorted(
				self.calculate_probabalities(i).items(),
				key=lambda x: x[-1])[-1][0]
			labels.append(label)
		return np.array(labels)

	# Calculate accuracy percentage
	def score(self, X_test, y_test):
		label = self.predict(X_test)
		right = [1 if l == y else 0 for l, y in zip(label, y_test)]
		return sum(right) / float(len(X_test))
```



## 5. 面试真题

1、为什么朴素贝叶斯如此“朴素”？

因为朴素贝叶斯提出了一个很强的假设：假定**所有的特征在数据集中是独立同分布的**，这对条件概率的计算进行了简化，但这个假设在现实世界中是很不真实的。



2、简单说说贝叶斯定理。

贝叶斯公式的一个用途，即透过已知的三个概率而推出第四个概率。贝叶斯定理跟随机变量的条件概率以及边际概率分布（只包含其中部分变量的概率分布）有关。

贝叶斯定理的公式为：
$$
P(Y|X)=\frac{P(X|Y)P(Y)}{P(X)}=\frac{P(X|Y)P(Y)}{\sum_yP(X|Y=y)P(Y=y)}
$$

3、朴素贝叶斯基本原理和预测过程。

证明最小经验损失等价于最大后验概率---->最大后验概率如何求解---->通过联合概率求出后验概率---->先验概率和类条件概率的估计（假设特征独立同分布）---->求出最大后验概率。



4、朴素贝叶斯如何垃圾分类【字节】

【问题定义】垃圾分类为一个多（4）分类问题（服从伯努利分布），假设垃圾的特征共有$n$个（服从高斯分布）。



5、人群中男人色盲的概率为 5%，女人为 0.25%。从男女人数相等的人群中随机选一 人，恰好是色盲。求此人是男人的概率。【字节、贝叶斯公式】

$P(Y|X)=0.05$ 

$P(Y|X)=0.0025$  

求$P(X|Y)=\frac{P(Y|X)P(X)}{P(Y)}=\frac{P(Y|X)P(X)}{P(Y|X)P(X)+P(Y|\tilde{X})P(\tilde{X})}=\frac{0.05*0.5}{0.05*0.5+0.0025*0.5}=\frac{500}{525}=\frac{20}{21}$



6、朴素贝叶斯的算法实现【字节】

大的框架需要理解，建立类模型，保存的内容有哪些？如何计算？如何预测？

```python
class Gaussian_NaiveBayes：
	def init(self):
    self.model = None  # 模型在得到训练集后就直接建立（包含哪些内容）
    
   def mean(self, X):  # 计算均值
    
   def stdev(self, X):  # 计算标准差
    
   def fit(self, X, y):  # 模型的拟合（包含每个类别-每个特征的均值以及标准差）
    # 统计类别的种类，set(y)
    # 每个类别对应有哪些样本分好
    # 遍历每个类别，来计算当前类别下每个特征列的均值和标准差
    
   def summaries(self, train_data):  # 服务于fit的第三步，计算单个类别中每个特征列的均值标准差
    
   def gaussian_probabality(self, x, mean, stdev): # 在确定类别后，计算每个特征列的高斯概率分布
    
   def calculate_probabalities(self, input_data):  # 计算后验概率
     # 通过模型中保存对应的mean、stdev，通过NB公式，求解每个类别的后验概率，类条件概率使用上述求解
      
   def predict(self, X_test):  # 对于每个样本，求出其最大后验概率
```



7、贝叶斯网络原理【字节】