<a name="content">目录</a>

[机器学习在生物信息中的应用](#title)
- [1. 根据免疫组库TCRβ预测病人的CMV感染状态](#predict-cmv-serostatus-using-tcr)
- [2. SC3：单细胞表达谱的无监督聚类](#unsupervise-clustering-for-single-cell-profile)
- [3. GATK的VQSR](#gatk-vqsr)



- [补充知识](#supplementary-knowledge)
	- [*1. beta分布](#beta-distribution)
		- [*1.1. 什么是beta分布](#what-is-beta-distribution)
		- [*1.2. 理解beta分布](#understand-beta-distribution)
		- [*1.3. Beta分布相关公式的推导](#formula-derivation-for-beta-distribution)
			- [*1..3.1. 后验概率分布为什么是$Beta(\alpha_0+hit,\beta_0+miss)$](#posterior-probability-of-beta-distribution)
	- [*2. 重复NG免疫组库TCRβ文章的图](#re-draw-pictrue-in-ng-paper)
		- [*2.1. Fisher检验筛选CMV阳性（CMV<sup>+</sup>）相关克隆](#aquire-cmv-positive-cdr3-clone)
		- [*2.2. 绘制表型负荷相关散点图](#draw-scaterplot-for-phenotype-burden)
			- [*2.2.1. 训练集](#draw-scaterplot-for-phenotype-burden-trainset)
			- [*2.2.2. 测试集](#draw-scaterplot-for-phenotype-burden-testset)
		- [*2.2. TCRβ在两组间分布偏好性的散点图](#scatterplot-showing-the-incidence-bias-of-tcr)
	- [*3. FDR的计算方法](#calculate-fdr)
		- [*3.1. 回顾那些统计检验方法](#review-those-statistic-methods)
			- [*3.1.1. T-test与Moderated t-Test](#t-test-and-moderated-t-Test)
		- [*3.2. 多重假设检验的必要性](#necessary-of-multiple-hypothesis-tests)
		- [*3.3. 区别p值和q值](#distinguish-pvalue-and-qvalue)
		- [*3.4. 如何计算FDR？](#how-to-calculate-fdr)
			- [*3.4.1. Benjamini-Hochberg procedure (BH)](#calculate-fdr-by-benjamini-procedure)
			- [*3.4.2. Bonferroni 校正](#calculate-fdr-by-bonferroni-correction)



<h1 name="title">机器学习在生物信息中的应用</h1>

<a name="predict-cmv-serostatus-using-tcr"><h2>1. 根据免疫组库TCRβ预测病人的CMV感染状态 [<sup>目录</sup>](#content)</h2></a>

这个项目用到了641个样本（cohort 1），包括352 例CMV阴性(CMV<sub>-</sub>)和289例CMV阳性(CMV<sub>+</sub>)

外部验证用到了120个样本（cohort 2）

该机器学习的任务为：

<p align="center"><img src=./picture/App-ML-in-Bioinfo-TCRbeta-classification-4.png width=600 /></p>

讨论 TCRβ 免疫组的数据特点：

> - 可能出现的TCRβ的集合非常大，而单个样本只能从中检测到稀疏的少数几个；
> - 一个新样本中很可能会出现训练样本集合中未出现的TCRβ克隆类型；
> - 对于一个给定的TCR，它对给定抗原肽的结合亲和力会受到HLA类型的调控，因此原始的用于判别分析的特征集合还受到隐变量——HLA类型的影响；

<p align="center"><img src=./picture/App-ML-in-Bioinfo-TCRbeta-classification-workflow.png width=600 /></p>

- 特征选择：鉴定表型相关的TCRβs

	使用Fisher精确检验（单尾检验，具体实现过程请查看文末 [*2.1. Fisher检验筛选CMV阳性（CMV<sup>+</sup>）相关克隆](#2-1-Fisher检验筛选CMV阳性（CMV-）相关克隆)）：

	<p align="center"><img src=./picture/App-ML-in-Bioinfo-TCRbeta-classification-3.png width=300 /></p>

	Fisher检验的阈值设为：$P<1\times 10^{-4}$，FDR<0.14（该FDR的计算方法见文末 [*3. FDR的计算方法](#3-FDR的计算方法) ），且富集在CMV<sup>+</sup>样本中，从而得到与CMV<sup>+</sup>相关的CDR3克隆集合，共有164个

	通过下面的TCRβ克隆在两组中的发生率的散点图可以明显地看到，筛出的表型相关的TCRβ克隆的确显著地表达在CMV+组中

	<p align="center"><img src=./picture/App-ML-in-Bioinfo-TCRbeta-classification-5.png width=500 /></p>

- 计算表型负荷（phenotype burden）

	一个样本的表型负荷（phenotype burden）被定义为：

	> 该样本的所有unique TCRβs中与阳性表型相关的克隆的数量所占的比例
	>
	> 若阳性表型相关的克隆的集合记为CDR，样本i的unique克隆集合记为CDR<sub>i</sub>，则它的表型负荷为：
	>
	> $$PB_i=\frac{||CDR_i \cap CDR||}{||CDR_i||}$$
	>
	> 其中$||·||$表示集合`·`中元素的数量

	下图是将上面的表型负荷计算公式中的分子与分母分别作为纵轴和横轴，画成二维的散点图

	<p align="center"><img src=./picture/App-ML-in-Bioinfo-TCRbeta-classification-6.png width=600 /></p>

	可以明显地看出两类样本在这个层面上来看，有很好的区分度

- 基于二项分布的贝叶斯判别模型

	基本思想：

	> 对于$CMV^+$相关TCR数为$k'$，total unique TCR数为$n'$的样本，认为它一个概率为它的表型负荷$\theta'$（$\theta'$服从Beta分布），$n=n'$, $k=k'$的二项分布（伯努利分布），根据贝叶斯思想，构造最优贝叶斯分类器，即
	>
	> $$h(k',n')=arg \max_{x \in \{+,\,-\}} p(c=x \mid k',\,n')$$
	>
	> 其中
	>
	> $$p(c=x \mid k',\,n')=\frac{p(k' \mid n',\,c)p(c)}{p(k')}$$
	>
	> 而$p(k')$是一个常数，对分类器的结果没有影响，可以省略

	那么就需要根据训练集估计：

	- 类先验概率$p(c)$
	- 类条件概率（似然）$p(k' \mid n',\,c)$

	**（1）首先根据概率图模型推出单个样本的概率表示公式**

	概率图模型如下：

	<p align="center"><img src=./picture/App-ML-in-Bioinfo-TCRbeta-classification-7.png width=600 /></p>

	则对于j样本，我们可以算出它的 $\theta_i$ 的后验分布：

	$$p(\theta_i \mid y_{ij},\,\alpha_i,\,\beta_i)=\frac{p(\theta_i \mid \alpha_i,\,\beta_i)p(y_{ij} \mid \theta_i)}{p(y_{ij})}$$

	其中，
	
	> - $p(y_{ij})$：表示事件$(k_{ij} \mid n_{ij})$发生的概率，即$p(k_{ij}\mid n_{ij},\,c_i)$
	> - $p(y_{ij} \mid \theta_i)$：表示$Binomial(k_{ij} \mid n_{ij},\,\theta_i)=\left(\begin{matrix}n_{ij}\\ k_{ij}\end{matrix}\right)\theta_i^{k_{ij}}(1-\theta_i)^{n_{ij}-k_{}ij}$
	> - $p(\theta_i \mid \alpha_i,\,\beta_i)$：表示$\theta_i$的先验分布$Beta(\theta_i\mid\alpha_i,\,\beta_i)$

	对上面的公式进一步推导

	$$\begin{aligned}
	p(\theta_i \mid y_{ij},\,\alpha_i,\,\beta_i) &=\frac{p(\theta_i \mid \alpha_i,\,\beta_i)p(y_{ij} \mid \theta_i)}{p(y_{ij})}\\
	&=\frac{1}{p(y_{ij})}Beta(\theta_i \mid \alpha_i,\,\beta_i)Binomial(k_{ij}\mid n_{ij},\,\theta_i) \\
	&=\frac{1}{p(y_{ij})} \quad \{\frac{1}{B(\alpha_i,\,\beta_i)}\theta_i^{\alpha_i-1}(1-\theta_i)^{\beta_i-1}\} \quad \{\left(\begin{matrix}n_{ij}\\ k_{ij}\end{matrix}\right)\theta_i^{k_{ij}}(1-\theta_i)^{n_{ij}-k_{ij}}\} \\
	&=\frac{1}{p(y_{ij})} \quad \frac{1}{B(\alpha_i,\,\beta_i)} \left(\begin{matrix}n_{ij}\\ k_{ij}\end{matrix}\right) \theta_i^{(\alpha_i+k_{ij})-1}(1-\theta_i)^{(\beta_i+n_{ij}-k_{ij})-1} \\
	&=\frac{1}{p(y_{ij})} \quad  \frac{B(\alpha_i+k_{ij},\,\beta_i+n_{ij}-k_{ij})}{B(\alpha_i,\,\beta_i)} \left(\begin{matrix}n_{ij}\\ k_{ij}\end{matrix}\right) \frac{1}{B(\alpha_i+k_{ij},\,\beta_i+n_{ij}-k_{ij})} \theta_i^{(\alpha_i+k_{ij})-1}(1-\theta_i)^{(\beta_i+n_{ij}-k_{ij})-1} \\
	&=\frac{1}{p(y_{ij})} \quad  \frac{B(\alpha_i+k_{ij},\,\beta_i+n_{ij}-k_{ij})}{B(\alpha_i,\,\beta_i)} \left(\begin{matrix}n_{ij}\\ k_{ij}\end{matrix}\right) \quad Beta(\alpha_i+k_{ij},\,\beta_i+n_{ij}-k_{ij})
	\end{aligned}
	$$
	
	根据Beta分布的先验分布的，已知

	$$p(\theta_i \mid y_{ij},\,\alpha_i,\,\beta_i)=Beta(\alpha_i+k_{ij},\,\beta_i+n_{ij}-k_{ij})$$

	因此

	$$p(k_{ij}\mid n_{ij},\,c_i)=p(y_{ij})=\left(\begin{matrix}n_{ij}\\ k_{ij}\end{matrix}\right) \frac{B(\alpha_i+k_{ij},\,\beta_i+n_{ij}-k_{ij})}{B(\alpha_i,\,\beta_i)} $$

	这样我们就得到单个样本的概率表示公式，其中$B(·)$是Beta函数，从上面的表达式中，我们可以看出$p(k_{ij}\mid n_{ij},\,c_i)$是$\alpha_i$和$\beta_i$的函数

	**（2）优化每个组的表型负荷 $\theta_i$ 的先验分布的两个参数 $\alpha_i$ 和 $\theta_i$——最大似然法**

	我们要最大化$c_i$组的样本集合它们的联合概率：

	$$
	\begin{aligned}
	p(k_i \mid n_i,\,c_i) &=\prod_{j,j \in c_i} \left(\begin{matrix}n_{ij}\\ k_{ij}\end{matrix}\right) \frac{B(\alpha_i+k_{ij},\,\beta_i+n_{ij}-k_{ij})}{B(\alpha_i,\,\beta_i)}\\
	&= \prod_{j,j \in c_i} \left(\begin{matrix}n_{ij}\\ k_{ij}\end{matrix}\right) \quad \prod_{j,j \in c_i} \frac{B(\alpha_i+k_{ij},\,\beta_i+n_{ij}-k_{ij})}{B(\alpha_i,\,\beta_i)}
	\end{aligned}
	$$

	其中，$\prod_{j,\,j \in c_i} \left(\begin{matrix}n_{ij}\\ k_{ij}\end{matrix}\right)$是常数，可以省略，则

	$$p(k_i \mid n_i,\,c_i)=\prod_{j,j \in c_i} \frac{B(\alpha_i+k_{ij},\,\beta_i+n_{ij}-k_{ij})}{B(\alpha_i,\,\beta_i)}$$

	对它取对数，得到

	$$l(\alpha_i,\,\beta_i)=\log p(k_i \mid n_i,\,c_i) \\ = \sum_{j,\,j \in c_i} \log B(\alpha_i+k_{ij}, \, \beta_i+n_{ij}-k_{ij})-N_i\log B(\alpha_i, \, \beta_i)$$

	其中，$N_i$是属于$c_i$组的样本数

	因此优化目标为：

	$$(\alpha_i^*,\beta_i^*)=arg \, \max\limits_{\alpha_i,\,\beta_i}l(\alpha_i,\,\beta_i)$$

	$l(\alpha_i,\,\beta_i)$分别对$\alpha_i$和$\beta_i$求偏导

	$$\frac{\partial l}{\partial \alpha_i}=-N_i(\psi(\alpha_i)-\psi(\alpha_i+\beta_i))+\sum_{j,\,j \in c_i}(\psi(\alpha_i+k_{ij})-\psi(\alpha_i+\beta_i+n_{ij}))$$

	<br>

	$$\frac{\partial l}{\partial \beta_i}=-N_i(\psi(\beta_i)-\psi(\alpha_i+\beta_i))+\sum_{j,\,j \in c_i}(\psi(\beta_i+n_{ij}-k_{ij})-\psi(\alpha_i+\beta_i+n_{ij}))$$

	其中，$\psi(·)$是伽马函数

	使用梯度上升（gradient ascent）法来求解优化目标，其中梯度的公式为：

	$$\alpha_i := \alpha_i+\alpha\frac{\partial l}{\partial \alpha_i}\\ \beta_i := \beta_i+\alpha\frac{\partial l}{\partial \beta_i}$$

	最终得到的解记为$\alpha_i^*$和$\beta_i^*$，其中

	$$
	\alpha_+^*=4.0,\quad \beta_+^*=51,820.1\\
	\alpha_-^*=26.7,\quad \beta_-^*=2,814,963.8
	$$

	（3）根据训练好的分类器对新样本进行分类

	分类器为

	$$
	\begin{aligned}
	h(k',n') &= arg \max_{x \in \{CMV^+,\,CMV^-\}} p(c=x \mid n',k') \\
	&=arg \max_{x \in \{CMV^+,\,CMV^-\}} p(k' \mid n', c=x)p(c=x) \\
	&=arg \max_{x \in \{CMV^+,\,CMV^-\}}\left(\begin{matrix}n'\\k'\end{matrix}\right)\frac{B(\alpha_x^*+k',\,\beta_x^*+n'-k')}{B(\alpha_x^*,\,\beta_x^*)}\frac{N_x}{N}
	\end{aligned}
	$$

<a name="predict-cmv-serostatus-using-tcr"><h2>2. SC3：单细胞表达谱的无监督聚类 [<sup>目录</sup>](#content)</h2></a>

该聚类方法名为SC3（Single-Cell Consensus Clustering）

该方法本质上就是K-means聚类，不过在执行K-means聚类的前后进行了一些特殊的操作：

> - **k-means聚类前**：进行了数据预处理，即特征的构造，称为特征工程，该方法中是对输入的原始特征空间进行PCA变换或拉普拉斯矩阵变换，对变换后的新特征矩阵逐渐增加提取的主成分数，来构造一系列新特征；
> - **k-means聚类后**：特征工程构造出来的一系列新特征集合，基于这些新特征集合通过k-means聚类能得到一系列不同的聚类结果，尝试对这些聚类结果总结出consensus clustering

<p align="center"><img src=./picture/T-cell-sequencing-in-cancers-colorectal-cancer-2.png width=800 /></p>

本人比较好奇的地方是：**怎么从一系列不同的聚类结果中总结出consensus clustering？**

> 使用CSPA算法（cluster-based similarity partitioning algorithm）
> 
> （1）对每一个聚类结果按照以下方法构造二值相似度矩阵S：如果两个样本i和j在该聚类结果中被聚到同一个集合中，则它们之间的相似度为1，在二值相似度矩阵中对应的值 S<sub>i,j</sub> = 1，否则S<sub>i,j</sub> = 0；
> 
> （2）对所有的聚类结果的二值相似度矩阵S取平均，得到consensus matrix；
> 
> （3）基于consensus matrix进行层次聚类，得到最终的consensus clustering；

<a name="gatk-vqsr"><h2>2. GATK的VQSR [<sup>目录</sup>](#content)</h2></a>











<a name="supplementary-knowledge"><h2>补充知识 [<sup>目录</sup>](#content)</h2></a>

<a name="beta-distribution"><h3>*1. beta分布 [<sup>目录</sup>](#content)</h3></a>

<a name="what-is-beta-distribution"><h4>*1.1. 什么是beta分布 [<sup>目录</sup>](#content)</h4></a>

对于硬币或者骰子这样的简单实验，我们事先能很准确地掌握系统成功的概率。

然而通常情况下，系统成功的概率是未知的，但是根据频率学派的观点，我们可以**通过频率来估计概率**。为了测试系统的成功概率，我们做n次试验，统计成功的次数k，于是很直观地就可以计算出。然而由于系统成功的概率是未知的，这个公式计算出的只是系统成功概率的**最佳估计**。也就是说实际上也可能为其它的值，只是为其它的值的概率较小。因此我们并不能完全确定硬币出现正面的概率就是该值，所以也是一个随机变量，它符合Beta分布，其取值范围为0到1

用一句话来说，beta分布可以看作一个**概率的概率密度分布**，当你不知道一个东西的具体概率是多少时，它可以给出了所有概率出现的可能性大小。

Beta分布是关于连续变量$\mu \in [0,1]$的概率分布，它由两个参数a>0和b>0确定，其概率密度函数的图形如下：

$$
p(\mu \mid a,b)=Beta(\mu \mid a,b)=\frac{1}{B(a,b)}\mu^{a-1}(1-\mu)^{b-1}
$$

其中B(a,b)是Beta函数：

$$B(a,b)=\frac{\tau(a)\tau(b)}{\tau(a+b)}$$

Beta分布的两个重要参数分别为：

$$E[\mu]=\frac{a}{a+b}$$

<br>

$$Var[\mu]=\frac{ab}{(a+b)^2(a+b+1)}$$

<a name="understand-beta-distribution"><h4>*1.2. 理解beta分布 [<sup>目录</sup>](#content)</h4></a>

举一个简单的例子，熟悉棒球运动的都知道有一个指标就是棒球击球率(batting average)，就是用一个运动员击中的球数除以击球的总数，我们一般认为0.266是正常水平的击球率，而如果击球率高达0.3就被认为是非常优秀的。

现在有一个棒球运动员，我们希望能够预测他在这一赛季中的棒球击球率是多少。你可能就会直接计算棒球击球率，用击中的数除以击球数，但是如果这个棒球运动员只打了一次，而且还命中了，那么他就击球率就是100%了，这显然是不合理的，因为根据棒球的历史信息，我们知道这个击球率应该是0.215到0.36之间才对啊。

对于这个问题，一个最好的方法来表示这些经验（在统计中称为先验信息）就是用beta分布，这表示在我们没有看到这个运动员打球之前，我们就有了一个大概的范围。beta分布的定义域是(0,1)这就跟概率的范围是一样的。

接下来我们将这些先验信息转换为beta分布的参数，我们知道一个击球率应该是平均0.27左右，而他的范围是0.21到0.35，那么根据这个信息，我们可以取α=81,β=219

之所以取这两个参数是因为：

> - beta分布的均值
> 
> 	<p align="center"><img src=./picture/App-ML-in-Bioinfo-supplementary-knowledge-beta-distribution-1.png height=70 /></p>
> 
> - 这个分布主要落在了(0.2,0.35)间，这是从经验中得出的合理的范围

<p align="center"><img src=./picture/App-ML-in-Bioinfo-supplementary-knowledge-beta-distribution-2.png width=500 /></p>

<p align="center">Beta(81, 219)</p>

在这个例子里，我们的x轴就表示各个击球率的取值，x对应的y值就是这个击球率所对应的概率密度。也就是说beta分布可以看作一个概率的概率密度分布。

那么有了先验信息后，现在我们考虑一个运动员只打一次球，那么他现在的数据就是“1中；1击”。这时候我们就可以更新我们的分布了，让这个曲线做一些移动去适应我们的新信息，移动的方法很简单

<p align="center"><img src=./picture/App-ML-in-Bioinfo-supplementary-knowledge-beta-distribution-3.png height=40 /></p>

其中α0和β0是一开始的参数，在这里是81和219。所以在这一例子里，α增加了1（击中了一次）。β没有增加(没有漏球)。这就是我们的新的beta分布 Beta(81+1,219)，我们跟原来的比较一下：

<p align="center"><img src=./picture/App-ML-in-Bioinfo-supplementary-knowledge-beta-distribution-4.png width=500 /></p>

可以看到这个分布其实没多大变化，这是因为只打了1次球并不能说明什么问题。但是如果我们得到了更多的数据，假设一共打了300次，其中击中了100次，200次没击中，那么这一新分布就是 Beta(81+100,219+200) ： 

<p align="center"><img src=./picture/App-ML-in-Bioinfo-supplementary-knowledge-beta-distribution-5.png width=500 /></p>

注意到这个曲线变得更加尖，并且平移到了一个右边的位置，表示比平均水平要高

有趣的现象：

> 根据这个新的beta分布，我们可以得出他的数学期望为：
>
> $$\frac{a}{a+b}=\frac{82+100}{82+100+219+200}=0.303$$
>
> 这一结果要比直接的估计要小 100/(100+200)=0.333 。你可能已经意识到，我们事实上就是在这个运动员在击球之前可以理解为他已经成功了81次，失败了219次这样一个先验信息。

<a name="posterior-probability-of-beta-distribution"><h4>*1..3.1. 后验概率分布为什么是$Beta(\alpha_0+hit,\beta_0+miss)$ [<sup>目录</sup>](#content)</h4></a>

该运动员击球时间的概率图模型如下图：

<p align="center"><img src=./picture/App-ML-in-Bioinfo-supplementary-knowledge-beta-distribution-6.jpg width=200 /></p>

假设该用户的击球率的分布是一个参数为 $\theta$ 的分布（这里 $\theta$ 既表示一个分布，也是这个分布的参数。因为在概率图模型中，我们经常使用某个分布的参数来代替说明某个模型），也就是说 $\theta$ 是用户击球成功的概率

假设，到目前为止，用户在这个赛季总共打了 $n$ 次球，击中的次数是 $x$，这是一个二项式分布，即 $p(y \mid \theta) = \text{Binomial}(x;n,\theta)$（y表示：总共打了 $n$ 次球，击中的次数是 $x$ 这个事件）

我们的目标就是推导 $\theta$ 分布的形式并估算这个参数的值。这就变成了在贝叶斯推断中的求后验概率的问题了：

$$p(\theta \mid y,\alpha,\beta)=\frac{p(\theta,y \mid \alpha,\beta)}{p(y)}=\frac{p(\theta \mid \alpha,\beta)p(y \mid \theta)}{p(y)}$$

在这里，分母$p(y)$是数据结果，也就是常数。分子第一个项是二项式分布，即 $p(y|\theta)=\theta^{x}(1-\theta)^{(n-x)}$，分子的第二项是$\theta$的先验分布，是一个Beta分布

可以很容易看出，这里的$\theta$的后验分布$p(\theta \mid y,\alpha,\beta)$也是一个Beta分布

$$p(\theta \mid y,\alpha,\beta) \sim Beta(\alpha+x,\beta+n-x)$$




<a name="re-draw-pictrue-in-ng-paper"><h3>*2. 重复NG免疫组库TCRβ文章的图 [<sup>目录</sup>](#content)</h3></a>

<a name="aquire-cmv-positive-cdr3-clone"><h4>*2.1. Fisher检验筛选CMV阳性（CMV<sup>+</sup>）相关克隆 [<sup>目录</sup>](#content)</h4></a>

目的是得到类似下面的结果：

<p align="center"><img src=./picture/App-ML-in-Bioinfo-supplementary-redraw-pictures-for-NGpapre-1.png width=500 /></p>

提供的输入：

> - Observed matrix：行为CDR3克隆，列为样本，矩阵中的值只能取0或1，0表示该feature在该样本中未被观测到，1为被观测到了
> - 样本的分组信息文件（该文件非必须，若样本名中包含分组信息，则可以不提供这个文件，只需提供分从样本名中提取分组信息的正则表达式）
> - 当前感兴趣的两个组，因为考虑到可能提供的样本是多组的，而Fisher检验只能进行两组间的比较，所以多组时需要指定分析的两组是谁

实现的思路：

> 原始提供的输入文件是profile matrix，设定一个阈值（默认为0），将大于阈值的设为1，等于阈值的设为0，从而得到Observed matrix。这一步操作不在下面脚本的功能当中，需要自己在执行下面的脚本之前完成这个操作；
>
> 将准备好的Observed matrix输入，行为TCR克隆（feature），列为sample，因为是要对TCR克隆分析它与分组的相关性，因此每次对矩阵的行执行Fisher检验，根据提供的样本的分组信息，得到该TCR克隆的2X2列联表
>
> $$\begin{matrix}\hline& Group \\ Observe & A & B \\ \hline Yes & a & b \\ \hline No & c & d \\ \hline \end{matrix}$$
>
> 然后再对这个列联表计算Fisher检验的p-value，将TCR克隆，列联表中的a，b和算出的p-value输出，就得到了我们想要的结果

脚本名：`FisherTestForMatrix.R`

```

####################################################################
# 该脚本用于对样本的Oberved矩阵执行Fisher精确检验（单尾），用于 #
# 检测feature与分组的相关性                                       #
####################################################################

# 参数说明：
# - （1）样本的Oberved矩阵，行为特征，列为样本，矩阵中的值只能取0
#	或1，0表示该feature在该样本中未被观测到，1为被观测到了
#
# - （2）是否提供样本的分组信息文件，1为是，0位否
#
# - （3）若上一个参数选择1（提供分组信息文件），则该参数应该设为
#	分组信息文件的路径，分组文件要求至少包含两列——SampleId和Group；
#	若上一个参数选择0（不提供分组信息文件），则认为分组信息已经包
#	含的样本的命名中，则可以通过提供分组信息在样本名中匹配的正则
#	表达式
#
# - （4）感兴趣的两组，例如是A组和B组，则写成"A-B"，即中间用连字
#	符连接
#
# - （5）设置的p值阈值
#
# - （6）设置并行化线程数，若不设置这个参数，则默认不采用并行化计
#	算方法

library(stringr)
library(ggplot2)
library(parallel)

Args <- commandArgs(TRUE)
MatFile <- Args[1]
Bool_GroupFile <- Args[2]
GroupFile <- Args[3]
TargetGroups <- Args[4]
pvalue <- Args[5]

# 开启并行化
if(!is.na(Args[6])){
	no_cores <- as.integer(Args[6])
	cl <- makeCluster(no_cores)
}

Matrix <- read.table(MatFile, header=T, row.names=1)
TargetGroups <- unlist(strsplit(TargetGroups, '-'))

print("Loading Observed Matrix successfully")

#################################################
# 1. 获得Oberved矩阵样本对应的分组信息
#################################################

if(Bool_GroupFile %in% c(1, 0)){
	# 从提供的分组信息文件中获得
	if(Bool_GroupFile==1){
		Group <- read.table(GroupFile, header=T)
		# 考虑到可能存在Observe矩阵列名命名不规范，即以数字起始（会在开头
		#	添加X字符），或其中包含连字符（将连字符替换为点）因此需要将
		#	Group变量中的SampleId进行相应的替换，以保证一致
		colname_matrix <- ifelse(grepl('^X',colnames(Matrix)),sub('^X','',colnames(Matrix)),colnames(Matrix))
		colname_matrix <- gsub('\\.','-',colname_matrix)
		SampleGroup <- Group$Group[match(colname_matrix, Group$SampleId)]
	# 从样本名中用正则提取
	}else{
		GroupPattern <- GroupFile
		SampleGroup <- unlist(str_extract(colnames(Matrix), GroupPattern))
	}
}else{
	stop("参数指定错误！第二个参数必须为0或1")
}

print("Load/get group info for coresponding samples in each col in Observed Matrix")

#################################################
# 2. 对每个feature执行Fisher检验，得到检验结果
#################################################

# 用于执行fisher检验的函数，最终返回的是TRUE或FALSE
FisherTest <- function(ObserveVec, GroupIndex, TargetGroups){
	# 初始化列联表
	FisherMat <- matrix(c(0,0,0,0),
						nrow = 2,
						dimnames = list(Observe=c('Yes','No'),
										Group=TargetGroups
										))
	# 为列联表的每一项填上对应的值
	FisherMat[1,1] <- sum(ObserveVec==1&GroupIndex==colnames(FisherMat)[1])
	FisherMat[1,2] <- sum(ObserveVec==1&GroupIndex==colnames(FisherMat)[2])
	FisherMat[2,1] <- sum(ObserveVec==0&GroupIndex==colnames(FisherMat)[1])
	FisherMat[2,2] <- sum(ObserveVec==0&GroupIndex==colnames(FisherMat)[2])
	# 进行fisher检验，得到p值
	pvalue <- fisher.test(FisherMat, alternative = "two.sided")$p.value
	# 返回向量：组1计数、组2计数、p值
	c(FisherMat[1,1], FisherMat[1,2], pvalue)
}

# 为每个feature（即矩阵的行）执行fisher检验，得到是每个feature的pvalue，每一个的返回值以列形式进行追加
if(!is.na(Args[6])){
	StatOut <- parApply(cl,Matrix, 1, FisherTest, SampleGroup, TargetGroups)
}else{
	StatOut <- apply(Matrix, 1, FisherTest, SampleGroup, TargetGroups)
}
StatOut_Table <- data.frame(Feature=colnames(StatOut),Group1=StatOut[1,],Group2=StatOut[2,],Pval=StatOut[3,])
colnames(StatOut_Table) <- c('Feature', paste("Group_",TargetGroups[1],sep=''), paste("Group_",TargetGroups[2],sep=''), 'Pval')

print("Finish Fisher's Exact Test for each feature")

# 将Fisher检验结果写入文件
write.table(StatOut_Table, paste(str_extract(MatFile,'^(.*?)\\.'),TargetGroups[1],"-",TargetGroups[2],".stat",sep=''),row.names=F,col.names=T,sep="\t",quote=F)

print("Finish writing Fisher's Exact Test Output into file")

#################################################
# 3. 表型负荷（Phenotype Burden）相关的计算与画图
#################################################

# 计算每个样本的表型负荷相关的两个值，该样本中与分组1相关的features数，以及该样本中观测到的features数
if(!is.na(Args[6])){
	RelativeFeatures <- parApply(cl, Matrix[,SampleGroup %in% TargetGroups], 2, function(x,y,p,z) sum(x==1&y<p&z), 
							StatOut_Table$Pval, 
							pvalue, 
							StatOut_Table[,2]>StatOut_Table[,3])
}else{
	RelativeFeatures <- apply(Matrix[,SampleGroup %in% TargetGroups], 2, function(x,y,p,z) sum(x==1&y<p&z), 
							StatOut_Table$Pval, 
							pvalue, 
							StatOut_Table[,2]>StatOut_Table[,3])
}
stopCluster(cl)
TotalFeatures <- colSums(Matrix[,SampleGroup %in% TargetGroups])
PhenotypeBurden <- data.frame(Relative=RelativeFeatures, Total=TotalFeatures, Group=SampleGroup[SampleGroup %in% TargetGroups])

print("Finish calculate statistics for Phenotype-Burden")

# 画散点图
print("Start dotplot")
png(paste("PhenotypeBurdenDot_",pvalue,"_",TargetGroups[1],"-",TargetGroups[2],".png",sep=''))
ggplot(PhenotypeBurden)+geom_point(aes(x=Total, y=Relative, color=Group))
dev.off()
# 保存数据
write.table(PhenotypeBurden,paste(str_extract(MatFile,'^(.*?)\\.'),pvalue,".", TargetGroups[1],"-",TargetGroups[2],".data",sep=''),row.names=T,col.names=T,sep="\t",quote=F)
```

用法：

```
$ Rscript FisherTestForMatrix.R <profile matrix> <1|0> <group info|regx> <2 interested group> <pval>
```

具体的参数说明，请查看脚本的注释信息

<a name="draw-scaterplot-for-phenotype-burden"><h4>*2.2. 绘制表型负荷相关散点图 [<sup>目录</sup>](#content)</h4></a>

<a name="draw-scaterplot-for-phenotype-burden-trainset"><h5>*2.2.1. 训练集 [<sup>目录</sup>](#content)</h5></a>

在上一步 [*2.1. Fisher检验筛选CMV阳性（CMV<sup>+</sup>）相关克隆](aquire-cmv-positive-cdr3-clone) 的操作中会同时得到这张图

画出的表型负荷相关散点图如下：

<p align="center"><img src=./picture/App-ML-in-Bioinfo-supplementary-PhenotypeBurdenDot_0.05_A-B.png width=500 /></p>

<a name="draw-scaterplot-for-phenotype-burden-testset"><h5>*2.2.2. 测试集 [<sup>目录</sup>](#content)</h5></a>

测试集的表型负荷相关散点图需要基于训练集的结果，需要执行另外的操作

需要提供的输入：

> - 测试集的Observed matrix：行为CDR3克隆，列为样本，矩阵中的值只能取0或1，0表示该feature在该样本中未被观测到，1为被观测到了
>
> - 样本的分组信息文件（该文件非必须，若样本名中包含分组信息，则可以不提供这个文件，只需提供分从样本名中提取分组信息的正则表达式）
>
> - 当前感兴趣的两个组，因为考虑到可能提供的样本是多组的，而Fisher检验只能进行两组间的比较，所以多组时需要指定分析的两组是谁
>
> - 训练集的Fisher检验输出结果

脚本名：`PhenotypeBurden2TestCohort.R`

```

###################################################
# 该脚本用于对测试集的每个样本计算表型负荷相关 #
# 的两个值：该样本中与分组1相关的features数，以 #
# 及该样本中观测到的features数                   #
###################################################

# 参数说明：
# - （1）测试集Oberved矩阵，行为特征，列为样本，矩阵中的值只能取0
#	或1，0表示该feature在该样本中未被观测到，1为被观测到了
#
# - （2）是否提供样本的分组信息文件，1为是，0位否
#
# - （3）若上一个参数选择1（提供分组信息文件），则该参数应该设为
#	分组信息文件的路径，分组文件要求至少包含两列——SampleId和Group；
#	若上一个参数选择0（不提供分组信息文件），则认为分组信息已经包
#	含的样本的命名中，则可以通过提供分组信息在样本名中匹配的正则
#	表达式
#
# - （4）感兴趣的两组，例如是A组和B组，则写成"A-B"，即中间用连字
#	符连接
#
# - （5）设置的p值阈值
#
# - （6）上一步对训练集执行Fisher检验的检验结果输出文件

library(stringr)
library(ggplot2)

Args <- commandArgs(TRUE)
MatFile <- Args[1]
Bool_GroupFile <- Args[2]
GroupFile <- Args[3]
TargetGroups <- Args[4]
pvalue <- Args[5]
StatFile <- Args[6]

Matrix <- read.table(MatFile, header=T, row.names=1)
TargetGroups <- unlist(strsplit(TargetGroups, '-'))
StatOut <- read.table(StatFile, header=T)

print("Loading Observed Matrix and Fisher's Exact Test output successfully")

#################################################
# 1. 获得Oberved矩阵样本对应的分组信息
#################################################

if(Bool_GroupFile %in% c(1, 0)){
	# 从提供的分组信息文件中获得
	if(Bool_GroupFile==1){
		Group <- read.table(GroupFile, header=T)
		# 考虑到可能存在Observe矩阵列名命名不规范，即以数字起始（会在开头
		#	添加X字符），或其中包含连字符（将连字符替换为点）因此需要将
		#	Group变量中的SampleId进行相应的替换，以保证一致
		colname_matrix <- ifelse(grepl('^X',colnames(Matrix)),sub('^X','',colnames(Matrix)),colnames(Matrix))
		colname_matrix <- gsub('\\.','-',colname_matrix)
		SampleGroup <- Group$Group[match(colname_matrix, Group$SampleId)]
	# 从样本名中用正则提取
	}else{
		GroupPattern <- GroupFile
		SampleGroup <- unlist(str_extract(colnames(Matrix), GroupPattern))
	}
}else{
	stop("参数指定错误！第二个参数必须为0或1")
}

#################################################
# 2. 表型负荷（Phenotype Burden）相关的计算与画图
#################################################

# 计算每个样本的表型负荷相关的两个值，该样本中与分组1相关的features数，以及该样本中观测到的features数
RelativeFeatures_list <- StatOut$Feature[StatOut[,2]>StatOut[,3]&StatOut$Pval<pvalue]
RelativeFeatures <- apply(Matrix[rownames(Matrix) %in% RelativeFeatures_list,SampleGroup %in% TargetGroups], 2, sum)
TotalFeatures <- colSums(Matrix[,SampleGroup %in% TargetGroups])
PhenotypeBurden <- data.frame(Relative=RelativeFeatures, Total=TotalFeatures, Group=SampleGroup[SampleGroup %in% TargetGroups])

print("Finish calculate statistics for Phenotype-Burden")

# 画散点图
print("Start dotplot")
png(paste("PhenotypeBurdenDot_",pvalue,"_",TargetGroups[1],"-",TargetGroups[2],".png",sep=''))
ggplot(PhenotypeBurden)+geom_point(aes(x=Total, y=Relative, color=Group))+
	labs(title="CMV-associated vs. total-unique")
dev.off()
# 保存数据
write.table(PhenotypeBurden,paste(str_extract(MatFile,'^(.*?)\\.'),pvalue,".", TargetGroups[1],"-",TargetGroups[2],".data",sep=''),row.names=T,col.names=T,sep="\t",quote=F)
```

用法：

```
$ Rscript PhenotypeBurden2TestCohort.R <profile matrix> <1|0> <group info|regx> <2 interested group> <pval> <stat>
```

<a name="scatterplot-showing-the-incidence-bias-of-tcr"><h4>*2.2. TCRβ在两组间分布偏好性的散点图 [<sup>目录</sup>](#content)</h4></a>

就是画这幅图：

<p align="center"><img src=./picture/App-ML-in-Bioinfo-TCRbeta-classification-5.png width=500 /></p>

注意：这幅图的横纵坐标都进行了log10变换

在上一步 [*2.1. Fisher检验筛选CMV阳性（CMV<sup>+</sup>）相关克隆](aquire-cmv-positive-cdr3-clone) 的操作中会同时得到这张图

画出的散点图如下：

<p align="center"><img src=./picture/App-ML-in-Bioinfo-supplementary_TCRβ_incidence_bias.0.05.A-B.png width=500 /></p>

<a name="calculate-fdr"><h3>*3. FDR的计算方法 [<sup>目录</sup>](#content)</h3></a>

<a name="review-those-statistic-methods"><h4>*3.1. 回顾那些统计检验方法 [<sup>目录</sup>](#content)</h4></a>

<a name="t-test-and-moderated-t-Test"><h5>*3.1.1. T-test与Moderated t-Test [<sup>目录</sup>](#content)</h5></a>

t-test的统计量：

$$
t= \frac{\overline X_1(i)-\overline X_2(i)}{S(i)}
$$

Moderated t-Test的统计量：

$$
d= \frac{\overline X_1(i)-\overline X_2(i)}{S(i)+S_0}
$$

Moderated t-Test的统计量d与t-test的t的计算方法很相似，差别就在于分母中方差的计算方法，


| ` ` | T1 | T2 | T3 | C1 | C2 | C3 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gene | XT,1 | XT,2 | XT,3 | XC,1 | XC,2 | XC,3 |

由上面展示的该基因的实际样本分组，计算出方差$S(i)=S_{X_1X_2}\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}$

然后随机打乱上面的样本分组，得到：

| ` ` | T1 | T2 | T3 | C1 | C2 | C3 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gene | XC,2 | XT,1 | XC,3 | XT,3 | XT,2 | XC,1 |

根据打乱的结果算出$S_0$，进行n次这样的随机打乱，计算得到$d_1,d_2,...,d_n$

最后算出它的P值：

$$P=\frac{\#\{d_i\ge d,i=1,2,...,n\}}{n} $$

之所以不用t检验的统计量查表法，是因为Moderated t-Test的统计量已经不再符合某种统计分布了，而且这样算出来的P值也具有一定的统计意义

<a name="necessary-of-multiple-hypothesis-tests"><h4>*3.2. 多重假设检验的必要性 [<sup>目录</sup>](#content)</h4></a>

统计学中的假设检验的基本思路是：

> 设立零假设（null hypothesis）$H_0$，以及与零假设$H_0$相对应的非零假设（alternative hypothesis， or reject null hypothesis）$H_1$，在假设$H_0$成立的前提下，计算出$H_0$发生的概率，若$H_0$的发生概率很低，基于小概率事件几乎不可能发生，所以可以拒绝零假设

但是这些传统的假设检验方法研究的对象，都是一次试验

在一次试验中（注意：是一次试验， 即single test），0.05 或0.01的cutoff足够严格了(想象一下，一个口袋有100个球，95个白的，5个红的, 只让你摸一次，你能摸到红的可能性是多大？)

但是对于多次试验，又称多重假设检验，再使用p值是不恰当的，下面来分析一下为什么：

大家都知道墨菲定律：如果事情有变坏的可能，不管这种可能性有多小，它总会发生

用统计的语言去描述墨菲定律：

> 在数理统计中，有一条重要的统计规律：假设某意外事件在一次实验（活动）中发生的概率为p（p>0），则在n次实验（活动）中至少有一次发生的概率为 $p_n=1-(1-p)^n$
>
> 由此可见，无论概率p多么小（即小概率事件），当n越来越大时，$p_n$越来越接近1

这和我们的一句俗语非常吻合：常在河边走，哪有不湿鞋；夜路走多了，总能碰见鬼

在多重假设检验中，我们一般关注的不再是每一次假设检验的准确性，而是控制在作出的多个统计推断中犯错误的概率，即False Discover Rate（FDR），这对于医院的诊断情景下尤其重要：

> 假如有一种诊断艾滋病(AIDS)的试剂，试验验证其准确性为99%（每100次诊断就有一次false positive）。对于一个被检测的人（single test)）来说，这种准确性够了；但对于医院 （multiple test)）来说，这种准确性远远不够
>
> 因为每诊断10 000个个体，就会有100个人被误诊为艾滋病(AIDS)，每一个误诊就意味着一个潜在的医疗事故和医疗纠纷，对于一些大型医院，一两个月的某一项诊断的接诊数就能达到这个级别，如果按照这个误诊率，医院恐怕得关门，所以医院需要严格控制误诊的数量，宁可错杀一万也不能放过一个，因为把一个没病的病人误判为有病，总比把一个有病的病人误判为没病更保险一些






- 100 independent genes. (We have 100 hypotheses to test)

- No significant differences in gene expression between 2 classes (H0 is true). Thus, the probability that a particular test (say, for gene 1) is declared significant at level 0.05 is exactly 0.05. (Probability of reject H0  in one test if H0 is true = 0.05)

- However, the probability of declaring at least one of the 100 hypotheses false (i.e. rejecting at least one, or finding at least one result significant) is: 

$$1-(1-0.05)^{100}\approx 0.994$$

<a name="distinguish-pvalue-and-qvalue"><h4>*3.3. 区别p值和q值 [<sup>目录</sup>](#content)</h4></a>

| ` ` | $H_0$ is true | $H_1$ is true | Total |
|:---:|:---:|:---:|:---:|
| Not Significant | TN | FN | TN+FN |
| Significant | FP | TP | FP+TP |
| Total | TN+FP | FN+TP | m |

首先从上面的混淆矩阵来展示p值域q值的计算公式，就可以看出它们之间的区别：

> - **p值**
>
>	p值实际上就是false positive rate(FPR，假正率)：
>
>	$$p-value=FPR=\frac{FP}{FP+TN}$$
>
>	直观来看，p值是用上面混淆矩阵的**第一列**算出来的
>
> - **q值**
>
>	q值实际上就是false discovery rate (FDR)：
>
>	$$q-value=FDR=\frac{FP}{FP+TP}$$
>
>	直观来看，q值是用上面混淆矩阵的**第二行**算出来的

但是仅仅知道它俩的计算公式的差别还不够，我们还有必要搞清楚一个问题：它俩在统计学意义上有什么不同呢？

> p值衡量的是一个原本应该是$H_0$的判断被错误认为是$H_1 \, (reject H_0)$的比例，所以它是针对单次统计推断的一个置信度评估；
>
> q值衡量的是在进行多次统计推断后，在所有被判定为显著性的结果里，有多大比例是误判的

据此，我们可以推导出p值域q值之间的关系：

> 总共有n个features(可以是基因，GWAS中的snp位点等)，对它们知道n重假设假设检验后，得到各自对应的p值分别为$\{p^{(i)} \mid i=1,2,...,n\}$
> 
> 当p值显著性水平取$\alpha$时，得到$k$个features具有p值显著性，它们的p值为$\{p^{(i)}_{(j)} \mid j=1,2,...,k\}$，其中$p^{(i)}_{(j)}$表示第i个feature它的p值在升序中的排名为j，那么这k个features的FDR可以表示为：
>
> $$FDR=1-\prod_{j=1}^{k}(1-p^{(i)}_{(j)})$$



<a name="how-to-calculate-fdr"><h4>*3.4. 如何计算FDR？ [<sup>目录</sup>](#content)</h4></a>

统计检验的混淆矩阵：

| ` ` | $H_0$ is true | $H_1$ is true | Total |
|:---:|:---:|:---:|:---:|
| Significant | V | S | R |
| Not Significant | U | T | m-R |
| Total | m<sub>0</sub> | m-m<sub>0</sub> | m |

- **FWER (Family Wise Error Rate)**

	作出一个或多个假阳性判断的概率

	$$FWER=Pr(V\ge 1)$$

	使用这种方法的统计学过程：

	- The Bonferroni procedure
	- Tukey's procedure
	- Holm's step-down procedure 

- **FDR (False Discovery Rate)**

	在所有的单检验中作出假阳性判断比例的期望

	$$FDR=E\left[\frac{V}{R}\right]$$

	使用这种方法的统计学过程：

	- Benjamini–Hochberg procedure
	- Benjamini–Hochberg–Yekutieli procedure

<a name="calculate-fdr-by-benjamini-procedure"><h5>*3.4.1. Benjamini-Hochberg procedure (BH) [<sup>目录</sup>](#content)</h5></a>

对于m个独立的假设检验，它们的P-value分别为：$p_i,i=1,2,...,m$

（1）按照升序的方法对这些P-value进行排序，得到：

$$p_{(1)} \le p_{(2)} \le ... \le p_{(m)}$$

（2）对于给定是统计显著性值$\alpha \in (0,1)$，找到最大的k，使得

$$p_{(k)} \le \frac{\alpha * k}{m}$$

（3）对于排序靠前的k个假设检验，认为它们是真阳性 (positive )

即：$reject \, H_0^{(i)},\, 1 \le i \le k$ 

$$
\begin{array}{c|l}
\hline
Gene & p-value \\
\hline
G1 & P1 =0.053 \\
\hline
G2 & P2 =0.001 \\
\hline
G3	& P3 =0.045 \\
\hline
G4	& P4 =0.03 \\
\hline
G5 & P5 =0.02 \\
\hline
G6 & P6 =0.01 \\
\hline
\end{array}
\, \Rightarrow \,
\begin{array}{c|l}
\hline
Gene & p-value \\
\hline
G2	& P(1) =0.001 \\
\hline
G6	& P(2) =0.01 \\
\hline
G5	& P(3) =0.02 \\
\hline
G4	& P(4) =0.03 \\
\hline
G3	& P(5) =0.045 \\
\hline
G1	& P(6) =0.053 \\
\hline
\end{array}
$$

<br>

$$\alpha = 0.05$$

> $P(4) =0.03<0.05*\frac46=0.033$
>
> $P(5) =0.045>0.05*\frac56=0.041$
>
> 因此最大的k为4，此时可以得出：在FDR<0.05的情况下，G2，G6，G5 和 G4 存在差异表达

可以计算出q-value：

$$p_{(k)} \le \frac{\alpha * k}{m} \, \Rightarrow \frac{p_{(k)}*m}{k} \le \alpha$$

<br>

| Gene | P | q-value |
|:---:|:---|:---|
| G2 | P(1) =0.001 | 0.006 |
| G6 | P(2) =0.01 | 0.03 |
| G5 | P(3) =0.02 | 0.04 |
| G4 | P(4) =0.03 | 0.045 |
| G3 | P(5) =0.045 | 0.054 |
| G1 | P(6) =0.053 | 0.053 |

根据q-valuea的计算公式，我们可以很明显地看出：

$$q^{(i)}=p_{(k)}^{(i)}*\frac{Total \, Gene \, Number}{rank(p^{(i)})}==p_{(k)}^{(i)}*\frac{m}{k}$$

即，根据该基因p值的排序对它进行放大，越靠前放大的比例越大，越靠后放大的比例越小，排序最靠后的基因的p值不放大，等于它本身

我们也可以从可视化的角度来看待这个问题：

对于给定的$\alpha \in (0,1)$，设函数$y=\frac{\alpha}{m}x \quad (x=1,2,...,m)$，画出这条线，另外对于每个基因，它在图上的坐标为$(rank(p_{(k)}^{(i)}),p_{(k)}^{(i)})=(k,p_{(k)}^{(i)})$，图如下：

<p align="center"><img src=./picture/App-ML-in-Bioinfo-supplementary-calculate-FDR.png width=600 /></p>

通过设置$\alpha$可以改变图中直线的斜率，$\alpha$越大，则直线的斜率越大，落在直线下方的点就越多，通过FDR检验的基因也就越多，反之，直线的斜率越小，落在直线下方的点就越少，通过FDR检验的基因也就越少

不知道大家看到这里有没有产生这样的疑惑：

> 在上文 [*3.3. 区别p值和q值](#distinguish-pvalue-and-qvalue) 中已经推导出了q值与p值的理论上的表达关系式，如下：
>
> $$q-value=FDR=1-\prod_{j=1}^{k}(1-p^{(i)}_{(j)})$$
>
> 那么对于按照升序方法进行排序的p值序列$\{p^{(i)}_{(j)} \mid j=1,2,...,n\}$，我们可以算出当$p^{(i)}_{(j)} \le \alpha$，得到k个具有统计学显著性的feature时的q值，记为$q_{k}$，表示的是当取p值最小的前k个features，判定它们是显著时的q值（或FDR）

也就是说，可以直接用上面的公式算出来q值，而且这个公式的统计学意义也非常清楚，为什么不直接用这个公式去算q值，而要人为的去再提出一个新的计算方法，而且这个公式的统计学意义还不是很容易理解

<a name="calculate-fdr-by-bonferroni-correction"><h5>*3.4.2. Bonferroni 校正 [<sup>目录</sup>](#content)</h5></a>

Bonferroni 校正的基本思想：

> 如果在同一数据集上同时检验n个独立的假设，那么用于每一假设的统计显著水平，应为仅检验一个假设时的显著水平的1/n
>
> 举个例子：如要在同一数据集上检验两个独立的假设，显著水平设为常见的0.05，此时用于检验该两个假设应使用更严格的0.025，即0.05* (1/2)

序列化的 Bonferroni 校正步骤：

> 对k个独立的检验，在给定的显著性水平α下，把每个检验对应的 P 值从小到大排列
>
> $$p_{(1)} \le p_{(2)} \le ... \le p_{(k)}$$
>
> 首先看最小的 P 值 $p_{(1)}$，如果$p_{(1)} \le \frac{\alpha}{k}$，就认为对应的检验在总体上（table wide）α水平上显著；如果不是，就认为所有的检验都不显著；
>
> 当且仅当 $p_{(1)} \le \frac{\alpha}{k}$ 时，再来看第二个P值$p_{(2)}$。如果$p_{(2)} \le \frac{\alpha}{k-1}$，就认为在总体水平上对应的检验在α水平上是显著的；
>
> 之后再进行下一个P值……一直进行这个过程，直到 $p_{(i)} \le \frac{\alpha}{k-i+1}$不成立；下结论i和以后的检验都不显著





---

参考资料：

(1)  Emerson R O , Dewitt W S , Vignali M , et al. Immunosequencing identifies signatures of cytomegalovirus exposure history and HLA-mediated effects on the T cell repertoire[J]. Nature Genetics, 2017, 49(5):659-665.

(2) Kiselev, V. Y. et al. SC3: consensus clustering of single-cell RNA-seq data[J]. Nat. Methods 14, 483–486 (2017).

(3) [CSDN·chivalry《二项分布和Beta分布》](https://blog.csdn.net/sunmenggmail/article/details/17153053)

(4) [CSDN·Jie Qiao《带你理解beta分布》](https://blog.csdn.net/a358463121/article/details/52562940)

(5) [贝塔分布（Beta Distribution）简介及其应用](https://www.datalearner.com/blog/1051505532393058)

(6) [StatLect《Beta distribution》](https://www.statlect.com/probability-distributions/beta-distribution)

(7)  Storey, J.D. & Tibshirani, R. Statistical signifcance for genomewide studies.Proc. Natl. Acad. Sci. USA 100, 9440–9445 (2003)

(8) 国科大研究生课程《生物信息学》，陈小伟《基因表达分析》

(9) [新浪博客·菜鸟《Bonferroni 校正》](http://blog.sina.com.cn/s/blog_4af3f0d20100bzx9.html)

(10) [简书·Honglei_Ren《多重检验中的FDR错误控制方法与p-value的校正及Bonferroni校正》](https://www.jianshu.com/p/a262cf3d18b9)
