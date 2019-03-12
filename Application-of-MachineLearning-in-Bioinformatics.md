<a name="content">目录</a>

[机器学习在生物信息中的应用](#title)
- [1. 根据免疫组库TCRβ预测病人的CMV感染状态](#predict-cmv-serostatus-using-tcr)
- [补充知识](#supplementary-knowledge)
	- [*1. beta分布](#beta-distribution)
		- [*1.1. 什么是beta分布](#what-is-beta-distribution)
		- [*1.2. 理解beta分布](#understand-beta-distribution)






<h1 name="title">机器学习在生物信息中的应用</h1>

<a name="predict-cmv-serostatus-using-tcr"><h2>根据免疫组库TCRβ预测病人的CMV感染状态 [<sup>目录</sup>](#content)</h2></a>

- 基于单特征的贝叶斯判别模型

	首先，鉴定出那些在CMV感染状态表现为阳性的（phenotype-positive）患者中出现明显克隆扩张的TCRβs，称为阳性表型相关性TCRβs克隆
	
	一个样本的表型负荷（phenotype burden）被定义为：

	> 该样本的所有unique TCRβs中与阳性表型相关的克隆的数量所占的比例

	然后分别对表型阳性和表型阴性两组样本，利用Beta分别（补充知识——[Beta分布](#beta-distribution)）估计出各组的表型负荷的概率密度分布，分别记作 Beta<sub>+</sub>(α<sub>+</sub> , β<sub>+</sub>) 和 Beta<sub>-</sub>(α<sub>-</sub> , β<sub>-</sub>)

	<p align="center"><img src=/picture/App-ML-in-Bioinfo-TCRbeta-classification-1.png height=100 />

	对于一个新的样本可以计算出它来自表型阳性组和表型阴性组的后验概率：

	<p align="center"><img src=/picture/App-ML-in-Bioinfo-TCRbeta-classification-2.png height=100 />

	若P ( c<sub>+</sub> | x <sub>i</sub> ) > P ( c <sub>-</sub> | x <sub>i</sub> )，则判断为阳性组，否则为阴性组








<a name="supplementary-knowledge"><h2>补充知识 [<sup>目录</sup>](#content)</h2></a>

<a name="beta-distribution"><h3>*1. beta分布 [<sup>目录</sup>](#content)</h3></a>

<a name="what-is-beta-distribution"><h4>*1.1. 什么是beta分布 [<sup>目录</sup>](#content)</h4></a>

对于硬币或者骰子这样的简单实验，我们事先能很准确地掌握系统成功的概率。

然而通常情况下，系统成功的概率是未知的，但是根据频率学派的观点，我们可以**通过频率来估计概率**。为了测试系统的成功概率，我们做n次试验，统计成功的次数k，于是很直观地就可以计算出。然而由于系统成功的概率是未知的，这个公式计算出的只是系统成功概率的**最佳估计**。也就是说实际上也可能为其它的值，只是为其它的值的概率较小。因此我们并不能完全确定硬币出现正面的概率就是该值，所以也是一个随机变量，它符合Beta分布，其取值范围为0到1

用一句话来说，beta分布可以看作一个**概率的概率密度分布**，当你不知道一个东西的具体概率是多少时，它可以给出了所有概率出现的可能性大小。

Beta分布有和两个参数α和β，其中α为成功次数加1，β为失败次数加1。

<a name="understand-beta-distribution"><h4>*1.2. 理解beta分布 [<sup>目录</sup>](#content)</h4></a>

举一个简单的例子，熟悉棒球运动的都知道有一个指标就是棒球击球率(batting average)，就是用一个运动员击中的球数除以击球的总数，我们一般认为0.266是正常水平的击球率，而如果击球率高达0.3就被认为是非常优秀的。

现在有一个棒球运动员，我们希望能够预测他在这一赛季中的棒球击球率是多少。你可能就会直接计算棒球击球率，用击中的数除以击球数，但是如果这个棒球运动员只打了一次，而且还命中了，那么他就击球率就是100%了，这显然是不合理的，因为根据棒球的历史信息，我们知道这个击球率应该是0.215到0.36之间才对啊。

对于这个问题，一个最好的方法来表示这些经验（在统计中称为先验信息）就是用beta分布，这表示在我们没有看到这个运动员打球之前，我们就有了一个大概的范围。beta分布的定义域是(0,1)这就跟概率的范围是一样的。
s
接下来我们将这些先验信息转换为beta分布的参数，我们知道一个击球率应该是平均0.27左右，而他的范围是0.21到0.35，那么根据这个信息，我们可以取α=81,β=219

之所以取这两个参数是因为：

> - beta分布的均值
> 
> 	<p align="center"><img src=/picture/App-ML-in-Bioinfo-supplementary-knowledge-beta-distribution-1.png height=70 /></p>
> 
> - 这个分布主要落在了(0.2,0.35)间，这是从经验中得出的合理的范围

<p align="center"><img src=/picture/App-ML-in-Bioinfo-supplementary-knowledge-beta-distribution-2.png width=500 /></p>

<p align="center">Beta(81, 219)</p>

在这个例子里，我们的x轴就表示各个击球率的取值，x对应的y值就是这个击球率所对应的概率密度。也就是说beta分布可以看作一个概率的概率密度分布。

那么有了先验信息后，现在我们考虑一个运动员只打一次球，那么他现在的数据就是“1中；1击”。这时候我们就可以更新我们的分布了，让这个曲线做一些移动去适应我们的新信息，移动的方法很简单

<p align="center"><img src=/picture/App-ML-in-Bioinfo-supplementary-knowledge-beta-distribution-3.png height=40 /></p>

其中α0和β0是一开始的参数，在这里是81和219。所以在这一例子里，α增加了1（击中了一次）。β没有增加(没有漏球)。这就是我们的新的beta分布 Beta(81+1,219)，我们跟原来的比较一下：

<p align="center"><img src=/picture/App-ML-in-Bioinfo-supplementary-knowledge-beta-distribution-4.png width=500 /></p>

可以看到这个分布其实没多大变化，这是因为只打了1次球并不能说明什么问题。但是如果我们得到了更多的数据，假设一共打了300次，其中击中了100次，200次没击中，那么这一新分布就是 Beta(81+100,219+200) ： 

<p align="center"><img src=/picture/App-ML-in-Bioinfo-supplementary-knowledge-beta-distribution-5.png width=500 /></p>

注意到这个曲线变得更加尖，并且平移到了一个右边的位置，表示比平均水平要高

有趣的现象：

> 根据这个新的beta分布，我们可以得出他的数学期望为：αα+β=82+10082+100+219+200=.303 ，这一结果要比直接的估计要小 100100+200=.333 。你可能已经意识到，我们事实上就是在这个运动员在击球之前可以理解为他已经成功了81次，失败了219次这样一个先验信息。





---

参考资料：

(1)  Emerson R O , Dewitt W S , Vignali M , et al. Immunosequencing identifies signatures of cytomegalovirus exposure history and HLA-mediated effects on the T cell repertoire[J]. Nature Genetics, 2017, 49(5):659-665.

(2) [CSDN·chivalry《二项分布和Beta分布》](https://blog.csdn.net/sunmenggmail/article/details/17153053)

(3) [CSDN·Jie Qiao《带你理解beta分布》](https://blog.csdn.net/a358463121/article/details/52562940)
