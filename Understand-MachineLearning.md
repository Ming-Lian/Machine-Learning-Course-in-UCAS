<a name="content">目录</a>

[理解机器学习](#title)
- [1. 理解极大似然估计和EM算法](#maximize-likelihood-estimation-and-expection-maximum)
    - [1.1. 搞懂极大似然估计与EM算法定义及理清它们间的关系](#make-sense-the-defination-and-relationship-of-MLE-EM)
    - [1.2. EM算法的优化思想](#principle-of-em-algorithmn)
    - [1.3. 类比损失函数最小化与梯度下降法](#analogy-mle-em-with-minimize-loss-function-and-gradient-descent)

<h1 mame="title">理解机器学习</h1>

<a name="maximize-likelihood-estimation-and-expection-maximum"><h2>1. 理解极大似然估计和EM算法 [<sup>目录</sup>](#content)</h2></a>

<a name="make-sense-the-defination-and-relationship-of-MLE-EM"><h3>1.1. 搞懂极大似然估计与EM算法定义及理清它们间的关系 [<sup>目录</sup>](#content)</h3></a>

---

写在最前面的结论：

极大似然估计和EM算法都是对于概率模型而言的

极大似然是对概率模型参数学习优化目标的一种定义

EM算法是用于求解极大似然估计的一种迭代逼近的算法

---


机器学习的一大重要任务：

> 根据一些已观察到的证据（例如训练样本）来对感兴趣的未知变量（例如类别标记）进行估计

那么对于上面的任务，一种有效的实现途径就是使用**概率模型**

概率模型的本质是，将机器学习任务归结为计算变量的概率，其核心是基于可观测变量推断出位置变量的条件分布

> 假定所关心的变量的集合为Y，可观测变量集合为O，不可观测变量集合为I，则完全变量$U=\{O,I\}$，且$Y\subset U$
>
> 则基于概率模型的机器任务可以形式化地表示为
>
> $$Y^*=arg \max_Y P(Y \mid O,\theta)$$
>
> 即，对于已经训练好的模型$\theta$，给定一组观测值O，且已知感兴趣的未知变量Y的可能取值范围，计算出所以可能的$P(Y \mid O,\theta)$，那个概率最大的Y就是模型给出的判断

那么，如何从预先给出的训练数据集学习出概率模型的参数$\theta$呢？

（1）定义优化目标——**极大似然估计**

一般使用**极大似然估计**

极大似然估计 (maximize likelihood estimation, MLE)：

> 在概率模型中，给定观测数据作$O$为模型的训练样本，训练出对应的概率模型，即得到模型的参数$\theta$，使得基于此模型的产生观测数据的条件概率（称为似然）$P(O\mid \theta)$最大化
>
> 所以极大似然估计是概率模型的一种优化方法，可以表示为
>
> $$\theta^*=arg \max_{\theta}P(O \mid \theta)$$

前面已经提到了，在概率模型中，有时既含有可观测变量 (observable variable) O，又含有隐藏变量或潜在变量(latent variable) I

- 若只含有观测变量

    比如，贝叶斯分类器，可以不仅观测到每个样本的features（假设所有的与模型相关的features都被观测到了），还可以知道每个样本所属的类别，则没有隐变量，即$U=\{O,I\},而I=\emptyset,则U=O$

    则此时，给定数据，可以直接用极大似然估计法来估计模型参数

    $$\theta^*=arg \max_Y P(O \mid \theta)$$

- 若含有隐变量

    比如，隐马尔可夫模型 (hidden markov model, HMM)

    ![](./picture/Understand-MachineLearning-MLE-EM-1.png)

    则此时的优化目标仍然是极大似然估计(MLE)，但是是含有隐变量的极大似然估计，即

    $$\theta^*=arg \max_Y P(O \mid \theta)=arg \max_Y \sum_I P(O,I\mid \theta)$$

（2）求解优化目标

上面我们已经谈论了概率模型参数学习的一般步骤的第一步，即用极大似然估计来定义我们的优化目标

且对于可观测数据的不同，把极大似然估计分成了两种情况，即观测数据是完全数据$O \subseteq U$的情况，和观测数据是不完全数据$O\subsetneq U$的情况

$$
\theta^*=
\begin{cases}
arg \max_Y P(O \mid \theta) , &  if \quad O \subseteq U \\
arg \max_Y \sum_I P(O,I\mid \theta), & if \quad O\subsetneq U
\end{cases}
$$

而对于完全数据和不完全数据，极大似然估计的求解难度是不一样的：

- 对于完全数据，它的优化目标中只含有待求解的$\theta$，使用常规的优化目标求解法即可，例如梯度下降、拟牛顿法，或者直接导数法也可以；

- 对于不完全数据，其中有隐变量I的干扰，所以要想直接求解$\theta$是无法做到的

那么怎么解决含有隐含变量的极大似然估计？

这便是EM算法要做的事

<a name="principle-of-em-algorithmn"><h3>1.2. EM算法的优化思想 [<sup>目录</sup>](#content)</h3></a>


<a name="analogy-mle-em-with-minimize-loss-function-and-gradient-descent"><h3>1.3. 类比损失函数最小化与梯度下降法 [<sup>目录</sup>](#content)</h3></a>

---

参考资料：

(1) 周志华《机器学习》第14章《概率图模型》

(2) 李航《统计学习方法》第9章《EM算法》

(3) 吴军《数学之美》第5章《隐马尔科夫模型》和第27章《上帝的算法——期望最大化算法》
