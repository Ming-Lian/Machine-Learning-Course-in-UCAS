<a name="content">目录</a>

[期望最大化算法（EM算法）](#title)
- [1. EM算法的基本思想](#principle)
- [2. EM算法的数学推导](#math-derivation)
- [3. EM算法的实际应用情景](#applications)
	- [3.1. HMM的参数估计：鲍姆-韦尔奇算法](#baulm-werch)









<h1 name="title">期望最大化算法（EM算法）</h1>

<a name="principle"><h2>1. EM算法的基本思想 [<sup>目录</sup>](#)</h2></a>

要解决的问题：

> 我们经常会从样本观察数据中，找出样本的模型参数。 最常用的方法就是极大化模型分布的对数似然函数。
> 
> 但是在一些情况下，我们得到的观察数据有未观察到的隐含数据，此时我们未知的有隐含数据和模型参数，因而无法直接用极大化对数似然函数得到模型分布的参数。怎么办呢？这就是EM算法可以派上用场的地方了。

EM算法的解决策略：

> EM算法解决这个的思路是使用**启发式的迭代方法**
> 
> 既然我们无法直接求出模型分布参数，那么我们可以先猜想隐含数据（EM算法的E步），接着基于观察数据和猜测的隐含数据一起来极大化对数似然，求解我们的模型参数（EM算法的M步)。由于我们之前的隐藏数据是猜测的，所以此时得到的模型参数一般还不是我们想要的结果。不过没关系，我们基于当前得到的模型参数，继续猜测隐含数据（EM算法的E步），然后继续极大化对数似然，求解我们的模型参数（EM算法的M步)。以此类推，不断的迭代下去，直到模型分布参数基本无变化，算法收敛，找到合适的模型参数。

从上面的描述可以看出，EM算法是迭代求解最大值的算法，同时算法在每一次迭代时分为两步，**E步和M步**。一轮轮迭代更新隐含数据和模型分布参数，直到收敛，即得到我们需要的模型参数。

EM算法的直观理解的例子——**K-Means算法**：

> 一个最直观了解EM算法思路的是K-Means算法，见之前写的K-Means聚类算法原理。在K-Means聚类时，每个聚类簇的**质心是隐含数据**。我们会假设K个初始化质心，即EM算法的E步；然后计算得到每个样本最近的质心，并把样本聚类到最近的这个质心，即EM算法的M步。重复这个E步和M步，直到质心不再变化为止，这样就完成了K-Means聚类。

当然，K-Means算法是比较简单的，实际中的问题往往没有这么简单。上面对EM算法的描述还很粗糙，我们需要用数学的语言精准描述。

<a name="math-derivation"><h2>2. EM算法的数学推导 [<sup>目录</sup>](#)</h2></a>

对于m个样本观察数据x=(x<sup>(1)</sup>,x<sup>(2)</sup>,...x<sup>(m)</sup>)中，找出样本的模型参数θ, 极大化模型分布的对数似然函数如下：

<p align="center"><img src=./picture/EM-Algorithm-math-derivation-1.png height=60 />

如果我们得到的观察数据有未观察到的隐含数据z=(z<sup>(1)</sup>,z<sup>(2)</sup>,...z<sup>(m)</sup>)，其中隐含数据的类型空间为Z<sup>(j)</sup>, j=1,2,...,n，此时我们的极大化模型分布的对数似然函数如下：

<p align="center"><img src=./picture/EM-Algorithm-math-derivation-2.png height=60 />

注：

> 这里解释一下等式最后部分的式子是如何推出来的，即以下公式为什么成立：
> 
> <p align="center"><img src=./picture/EM-Algorithm-math-derivation-3.png height=50 />
> 
> 其实这就是简单的**全概率公式**的应用：
> 
> <p align="center"><img src=./picture/EM-Algorithm-math-derivation-4.png height=50 />

上面这个式子是没有办法直接求出θ的。因此需要一些特殊的技巧，我们首先对这个式子进行缩放如下：

<p align="center"><img src=./picture/EM-Algorithm-math-derivation-5.png height=120 />

上面第(1)式引入了一个未知的新的分布Qi(z(i))，第(2)式用到了Jensen不等式：

<p align="center"><img src=./picture/EM-Algorithm-math-derivation-6.png height=60 />

对Jensen不等式的直观理解：

> 对于凸函数（抛物线开口向下，线条向上凸起），`f(E(x)) ≥ E(f(x))`
> 
> 因为对数函数是凸函数所以，Jensen不等式成立，Jensen不等式是由这个性质的推出的一个定量而已
> 
> 下面举一个简单的例子来证明：
> 
> 
> 对于凸函数 f(x)，x<sub>1</sub> < x<sub>2</sub>，则如下图：
> 
> <p align="center"><img src=./picture/EM-Algorithm-math-derivation-7.png width=500 />
> 
> 所以，`f(E(x)) ≥ E(f(x))`

**第(2)式是我们的包含隐藏数据的对数似然的一个下界。如果我们能极大化这个下界，则也在尝试极大化我们的对数似然。**

但是，这里又冒出了一个新的问题：我们在前面的操作中引入了未知的新的分布Q<sub>i</sub>(z<sup>(i)</sup>)，那**如何求出Q<sub>i</sub>(z<sup>(i)</sup>)的表达式呢**？

理论上，随便取Q<sub>i</sub>(z<sup>(i)</sup>)带入（2）作为（1）的下界，然后这个下界每一次迭代都增大，而且满足这样条件的Q<sub>i</sub>(z<sup>(i)</sup>)可能有很多，但是获得Q<sub>i</sub>(z<sup>(i)</sup>)表达式最简便的方法是利用Jensen不等式取等的条件

**求解Q<sub>i</sub>(z<sup>(i)</sup>)表达式：**

> 如果要满足Jensen不等式的等号，则有(该式记为公式①）：
> 
> <p align="center"><img src=./picture/EM-Algorithm-math-derivation-8.png height=70 />
> 
> 由于Qi(z(i))是一个分布，所以满足(该式记为公式②）：
> 
> <p align="center"><img src=./picture/EM-Algorithm-math-derivation-9.png height=60 />
> 
> 将①代入②中得(该式记为公式③）：
> 
> <p align="center"><img src=./picture/EM-Algorithm-math-derivation-10.png height=50 />
> 
> 则
> 
> <p align="center"><img src=./picture/EM-Algorithm-math-derivation-11.png height=70 />
> 

因此我们的优化目标变成了

<p align="center"><img src=./picture/EM-Algorithm-math-derivation-12.png height=120 />
 
由于目标函数中

<p align="center"><img src=./picture/EM-Algorithm-math-derivation-13.png height=120 />

则前面的目标函数可以简化为：

<p align="center"><img src=./picture/EM-Algorithm-math-derivation-14.png height=100 />



<a name="applications"><h2>3. EM算法的实际应用情景 [<sup>目录</sup>](#)</h2></a>



<a name="baulm-werch"><h3>3.1. HMM的参数估计：鲍姆-韦尔奇算法 [<sup>目录</sup>](#)</h3></a>









---

参考资料：

(1) [【cnblogs】EM算法原理总结](http://www.cnblogs.com/pinard/p/6912636.html)
