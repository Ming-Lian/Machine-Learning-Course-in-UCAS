<a name="content">目录</a>

[Andrew Ng：Coursera机器学习课程](#title)
- [1. 机器学习导论](#introduction)
- [2. 线性代数基础](#linear-algebra)
- [3. Octave教程](#learning-octave)
- [4. 线性回归](#linear-regression)
	- [4.1. 模型描述](#linear-regression-model-description)
	- [4.2. 代价函数](#linear-regression-cost-function)
	- [4.3. 梯度下降](#linear-regression-gradient-descent)
		- [4.3.1. 梯度下降描述](#linear-regression-description-gradient-descent)
		- [4.3.2. 用梯度下降求解线性回归问题](#linear-regression-usage-gradient-descent)
		- [4.3.3. 梯度下降中的实用技巧](#linear-regression-technique-gradient-descent)
			- [4.3.3.1. 特征缩放(feature scaling)](#feature-scaling)
			- [4.3.3.2. 均值归一化(mean normalization)](#mean-normalization)
			- [4.3.3.3. 学习率α](#learning-rate)
	- [4.4. 多项式回归](#polynomial-regression)
	- [4.5. 其他解法：正规方程法](#linear-regression-normal-equation)
- [5. 分类问题](#classification)
	- [5.1. logistic回归](#logistic-regression)
		- [5.1.1. 问题描述](#logistic-regression-description)
		- [5.1.2. 理解决策边界](#logistic-regression-decision-boundary)
		- [5.1.3. 拟合logistic回归](#logistic-regression-fit)
			- [5.1.3.1. 确定损失函数](#logistic-regression-cost-function)
			- [5.1.3.2. 梯度下降求解](#logistic-regression-solve-fit-using-gradient-descent)
			- [5.1.3.3. 高级优化算法简介与使用](#logistic-regression-advanced-optimal-algorithmn)





<h1 name="title">Andrew Ng：Coursera机器学习课程</h1>

<a name="introduction"><h2>1. 机器学习导论 [<sup>目录</sup>](#content)</h2></a>

<p align="center"><img src=./picture/AndrewNG-ML-001.jpg width=800 /></p>

<a name="linear-algebra"><h2>2. 线性代数基础 [<sup>目录</sup>](#content)</h2></a>

<p align="center"><img src=./picture/AndrewNG-ML-009-2.jpg width=800 /></p>

<p align="center"><img src=./picture/AndrewNG-ML-010.jpg width=800 /></p>

<a name="learning-octave"><h2>3. Octave教程 [<sup>目录</sup>](#content)</h2></a>

<p align="center"><img src=./picture/AndrewNG-ML-016-2.jpg width=800 /></p>

<p align="center"><img src=./picture/AndrewNG-ML-017.jpg width=800 /></p>

<a name="linear-regression"><h2>4. 线性回归 [<sup>目录</sup>](#content)</h2></a>

<p align="center"><img src=./picture/AndrewNG-ML-002.jpg width=800 /></p>

<a name="linear-regression-model-description"><h3>4.1. 模型描述 [<sup>目录</sup>](#content)</h3></a>

<p align="center"><img src=./picture/AndrewNG-ML-003.jpg width=800 /></p>

<a name="linear-regression-cost-function"><h3>4.2. 代价函数 [<sup>目录</sup>](#content)</h3></a>

<p align="center"><img src=./picture/AndrewNG-ML-004.jpg width=800 /></p>

<p align="center"><img src=./picture/AndrewNG-ML-005.jpg width=800 /></p>

<a name="linear-regression-gradient-descent"><h3>4.3. 梯度下降 [<sup>目录</sup>](#content)</h3></a>

<a name="linear-regression-description-gradient-descent"><h4>4.3.1. 梯度下降描述 [<sup>目录</sup>](#content)</h4></a>

<p align="center"><img src=./picture/AndrewNG-ML-006.jpg width=800 /></p>

<p align="center"><img src=./picture/AndrewNG-ML-007.jpg width=800 /></p>

<a name="linear-regression-usage-gradient-descent"><h4>4.3.2. 用梯度下降求解线性回归问题 [<sup>目录</sup>](#content)</h4></a>

<p align="center"><img src=./picture/AndrewNG-ML-008.jpg width=800 /></p>

<p align="center"><img src=./picture/AndrewNG-ML-009-1.jpg width=800 /></p>

<a name="linear-regression-technique-gradient-descent"><h4>4.3.3. 梯度下降中的实用技巧 [<sup>目录</sup>](#content)</h4></a>

<a name="feature-scaling"><h4>4.3.3.1. 特征缩放(feature scaling) [<sup>目录</sup>](#content)</h4></a>

<p align="center"><img src=./picture/AndrewNG-ML-011.jpg width=800 /></p>

<a name="mean-normalization"><h4>4.3.3.2. 均值归一化(mean normalization) [<sup>目录</sup>](#content)</h4></a>

<p align="center"><img src=./picture/AndrewNG-ML-012-1.jpg width=800 /></p>

<a name="learning-rate"><h4>4.3.3.3. 学习率α [<sup>目录</sup>](#content)</h4></a>

<p align="center"><img src=./picture/AndrewNG-ML-012-2.jpg width=800 /></p>

<p align="center"><img src=./picture/AndrewNG-ML-013-1.jpg width=800 /></p>

<a name="polynomial-regression"><h3>4.4. 多项式回归 [<sup>目录</sup>](#content)</h3></a>

<p align="center"><img src=./picture/AndrewNG-ML-013-2.jpg width=800 /></p>

<a name="linear-regression-normal-equation"><h3>4.5. 其他解法：正规方程法 [<sup>目录</sup>](#content)</h3></a>

<p align="center"><img src=./picture/AndrewNG-ML-014.jpg width=800 /></p>

<p align="center"><img src=./picture/AndrewNG-ML-015.jpg width=800 /></p>

<p align="center"><img src=./picture/AndrewNG-ML-016-1.jpg width=800 /></p>

<a name="classification"><h2>5. 分类问题 [<sup>目录</sup>](#content)</h2></a>

<a name="logistic-regression"><h3>5.1. logistic回归 [<sup>目录</sup>](#content)</h3></a>

<a name="logistic-regression-description"><h4>5.1.1. 问题描述 [<sup>目录</sup>](#content)</h4></a>

<a name="logistic-regression-decision-boundary"><h4>5.1.2. 理解决策边界 [<sup>目录</sup>](#content)</h4></a>

<a name="logistic-regression-fit"><h4>5.1.3. 拟合logistic回归 [<sup>目录</sup>](#content)</h4></a>

<a name="logistic-regression-cost-function"><h4>5.1.3.1. 确定损失函数 [<sup>目录</sup>](#content)</h4></a>

<a name="logistic-regression-solve-fit-using-gradient-descent"><h4>5.1.3.2. 梯度下降求解 [<sup>目录</sup>](#content)</h4></a>

<a name="logistic-regression-advanced-optimal-algorithmn"><h4>5.1.3.3. 高级优化算法简介与使用 [<sup>目录</sup>](#content)</h4></a>




