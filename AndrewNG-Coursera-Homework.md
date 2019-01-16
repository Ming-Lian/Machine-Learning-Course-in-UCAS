<a name="content">目录</a>

[Andrew NG Coursera 课程编程作业](#title)
- [Ex1: Linear Regression](#ex1)





<h1 name="title">Andrew NG Coursera 课程编程作业</h1>

使用课程推荐的`Octave`进行编程实现，可以将`Octave`理解为开源版本的`MATLAB`

<a name="ex1"><h2>Ex1: Linear Regression [<sup>目录</sup>](#content)</h2></a>

1. 读入数据

	```
	data = load('ex1data1.txt'); % 导入的数据文件为用逗号隔开的两列，第一列为x，第二列为y
	X = data(:, 1);
	y = data(:, 2);
	% 可以尝试绘图
	% figure;plot(x,y);
	m = length(y);
	```

	数据分布图如下：

	<p align="center"><img src=./picture/AndrewNG-homework-ex1-1.png width=800 /></p>

2. 梯度下降前的数据预处理与设置

	```
	X = [ones(m,1),data(:,1)]; % 添加x0列，都设为1
	theta = zeros(2,1); % 初始化θ值
	
	% 梯度下降的一些设置信息
	iterations = 1500; % 迭代次数
	alpha = 0.01; % 学习率α
	```
