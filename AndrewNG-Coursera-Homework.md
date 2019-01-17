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
3. 计算损失函数

	线性回归的损失函数为：
	
	<p align="center"><img src=./picture/AndrewNG-homework-ex1-2.png height=70 /></p>

	```
	% 定义一个函数computeCost来计算损失函数
	function J = computeCost(X, y, theta)
		m = length(y);
		predictions = X*theta; % 计算预测值hθ
		sqerrors = (predictions - y).^2; % 计算平方误，矩阵的点乘运算得到的是一个向量
		J = 1/(2*m)* sum(sqerrors);
	end
	```

4. 执行梯度下降

	<p align="center"><img src=./picture/AndrewNG-homework-ex1-3.png height=70 /></p>
	
	```
	% 定义一个函数gradientDescent来执行梯度下降，为了后面观察梯度下降过程中损失函数的变化，记录每一步迭代后的损失函数值
	function [theta, J_history] = gradientDescent(X, y, theta, alpha, iterations)
		m = length(y);
		J_history = zeros(num_iters, 1);
		% 以迭代次数为唯一迭代终止条件
		for iter = 1:num_iters
			% 计算梯度
			predictions = X*theta;
			updates = X'*(predictions - y);
			theta = theta - alpha*(1/m)*updates;
			J_history(iter) = computeCost(X, y, theta);
		end
	end		
	```

5. 绘制拟合直线

	<p align="center"><img src=./picture/AndrewNG-homework-ex1-4.png width=800 /></p>

	```
	hold on; % 保留之前的绘图窗口，在这个绘图窗口继续画出拟合直线
	plot(X(:,2), X*theta, '-');
	legend('Training data', 'Linear regression');
	```
	
	
