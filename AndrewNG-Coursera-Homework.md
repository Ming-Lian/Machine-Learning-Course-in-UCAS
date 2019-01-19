<a name="content">目录</a>

[Andrew NG Coursera 课程编程作业](#title)
- [Ex1: Linear Regression](#ex1)
- [Ex2: Logistic回归](#ex2)





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
	
<a name="ex2"><h2>Ex2: Logistic回归 [<sup>目录</sup>](#content)</h2></a>

目标：

> 构建一个logistics回归模型，依据学生两次考试的成绩来预测一个学生能否被大学录取

通过的输入数据文件为`ex2data1.txt`：

```
34.62365962451697,78.0246928153624,0
30.28671076822607,43.89499752400101,0
35.84740876993872,72.90219802708364,0
60.18259938620976,86.30855209546826,1
79.0327360507101,75.3443764369103,1
45.08327747668339,56.3163717815305,0
61.10666453684766,96.51142588489624,1
75.02474556738889,46.55401354116538,1
76.09878670226257,87.42056971926803,1
...
```


1. 读入数据，并画出数据分布散点图

	```
	data = load('ex2data1.txt');
	x = data(:,:2);
	y = data(:,3);

	% 画图
	%% 区分出两类样本
	pos = find(y==1);
	neg = find(y==0);
	figure;
	%% 画出pos类样本
	plot(x(pos,1), x(pos,2),'k+','MarkerSize', 3);
	hold on;
	%% 画出neg类样本
	plot(x(neg,1), x(neg,2),'ko','MarkerSize', 3);
	xlab('Exam 1 score');
	ylab('Exam 2 score');
	hold off;
	```

	<p align="center"><img src=./picture/AndrewNG-homework-ex2-1.png width=800 /></p>

2. 定义sigmoid函数

	<p align="center"><img src=./picture/AndrewNG-homework-ex2-2.png height=50 /></p>
	
	<p align="center"><img src=./picture/AndrewNG-homework-ex2-3.png height=70 /></p>
	
	```
	function g = sigmoid(x)
		g = 1./(1+exp(-x));
	end
	```

3. 定义costFunction函数，返回cost和gradient

	损失函数为：

	<p align="center"><img src=./picture/AndrewNG-homework-ex2-4.png height=70 /></p>
	
	梯度的计算公式为：
	
	<p align="center"><img src=./picture/AndrewNG-homework-ex2-5.png height=70 /></p>
	
	```
	function [jVal, gradient] = costFunction(x,y,theta)
		predict = sigmoid(x*theta);
		leftCost = -y'*log(predict);
		rightCost = -(1-y)'*log(1-predict);
		jVal = (1/m)*(leftCost+rightCost);
		gradient = (1/m)*((predict-y)'*x);
	end
	```	
