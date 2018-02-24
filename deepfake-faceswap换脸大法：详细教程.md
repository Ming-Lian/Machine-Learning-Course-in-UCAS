<a name="content">目录</a>

[deepfake-faceswap换脸大法：详细教程](#title)
- [准备](#prerequisites)
	- [硬件要求](#hardware)
	- [运行环境搭建](#install-env)
	- [获取faceswap代码](#get-code)
	- [启动virtualenv](#setup-virtualenv)
	- [配置你的project](#setup-project)



<h1 name="title">deepfake-faceswap换脸大法：详细教程</h1>

<p align="right">—— 基于 Windows CMD 命令行 & CPU</p>

<a name="prerequisites"><h3>准备 [<sup>目录</sup>](#content)</h3></a>

<h4 name="hardware">硬件要求</h4>

你至少要满足一下条件之一：
- A powerful CPU
- A powerful GPU
> 目前仅支持 Nvidia GPUs ，无法支持农企（AMD），这是由该工具的底层依赖的TensorFlow决定的，开发人员表示也很无奈(｡í _ ì｡)
> 仅仅是 Nvidia GPUs 还不够，它还得至少能支持 CUDA Compute Capability 3.0 或者更高，言外之意：**这是有钱人的游戏，没钱就滚吧**

<h4 name="install-env">运行环境搭建</h4>

**Python >= 3.2**

安装方法见：[安装Anaconda](https://github.com/Ming-Lian/Memo/blob/master/JupyterNotebook%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md#install-anaconda)

**Virtualenv**

virtualenv 是一个创建隔绝的Python环境的工具。virtualenv创建一个包含所有必要的可执行文件的文件夹，用来使用Python工程所需的包。

```
pip install virtualenvwrapper-win
```

<h4 name="get-code">获取 faceswap 代码</h4>

推荐用git将代码仓库克隆到本地

首先你得安装git到你的电脑上，下载地址：https://gitforwindows.org/ ，安装过程参考：
[git 2.14.1(windows)安装教程](http://blog.csdn.net/s740556472/article/details/77623453)

打开git Bash（或git CMD)

![](/picture/Faceswap-gitBash.png)

git bash命令行的输入语法类似于Linux而不同于Windows下的CMD。下面克隆faceswap repo（代码仓库）

```
# 进入指定工作目录
$ cd e:
$ cd DeepLearing_practice/

# git克隆
$ git clone https://github.com/deepfakes/faceswap.git
```

<h4 name="setup-virtualenv">启动virtualenv</h4>

初始化virtualenv，在CMD中执行：

```
# 进入指定工作目录
e:
cd DeepLearing_practice/

# initialize our virtualenv
mkvirtualenv faceswap
setprojectdir .
```

![](/picture/Faceswap-inital-virtualenv.png)

> - 如果要退出virtualenv，输入`deactivate`
> - 如果要重新激活virtualenv，输入`workon faceswap`

<h4 name="setup-project">配置你的project</h4>

当你已经激活virtualenv后，从requirement files安装依赖程序。requirement file位于faceswap repo中

![](/picture/Faceswap-requirementFiles.png)

由于GPU没有达到配置要求，我只能用CPU跑了，所以我们选择文件"**requirements-python36.txt**"

```
pip install -r requirements-python36.txt
```




参考资料：

(1) [deepfakes/faceswap: Prerequisites](https://github.com/deepfakes/faceswap/blob/master/INSTALL.md)
