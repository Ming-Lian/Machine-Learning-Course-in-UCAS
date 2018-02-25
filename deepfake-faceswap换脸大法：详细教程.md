<a name="content">目录</a>

[deepfake-faceswap换脸大法：详细教程](#title)
- [1. 内在算法思想](#infixed-algorithm)
- [2. 准备](#prerequisites)
	- [2.1. 硬件要求](#hardware)
	- [2.2. 运行环境搭建](#install-env)
	- [2.3. 获取faceswap代码](#get-code)
		- [2.3.1. 安装git](#install-git)
		- [2.3.2. git clone](#git-clone)
	- [2.4. 启动virtualenv](#setup-virtualenv)
	- [2.5. 配置你的project](#setup-project)
		- [2.5.1. 批量安装dependencies](#install-dependencies-one-step)
		- [2.5.2. 手动安装dependencies](#install-dependencies-step-by-step)
			- [2.5.2.1 解决dlib安装失败问题](#deal-with-dlib)
		
- [3. Workflow](#workflow)



<h1 name="title">deepfake-faceswap换脸大法：详细教程</h1>

<p align="center"><img src=/picture/Faceswap-exampleSwap.jpg width="800"></p>

<a name="infixed-algorithm"><h3>1. 内在算法思想 [<sup>目录</sup>](#content)</h3></a>

---

- 用 faceswap 底层依赖的工具包 face_recognition 以及训练好的 face_recognition_model 进行面部识别，其实这就是一个已经训练好的用于面部识别的 CNN（卷积神经网络）。然后在面部标注出关键的面容与表情特征点。对于原脸与目标脸（要替换成的脸），分别执行该步骤。


<img src=/picture/Faceswap-faceRecognition.png width="800">

<img src=/picture/Faceswap-faceRecognition2.png width="800">

- 利用脸部识别的结果，对原脸与目标脸进行匹配（该步骤依赖dlib工具包），即face align，主要利用在面部标注出关键的面容与表情特征点，将特征点分布比较一致的图组成一个个训练样本 (x<sup>(i)</sup>,y<sup>(i)</sup>)

- 用一个个训练样本来训练CNN，基于TensorFlow

<img src=/picture/Faceswap-CNN.jpg width="800">

想详细了解 **face_recognition** 请点 [这里](http://pythonhosted.org/face_recognition/index.html)

想详细了解 **dlib** 请点 [这里](https://pypi.python.org/pypi/dlib)

<a name="prerequisites"><h3>2. 准备 [<sup>目录</sup>](#content)</h3></a>

---

<h4 name="hardware">2.1. 硬件要求</h4>

你至少要满足一下条件之一：
- A powerful CPU
- A powerful GPU
> 目前仅支持 Nvidia GPUs ，无法支持农企（AMD），这是由该工具的底层依赖的TensorFlow决定的，开发人员表示也很无奈(｡í _ ì｡)
> 仅仅是 Nvidia GPUs 还不够，它还得至少能支持 CUDA Compute Capability 3.0 或者更高，言外之意：**这是有钱人的游戏，没钱就滚吧**

<h4 name="install-env">2.2. 运行环境搭建</h4>

**Python >= 3.2**

安装方法见：[安装Anaconda](https://github.com/Ming-Lian/Memo/blob/master/JupyterNotebook%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md#install-anaconda)

当你的系统中同时安装了Anacoda2和Anaconda3（对应着python2.7和Python3.6)，而你的日常工作中又同时要用到着两个版本的Python时，你可以按照以下方法自如地在两个版本之间转换：

```
# 将两个版本的Anaconda的路径写入~/.bashrc
$ vim ~/.bashrc

# 在~/.bashrc中添加下面两句：
# export anaconda2=$HOME/Anaconda2/bin
# export anaconda3=$HOME/Anaconda3/bin

# 用Python2.7时
$ $anaconda2/python
# 用Python3.6时
$ $anaconda3/python
```

**Virtualenv**

virtualenv 是一个创建隔绝的Python环境的工具。virtualenv创建一个包含所有必要的可执行文件的文件夹，用来使用Python工程所需的包。

```
# for Windows
pip install virtualenvwrapper-win

# for linux
$ $anaconda3/pip install virtualenv
```

<h4 name="get-code">2.3. 获取 faceswap 代码</h4>

推荐用git将代码仓库克隆到本地

<a name="install-git"><strong>2.3.1. 安装git</strong></a>

首先你得安装git到你的电脑上
> - **Windows**
> 下载地址：https://gitforwindows.org/ ，安装过程参考：
[git 2.14.1(windows)安装教程](http://blog.csdn.net/s740556472/article/details/77623453)
> - **Linux**
> 
> ```
> # 进入管理员身份
> $ su # 然后输入管理员密码
> # 安装git (for CentOS)
> # yum install git
> ```

<a name="git-clone"><strong>2.3.2. git clone</strong></a>

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

<h4 name="setup-virtualenv">2.4. 启动virtualenv</h4>

- **Windows**

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

- Linux

```
virtualenv faceswap_env/
```

> - 如果要退出virtualenv，输入`deactivate`
> - 如果要重新激活virtualenv，输入`source faceswap_env/bin/activate`


<h4 name="setup-project">2.5. 配置你的project</h4>

<h4 name="install-dependencies-one-step">2.5.1. 批量安装dependencies</h4>

当你已经激活virtualenv后，从requirement files安装依赖程序。requirement file位于faceswap repo中

![](/picture/Faceswap-requirementFiles.png)

由于GPU没有达到配置要求，我只能用CPU跑了，所以我们选择文件"**requirements-python36.txt**"

```
pip install -r requirements-python36.txt
```
安装dependencies需要花费一些时间，请耐心等待。。。

悲剧了，安装dependencies出了一些问题，由于使用了requirements-python36.txt进行批量化安装，我们不清楚到底哪一步出了问题，所以我们采取手动安装，按照requirements-python36.txt中的顺序，一个一个安装

<h4 name="install-dependencies">2.5.2. 手动安装dependencies</h4>


```
# requirements-python36.txt内容

pathlib==1.0.1
scandir==1.6
h5py==2.7.1
Keras==2.1.2
opencv-python==3.3.0.10
tensorflow==1.5.0
scikit-image
dlib
face_recognition
tqdm
```
<h4 name="deal-with-dlib">2.5.2.1. 解决dlib安装失败问题</h4>

按顺序执行`pip install *`进行安装，前几个都很顺利，直到遇到 **dlib** 时报错了：**Permission denied: 'cmake'** —— 没有cmake的执行权限，如果你是管理员那么你可以切换到管理员身份进行安装，如果不是的话，自己安装一个cmake。

```
# 用conda安装cmake
$ $anaconda3/conda install cmake
# 将cmake安装目录添加到环境变量
$ export PATH=$HOME/software/anaconda3/binL$PATH

# 测试cmake是否安装成功
$ cmake -h
```

安装好cmake后，继续尝试安装dlib

```
$ pip install dlib

# 或者可以用源码安装
$ python setup.py install 
```

可是还是报错，麻蛋，什么破玩意儿！

```
CMake Error at /share/disk5/lianm/basic_tool/dlib-19.9.0/dlib/external/pybind11/tools/pybind11Tools.cmake:32 (message):
  Unsupported compiler -- pybind11 requires C++11 support!
```

错误信息显示编译需要**C++11**的支持，也就是说我们还得再安装一个C++11，那就来吧

```
# 获取GCC 4.8.2包
$ wget -c -P basic_tool http://gcc.skazkaforyou.com/releases/gcc-4.8.2/gcc-4.8.2.tar.gz

# 解压
$ tar zxvf gcc-4.8.2.tar.gz

# 进入到目录gcc-4.8.2，运行：./contrib/download_prerequisites。这个神奇的脚本文件会帮我们下载、配置、安装依赖库，可以节约我们大量的时间和精力。
$ ./contrib/download_prerequisites

# 建立输出目录并到目录里
$ mkdir gcc-build-4.8.2；cd gcc-build-4.8.2

# 设置好configure(配置)
## --enable-languages 表示你要让你的gcc支持那些语言
## --disable-multilib 不生成编译为其他平台可执行代码的交叉编译器
## --disable-checking 生成的编译器在编译过程中不做额外检查
## --prefix 指定安装目录，如果不指定安装目录则或默认安装在/usr/local，不是管理员没有该目录的写权限会导致安装失败
$ ../configure --prefix=$HOME/software/gcc-4.8.2/bin --enable-checking=release --enable-languages=c,c++ --disable-multilib

# 开始编译，这一步比较耗时
$ make

# 安装
$ make install
```

检验是否安装成功

```
gcc -v
```

如果显示的gcc版本仍是以前的版本，就需要重启系统；或者可以查看gcc的安装位置：which gcc；




requirements安装好了以后，你可以尝试运行faceswap

```
python faceswap.py -h
```

<a name="workflow"><h3>3. Workflow [<sup>目录</sup>](#content)</h3></a>


参考资料：

(1) [deepfakes/faceswap: Prerequisites](https://github.com/deepfakes/faceswap/blob/master/INSTALL.md)

(2) [linux下安装或升级GCC4.8，以支持C++11标准](http://www.cnblogs.com/lizhenghn/p/3550996.html)
