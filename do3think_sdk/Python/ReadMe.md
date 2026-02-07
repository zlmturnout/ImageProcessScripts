### 介绍

Python版本：**必须**`3.6`（后续支持更多版本）

支持平台：Linux、Windows

**建议：**

1、在Conda环境下配制一个python3.6版本的环境来测试运行，有关Conda的安装与配置可以百度一下。

2、下载依赖库可能很慢，建议更换为**国内源**，或者使用科学上网工具。

3、确保电脑上已安装度申相机的驱动，有关驱动可以前往[度申开发者下载中心](http://developer.do3think.com/download/)下载。

 		目录：度申相机开源资料>度申相机驱动

有关依赖：

```python
numpy==1.16.2
opencv-python==4.0.0.21
```

安装依赖命令：

```
pip install -r requirements.txt
```

### 关于Python-API文档
```
执行 help(Camera) 可以查看相应的帮助信息
更多的帮助信息请参考DVPCamera.chm，并结合BasedCam的“开发者模式”
其中的dvpSet...和dvpGet...等函数在python都以属性赋值的形式出现
比如dvpGetGamma和dvpSetGamma，对应于variable = camera.Gamma和camera.Gamma = 100
```




### 运行

Linux有关dvp库在目录`lib\linux\python3.5m\`下：

```
 aarch64/   
 armhf/
 x32/
 x64/
```

Windows有关dvp库在目录`lib\windows\python3.6`下：

```
x32
x64
```

在运行OpenCV_Demo.py脚本之前需要把**对应版本的库**复制**到脚本所在目录**

例如：

![](./Image/tips.png)

运行命令：`python OpenCV_Demo.py`

之后会在控制台中会出现选择相机的提示，根据提示操作即可显示图片。

![](./Image/runtime.png)

