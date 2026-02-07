# -*- coding: utf-8 -*-

#执行 help(Camera) 可以查看相应的帮助信息
#更多的帮助信息请参考DVPCamera.chm，并结合BasedCam的“开发者模式”
#其中的dvpSet...和dvpGet...等函数在python都以属性赋值的形式出现
#比如dvpGetGamma和dvpSetGamma，对应于variable = camera.Gamma和camera.Gamma = 100

import sys,os,time
import win32api
dll_path=r'I:\Coding\GitReposity\ImageProcessScripts\do3think_sdk\Python\lib\windows\python3.6\x64'
#os.add_dll_directory(dll_path)
sys.path.append(".")
from ctypes import *
dvp_dll = cdll.LoadLibrary(os.path.join(dll_path, 'DVPCamera64.dll'))
dvp_pyd = os.path.join(dll_path, 'dvp.pyd')
info=win32api.GetFileVersionInfo(dvp_pyd, '\\')
ms=info['FileVersionMS']
ls=info['FileVersionLS']
print("dvp.pyd版本号:",ms,ls)
#from do3think_sdk.Python.demo.dvp import *                                           #将对应操作系统的dvp.pyd或dvp.so放入python安装目录下的Lib目录或者工程目录
from dvp import *    
import numpy as np                                          #用pip命令安装numpy库
import cv2                                                  #用pip命令安装opencv-python库

#将帧信息转换为numpy的矩阵对象，后续可以通过opencv的cvtColor转换为特定的图像格式
def frame2mat(frameBuffer):
    frame, buffer = frameBuffer
    bits = np.uint8 if(frame.bits == Bits.BITS_8) else np.uint16
    shape = None
    convertType = None
    if(frame.format >= ImageFormat.FORMAT_MONO and frame.format <= ImageFormat.FORMAT_BAYER_RG):
        shape = 1
    elif(frame.format == ImageFormat.FORMAT_BGR24 or frame.format == ImageFormat.FORMAT_RGB24):
        shape = 3
    elif(frame.format == ImageFormat.FORMAT_BGR32 or frame.format == ImageFormat.FORMAT_RGB32):
        shape = 4
    else:
        return None

    mat = np.frombuffer(buffer, bits)
    mat = mat.reshape(frame.iHeight, frame.iWidth, shape)   #转换维度
    return mat

#这个函数演示相机功能的基本调用方法，仅供参考
def setCameraParams(camera):
    camera.Dialog();                                        #显示参数调节对话框。在界面里面设置好参数以后可以存档，以避免重复设置

    #以下设置ROI
    roiDescr = camera.RoiDescr                              #ROI的描述
    roi = camera.Roi                                        #获取ROI
    roi.X = 0                                               #横纵坐标
    roi.Y = 0
    roi.W = 400                                             #宽度，高度
    roi.H = 400
    camera.Roi = roi                                        #设置ROI

    #camera.ResolutionModeSel = 0
    #print("分辨率模式0")

    #以下设置固定的曝光时间
    camera.AeOperation = AeOperation.AE_OP_OFF              #关闭自动曝光
    camera.AntiFlick = AntiFlick.ANTIFLICK_DISABLE          #如果在直流光源下则不用消除频闪
    expDescr = camera.ExposureDescr                         #曝光时间的描述
    time = camera.Exposure                                  #获取曝光时间
    camera.Exposure = time                                  #设置曝光时，可以根据expDescr来设置其他曝光时间
    #如果是自动曝光模式
    #camera.AntiFlick = AntiFlick.ANTIFLICK_50HZ             #启用消除50HZ的频闪
    camera.AeTarget = camera.AeTarget + 1                   #设置需要调节到的目标亮度
    camera.AeMode = AeMode.AE_MODE_AE_ONLY                  #仅仅自动调节曝光时间
    camera.AeOperation = AeOperation.AE_OP_CONTINUOUS       #启动自动曝光

    #设置完成后可以保存参数

    #设置RGB增益（彩色相机可设置）
    #camera.GGain = 1
    #camera.RGain = 1
    #camera.BGain = 1

    camera.SaveConfig("example.ini")

#定义主函数
def main():
    cameraInfo = Refresh();                                 #刷新并获取相机列表
    if(len(cameraInfo) == 0):                               #没有任何设备则退出
        print(u"没有找到设备")
        return

    for k, v in enumerate(cameraInfo):                      #打印相机索引和名称
        print(k, "->", v.FriendlyName)

    while(True):                                            #循环直到打开一台相机
        try:
            str = input("请选定将要打开的相机索引号(0,1,2...):")
            index = (int)(str)                              #输入的索引号字符串转换为整数
            camera = Camera(index)                          #以索引号的方式打开相机
            #camera = Camera(cameraInfo[index].FriendlyName)#或以名称的方式打开相机
            print(camera);                                  #打印相机信息
            break
        except dvpException as e:
            print(u"打开相机失败:", e.Status)                #如果是DVP的标准异常
        except BaseException as e:
            print(u"非法的索引号:", str)                     #其他异常
    
    try:
        camera.TriggerState = False                         #从触发模式切换到连续出图模式
        #setCameraParams(camera)                             #设置其他的相机的参数
        camera.Start()                                      #启动视频流
    except dvpException as e:
        print(u"操作相机出错:", e.Status)

    while (cv2.waitKey(1) != 27):                           #按ESC键则退出循环
        try:
            print(camera.FrameCount)                        #打印帧统计信息
            frame = camera.GetFrame(4000)                   #从相机采集图像数据，超时时间为4000毫秒
        except dvpException as e:
            print(u"采集图像数据失败:", e.Status)
            if(e.Status == Status.DVP_STATUS_TIME_OUT):     
                continue                                    #如果只是超时错误，则继续采集
            break                                           #其他错误则中止采集                                                                 

        mat = frame2mat(frame)                              #转换为标准数据格式
        cv2.imshow(u"Preview (Press ESC exit)", mat)        #显示图像数据

    cv2.destroyAllWindows()                                 #销毁窗口
    camera.Stop()                                           #停止视频流
    camera.Close()                                          #关闭相机

#执行主函数
main()


