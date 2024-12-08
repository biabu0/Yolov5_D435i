# Yolov5_D435i
## 项目介绍
通过YOLOV5与pyqt5实现一个使用D435i深度摄像头采集特定需求与场景下的深度数据的小程序<br>
1.指定需要屏蔽的检测区域，即使目标进入该区域也无法进行有效的检测，应用于特定场景的检测。<br>
2.只有目标在检测区域内，才进行数据的采集与保存，避免一直采集数据，目标离开检测区域则停止保存数据，避免在数据采集过程中存在大量的无效数据，节约数据清洗时间，节省磁盘容量。<br>
3.按照时间存储数据。<br>
4.使用pyqt5设计可视化界面，将UI界面与逻辑代码分离。<br>

## 视频演示

https://github.com/user-attachments/assets/584dcab8-f5c1-4d70-a6ea-a6eaf0038880

## 环境配置
        按照requements.txt文件配置yolov5环境，安装pyqt5和pyrealsense2。<br>
## 使用方法
        法1.直接运行detect_logical.py文件进入检测界面。<br>
        法 2.运行main_logic.py文件，进入登录界面，之后跳转到检测界面。<br>

## 核心代码功能解析
        **detect_logical.py**：负责加载模型，并初始化模型参数；选择遮蔽区域以及需要保存的数据文件地址；加载D435深度相机数据流，将数据送入检测，检测到特定目标返回数据保存的标志位进行数据存储。<br>
        **main_logic.py**:主界面，可以进行注册账号与登录账号。<br>
        **ui/ori_ui**:ui源文件，可以通过使用QTdesigner对UI界面进行修改，修改后使用**pyuic5 main.ui > ui_main.py**,（注意最好使用绝对路径，不然可能出现问题）转换成py文件。<br>
        **utlis/id_utlis.py与userInfo.csv**：用于写入账户信息。<br>        
## 博客地址
        https://blog.csdn.net/2201_75766594/article/details/144316937?spm=1001.2014.3001.5501

  
        

​

​
