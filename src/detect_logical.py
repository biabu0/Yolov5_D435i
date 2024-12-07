# -*- coding: utf-8 -*-
# @Modified by: Ruihao
# @ProjectName:yolov5-pyqt5
import pyrealsense2 as rs
import sys
import cv2
import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *

from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box2

from ui.detect_ui import Ui_MainWindow # 导入detect_ui的界面



class UI_Logic_Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(UI_Logic_Window, self).__init__(parent)
        self.timer_video = QtCore.QTimer() # 创建定时器
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_slots()
        self.num_stop = 1 # 暂停与播放辅助信号，note：通过奇偶来控制暂停与播放


        # 全局变量，只检测一次
        self.border = False

        # 初始化RealSense管道
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,
                             15)  #可以设置10,15,30 初始化RealSenseSense摄像头，并配置为捕获640x480的深度和颜色图像，每秒15帧
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # 你可以根据需要调整分辨率和帧率

        self.info_show_int = 2

        self.mouse_positions = []
        self.frame_shape = None

        # 权重初始文件名
        self.openfile_name_model = None
        self.openfile_name_dataset = None

    # 控件绑定相关操作
    def init_slots(self):
        self.ui.pushButton_mask.clicked.connect(self.select_mask)
        self.ui.pushButton_camer.clicked.connect(self.button_camera_open)
        self.ui.pushButton_weights.clicked.connect(self.open_model)
        self.ui.pushButton_file.clicked.connect(self.open_file)
        self.ui.pushButton_stop.clicked.connect(self.button_video_stop)
        self.ui.pushButton_finish.clicked.connect(self.finish_detect)

        self.timer_video.timeout.connect(self.show_video_frame) # 定时器超时，将槽绑定至show_video_frame

    # 打开权重文件
    def open_model(self):
        self.openfile_name_model, _ = QFileDialog.getOpenFileName(self.ui.pushButton_weights, '选择weights文件',
                                                             'weights/')
        if not self.openfile_name_model:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开权重失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            print('加载weights文件地址为：' + str(self.openfile_name_model))

    # 加载相关参数，并初始化模型
    #def model_init(self):
        # 模型相关参数配置
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        self.opt = parser.parse_args() \

        self.opt.classes = [0]
        #print(self.opt)
        # 默认使用opt中的设置（权重等）来对模型进行初始化
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size

        # 若openfile_name_model不为空，则使用此权重进行初始化
        if self.openfile_name_model:
            weights = self.openfile_name_model
            print("Using button choose model")

        #cpu模式打包版本
        #self.device = torch.device('cpu')
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        cudnn.benchmark = True

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.half:
          self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        print("model initial done")
        # 设置提示框
        QtWidgets.QMessageBox.information(self, u"Notice", u"模型加载完成", buttons=QtWidgets.QMessageBox.Ok,
                                      defaultButton=QtWidgets.QMessageBox.Ok)

    def open_file(self):

        self.openfile_name_dataset = QFileDialog.getExistingDirectory(self, '选择数据集目录')
        if not self.openfile_name_dataset:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开文件地址失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.information(self, u"Notice", u"数据集路径为：" + str(self.openfile_name_dataset), buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
    def mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            # 将位置标准化（可选，根据需求决定是否需要）
            normalized_x = x / self.frame_shape[1]
            normalized_y = y / self.frame_shape[0]

            # 将位置添加到二维数组中
            self.mouse_positions.append([normalized_x, normalized_y])
        return ;
    def select_mask(self):
        self.mouse_positions = []

        self.pipeline.start(self.config)
        frames = self.pipeline.wait_for_frames()
        img_color = frames.get_color_frame()


        # 检查摄像头是否成功打开
        if img_color is None:
            print("Error: Could not open video device.")
            exit()

        img_color = np.asanyarray(img_color.get_data())
        self.frame_shape = img_color.shape[:2]
        # 创建一个窗口
        cv2.namedWindow('Camera Image')
        # 设置鼠标回调函数
        cv2.setMouseCallback('Camera Image', self.mouse_callback)
        while True:
            # 显示图像
            cv2.imshow('Camera Image', img_color)
            #等待按键，如果按下'q'键，退出循环
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        # 释放D435i对象
        self.pipeline.stop()  # 停止RealSense管道

        # 销毁创建的窗口
        print("mouse_positions", self.mouse_positions)
        QtWidgets.QMessageBox.information(self, u"Notice", u"遮掩区域选择成功", buttons=QtWidgets.QMessageBox.Ok,
                                      defaultButton=QtWidgets.QMessageBox.Ok)

    def save_dataset(self, frames):


        align_to = rs.stream.color
        align = rs.align(align_to)  # 对齐

        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_data = np.asanyarray(aligned_depth_frame.get_data(), dtype="uint16")
        color_image = np.asanyarray(color_frame.get_data())


        t1 = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        if not self.openfile_name_dataset:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"请先选择数据集地址", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            return

        save_path = os.path.join(self.openfile_name_dataset, "outfile", t1)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "color"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "depth"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "depthjpg"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "depth_mapped_image"), exist_ok=True)
        saved_count = int(time.time() * 1000) #毫秒级的时间戳


        depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 彩色图片保存为png格式
        cv2.imwrite(save_path + "/color/" + "{}".format(saved_count) + '.jpg', color_image)
        # -----------深度图保存信息----------------#
        # 深度信息由采集到的float16直接保存为npy格式
        np.save(os.path.join(save_path, "depth", "{}".format(saved_count)), depth_data)  #
        # 黑白图
        # 使用jpg格式保存的图片，图像采集错误还能肉眼发现
        cv2.imwrite(save_path + "/depthjpg/" + "{}.jpg".format(saved_count), depth_image)

        # 渲染的图片
        cv2.imwrite(save_path + "/depth_mapped_image/"+"{}.jpg".format(saved_count), depth_mapped_image)
        return True
    # 目标检测
    def detect(self, name_list, img):
        '''
        :param name_list: 文件名列表
        :param img: 待检测图片
        :return: info_show:检测输出的文字信息
        '''
        #(1, 3, 480, 640) [[[145 146 143], [148 149 146
        # ]]]
        showimg = img

        hl1 = self.mouse_positions[0][1]  # 监测区域高度距离图片顶部比例
        wl1 = self.mouse_positions[0][0]  # 监测区域高度距离图片左部比例
        hl2 = self.mouse_positions[1][1]  # 监测区域高度距离图片顶部比例
        wl2 = self.mouse_positions[1][0]  # 监测区域高度距离图片左部比例
        hl3 = self.mouse_positions[3][1]  # 监测区域高度距离图片顶部比例
        wl3 = self.mouse_positions[3][0]  # 监测区域高度距离图片左部比例
        hl4 = self.mouse_positions[2][1]  # 监测区域高度距离图片顶部比例
        wl4 = self.mouse_positions[2][0]  # 监测区域高度距离图片左部比例
        mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        pts = np.array([[int(img.shape[1] * wl1), int(img.shape[0] * hl1)],  # pts1
                            [int(img.shape[1] * wl2), int(img.shape[0] * hl2)],  # pts2
                            [int(img.shape[1] * wl3), int(img.shape[0] * hl3)],  # pts3
                            [int(img.shape[1] * wl4), int(img.shape[0] * hl4)]], np.int32)
        cv2.fillPoly(mask, [pts], (255, 255, 255))
        mask = 255 - mask

        # 应用mask：将mask为0的部分设置为黑色（0,0,0）
        img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
        # 2========================================================================================
        if not self.border:
        # 只显示一次
            # 定义框的颜色和线宽
            border_color = (255, 0, 0)  # 红色
            border_thickness = 2
            cv2.polylines(img, [pts], True, border_color, border_thickness)
            self.border = True
        # 显示结果
            cv2.imshow('Image with Mask and Border', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # 2========================================================================================


        with torch.no_grad():

            img = letterbox(img, new_shape=self.opt.img_size)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            # 1==============================================================================================================


                # 1========================================================================================
            pred = self.model(img, augment=self.opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            info_show = ""
            info_show_target = ""
            # Process detections
            self.info_show_int = 1
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # 3=====================================================================================
                    condition = (det[:, 5] == 0.0) & (det[:, 4] > 0.6)
                    if condition.any():
                        #print("有人员进入监测区域")
                        info_show_target = "有人员进入检测区域"
                        self.info_show_int = 0
                    else:

                        info_show_target = "无人员进入检测区域"
                        self.info_show_int = 1

                    # 3================================================================================================================================================================
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        name_list.append(self.names[int(cls)])
                        single_info = plot_one_box2(xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)
                        # print(single_info)
                        info_show = info_show + single_info + "\n"
        return info_show_target, self.info_show_int


    # 打开摄像头检测
    def button_camera_open(self):
        print("Open D435i camera to detect")

        # 打开RealSense摄像头，不需要额外判断是否打开，因为start会处理
        self.pipeline.start(self.config)
        print("D435i camera open success")

        self.timer_video.start(66)
        # 禁用其他按键
        self.ui.pushButton_weights.setDisabled(False)
        self.ui.pushButton_file.setDisabled(True)
        self.ui.pushButton_mask.setDisabled(True)
        self.ui.pushButton_camer.setDisabled(True)

    # 定义视频帧显示操作
    def show_video_frame(self):
        frames = self.pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        #在此处就获取帧，后面获取帧会导致获取color含有检测框
        # depth_frame = frames.get_depth_frame()
        if not color_frame:
            self.finish_detect()
            return
        color_image = np.asanyarray(color_frame.get_data())
        color_image_detect = color_image.copy()
        info_show, info_show_int = self.detect([], color_image_detect)  # 检测结果写入到原始img上

        #print(info_show)
        if info_show_int == 0:
            #print("---开始处理保存数据程序---")
            flag = self.save_dataset(frames)
            if flag:
                #print("数据保存成功")
                info_show += " 数据保存成功"
        elif info_show_int == 1:
            #print("---停止保存数据程序---")
            info_show += " 停止保存数据"

        # 显示检测信息和图像
        self.ui.textBrowser.setText(info_show)
        show = cv2.resize(color_image_detect, (640, 480))
        self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                 QtGui.QImage.Format_RGB888)
        self.ui.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.ui.label.setScaledContents(True)

    # 暂停与继续检测
    def button_video_stop(self):
        self.timer_video.blockSignals(False)
        # 暂停检测
        # 若QTimer已经触发，且激活
        if self.timer_video.isActive() == True and self.num_stop%2 == 1:
            self.ui.pushButton_stop.setText(u'暂停检测') # 当前状态为暂停状态
            self.num_stop = self.num_stop + 1 # 调整标记信号为偶数
            self.timer_video.blockSignals(True)
        # 继续检测
        else:
            self.num_stop = self.num_stop + 1
            self.ui.pushButton_stop.setText(u'继续检测')

    # 结束视频检测
    def finish_detect(self):
        self.pipeline.stop()  # 停止RealSense管道

        self.ui.label.clear()
        self.ui.pushButton_weights.setDisabled(False)
        self.ui.pushButton_file.setDisabled(False)
        self.ui.pushButton_mask.setDisabled(False)
        self.ui.pushButton_camer.setDisabled(False)

        # 重置暂停按钮
        if self.num_stop % 2 == 0:
            self.ui.pushButton_stop.setText(u'暂停/继续')
            self.num_stop += 1
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    current_ui = UI_Logic_Window()
    current_ui.show()
    sys.exit(app.exec_())

