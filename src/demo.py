from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import cv2
import numpy as np
import time
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QTimer

from lib.opts import opts
from lib.Detector import Detector
from lib.dataset.Pascal import PascalVOC
from classify.similarity import Frame_Queue
from classify.config import MAX_SIZE, WINDOWS
from classify.train import merge
from classify.data_augment import res2vec
from classify.rnn_classify import rnn_demo

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

# sys.path.append("C:/User/13778/workshop/gitrepos/Air_Apron/src/")
# sys.path.append("/gs/home/tongchao/zc/Air_Apron/src/")


def demo(opt):
    cls_map_id = ['__background__', 'plane', 'head', 'wheel', 'wings', 'stair',
                           'oil_car', 'person', 'cone', 'engine', 'traction', 'bus', 'queue', 'cargo']
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = PascalVOC
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    detector = Detector(opt)
    rets = []
    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            cv2.imshow('input', img)
            ret = detector.run(img)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            rets.append(ret['results'])
            if cv2.waitKey(1) == 27:
                return  # esc to quit
    else:
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls, key=lambda x: int(x[:-4])):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]
        for (image_name) in image_names:
            ret = detector.run(image_name)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            rets.append(ret['results'])
    dets_total = []
    for i in range(len(rets)):
        all_bboxes = rets[i]
        detections = []
        for cls_ind in all_bboxes:
            category_id = cls_map_id[cls_ind]
            for bbox in all_bboxes[cls_ind]:
                score = bbox[4]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                bbox_out = [[x1, y1], [x1, y2], [x2, y1], [x2, y2]]

                detection = {
                    "class": category_id,
                    "bbox": bbox_out,
                    "center": [(x2+x1)/2, (y1+y2)/2],
                    "size": [x2-x1, y2-y1],
                    "score": float("{:.2f}".format(score))
                }
                if score >= 0.02 and x1 > 0 and y2 < 576:
                    detections.append(detection)
        dets_total.append(detections)
    return dets_total


def detection_demo(detector, img):
    ret = detector.run(img)
    time_str = ''
    for stat in time_stats:
        time_str += '{} {:.3f}s |'.format(stat, ret[stat])
    return ret, time_str


class Video(QWidget):

    def __init__(self, opt):
        super(Video, self).__init__()
        self.frame = []  # 存图片
        self.detectFlag = False  # 检测flag
        self.cap = []
        self.timer_camera = QTimer()  # 定义定时器

        # 外框
        self.resize(1200, 900)
        self.setWindowTitle("Event Detection Demo")
        # 图片label
        self.label = QLabel(self)
        self.label.setText("Waiting for video...")
        self.label.setFixedSize(1024, 576)  # width height
        self.label.move(88, 220)
        self.label.setStyleSheet("QLabel{background:rgb(255,255,240);}"
                                 "QLabel{color:rgb(100,100,100);font-size:15px;font-weight:bold;font-family:宋体;}"
                                 )

        # 时间显示
        self.time_label = QLabel(self)
        self.time_label.setText("Time Display")
        self.time_label.setFixedSize(250, 40)
        self.time_label.move(100, 20)
        self.time_label.setStyleSheet("QLabel{color:rgb(100,100,100);font-size:18px;font-weight:bold;font-family:宋体;}")

        # 事件显示
        self.oil_label = QLabel(self)
        self.oil_label.setText("Waiting for detection...")
        self.oil_label.setFixedSize(250, 40)
        self.oil_label.move(400, 20)
        self.oil_label.setStyleSheet("QLabel{color:rgb(100,100,100);font-size:18px;font-weight:bold;font-family:宋体;}")

        # 显示label
        self.stair_label = QLabel(self)
        self.stair_label.setText("Waiting for detection...")
        self.stair_label.setFixedSize(250, 40)  # width height
        self.stair_label.move(400, 70)
        self.stair_label.setStyleSheet("QLabel{color:rgb(100,100,100);font-size:18px;font-weight:bold;font-family:宋体;}")

        self.cargo_label = QLabel(self)
        self.cargo_label.setText("Waiting for detection...")
        self.cargo_label.setFixedSize(250, 40)
        self.cargo_label.move(400, 120)
        self.cargo_label.setStyleSheet("QLabel{color:rgb(100,100,100);font-size:18px;font-weight:bold;font-family:宋体;}")

        self.traction_label = QLabel(self)
        self.traction_label.setText("Waiting for detection...")
        self.traction_label.setFixedSize(250, 40)
        self.traction_label.move(400, 170)
        self.traction_label.setStyleSheet("QLabel{color:rgb(100,100,100);font-size:18px;font-weight:bold;font-family:宋体;}")

        self.board_label = QLabel(self)
        self.board_label.setText("Waiting for detection...")
        self.board_label.setFixedSize(250, 40)
        self.board_label.move(800, 20)
        self.board_label.setStyleSheet("QLabel{color:rgb(100,100,100);font-size:18px;font-weight:bold;font-family:宋体;}")

        self.prepare_label = QLabel(self)
        self.prepare_label.setText("Waiting for detection...")
        self.prepare_label.setFixedSize(250, 40)
        self.prepare_label.move(800, 70)
        self.prepare_label.setStyleSheet("QLabel{color:rgb(100,100,100);font-size:18px;font-weight:bold;font-family:宋体;}")

        self.t_label = QLabel(self)
        self.t_label.setText("Waiting for detection...")
        self.t_label.setFixedSize(250, 40)
        self.t_label.move(800, 120)
        self.t_label.setStyleSheet("QLabel{color:rgb(100,100,100);font-size:18px;font-weight:bold;font-family:宋体;}")

        self.alert_label = QLabel(self)
        self.alert_label.setText("Waiting for detection...")
        self.alert_label.setFixedSize(250, 40)
        self.alert_label.move(800, 170)
        self.alert_label.setStyleSheet("QLabel{color:rgb(100,100,100);font-size:18px;font-weight:bold;font-family:宋体;}")

        # 开启视频按键
        self.btn = QPushButton(self)
        self.btn.setText("Open")
        self.btn.move(250, 840)
        self.btn.clicked.connect(self.slotStart)
        # 检测按键
        # self.btn_detect = QPushButton(self)
        # self.btn_detect.setText("Detect")
        # self.btn_detect.move(400, 870)
        # self.btn_detect.setStyleSheet("QPushButton{background:red;}")  # 没检测红色，检测绿色
        # self.btn_detect.clicked.connect(self.detection)
        # 关闭视频按钮
        self.btn_stop = QPushButton(self)
        self.btn_stop.setText("Stop")
        self.btn_stop.move(950, 840)
        self.btn_stop.clicked.connect(self.slotStop)
        # 检测用
        self.cls_map_id = ['__background__', 'plane', 'head', 'wheel', 'wings', 'stair',
                           'oil_car', 'person', 'cone', 'engine', 'traction', 'bus', 'queue', 'cargo']
        self.stable = ['oil_car', 'stair', 'cargo', 'traction', 'plane']
        self.f = 0
        self.Dataset = PascalVOC
        self.opt = opts().update_dataset_info_and_set_heads(opt, self.Dataset)
        self.detector = Detector(self.opt)
        self.fq = Frame_Queue(max_size=MAX_SIZE, wind=WINDOWS)
        self.count = 0
        self.sequence = []
        self.rets = None
        self.result = None
        self.status = ['开始', '结束', '进行中', '未发生']
        self.alert_status= ['安全', '存在风险！', '危险！！！']


    def slotStart(self):
        """ Slot function to start the progamme
            """
        videoName, _ = QFileDialog.getOpenFileName(self, "Open", "", "*.mp4;;*.avi;;*.asf;;All Files(*)")
        if videoName != "":  # “”为用户取消
            self.cap = cv2.VideoCapture(videoName)
            self.timer_camera.start(40)
            self.timer_camera.timeout.connect(self.openFrame)

    def slotStop(self):
        """ Slot function to stop the programme
            """
        if self.cap != []:
            self.cap.release()
            self.timer_camera.stop()  # 停止计时器
            self.time_label.setText("Time Display")
            self.label.setText("This video has been stopped.")
            self.oil_label.setText("Waiting for detection...")
            self.stair_label.setText("Waiting for detection...")
            self.cargo_label.setText("Waiting for detection...")
            self.traction_label.setText("Waiting for detection...")
            self.board_label.setText("Waiting for detection...")
            self.prepare_label.setText("Waiting for detection...")
            self.t_label.setText("Waiting for detection...")
            self.alert_label.setText("Waiting for detection...")
            self.label.setStyleSheet("QLabel{background:rgb(255,255,240);}"
                                     "QLabel{color:rgb(100,100,100);font-size:15px;font-weight:bold;font-family:宋体;}"
                                     )
        else:
            # self.label_num.setText("Push the left upper corner button to Quit.")
            Warning = QMessageBox.warning(self, "Warning", "Push the left upper corner button to Quit.",
                                          QMessageBox.Yes)

    def openFrame(self):
        """ Slot function to capture frame and process it
            """
        if (self.cap.isOpened()):
            ret, self.frame = self.cap.read()
            if ret:
                # frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(self.frame, (1024, 576))
                if self.f % 25 == 0:
                    sec = self.f // 25
                    m, s = divmod(sec, 60)
                    h, m = divmod(m, 60)
                    self.time_label.setText("Now: {}h-{}m-{}s".format(h, m, s))
                if self.f % 25 == 0:
                # # frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                #     frame = self.frame
                    begin_time = time.time()
                    rets = self.detector.run(frame)
                    rets = self.format(rets['results'])
                    self.rets = rets
                    self.fq.ins(rets)
                    self.result = self.fq.get_result()
                    # print(result)
                    res, pos_res, size_res = merge(rets, self.result)
                    sample = res2vec(res, pos_res, size_res)
                    if self.count == 0:
                        for i in range(MAX_SIZE):
                            self.sequence.append(sample)
                    else:
                        self.sequence.pop()
                        self.sequence.append(sample)
                    # 加入到分类网络中

                    classification, prediction = rnn_demo(sample=[self.sequence],
                                                          save_dir='classify/models/rnn/epoch_1000_modify_0',
                                                          latest_iter=2000)
                    print(time.time()-begin_time)
                    self.count += 1
                    self.oil_label.setText("加油车连接"+self.status[classification[0]])
                    self.stair_label.setText("客舱车连接"+self.status[classification[1]])
                    self.cargo_label.setText("货舱车连接"+self.status[classification[2]])
                    self.traction_label.setText("牵引车连接"+self.status[classification[3]])
                    self.board_label.setText("乘客登机"+self.status[classification[4]])
                    self.prepare_label.setText("起飞准备"+self.status[classification[5]])
                    self.t_label.setText("牵引过程"+self.status[classification[6]])
                    self.alert_label.setText("当前人员位置"+self.alert_status[classification[7]])
                # print(self.rets)
                for split in self.stable:
                    if self.result['is_' + split]:
                        obj = self.result[split]
                        # print(obj)
                        color = (250, 240, 230)
                        cls = obj['class']
                        bbox = obj['bbox']
                        x, y, w, h = int(bbox[0][0]), int(bbox[0][1]), int(obj['size'][0]), int(obj['size'][1])
                        cv2.rectangle(frame, (x, y), (w + x, h + y), (0, 255, 0), 2)

                        cv2.rectangle(frame,
                                      pt1=(x, y),
                                      pt2=(x + 50, y + 15),
                                      color=color,
                                      thickness=-1)
                        cv2.putText(frame,
                                    text=str(cls),
                                    org=(x, y + 10),
                                    fontFace=1,
                                    fontScale=1,
                                    thickness=1,
                                    color=(0, 0, 0))

                for obj in self.rets:
                    color = (250, 240, 230)
                    cls = obj['class']
                    if cls not in self.stable:
                        bbox = obj['bbox']
                        x, y, w, h = int(bbox[0][0]), int(bbox[0][1]), int(obj['size'][0]), int(obj['size'][1])
                        cv2.rectangle(frame, (x, y), (w + x, h + y), (0, 255, 0), 1)

                        cv2.rectangle(frame,
                                      pt1=(x, y),
                                      pt2=(x + 50, y + 15),
                                      color=color,
                                      thickness=-1)
                        cv2.putText(frame,
                                    text=str(cls),
                                    org=(x, y + 10),
                                    fontFace=1,
                                    fontScale=1,
                                    thickness=1,
                                    color=(0, 0, 0))

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = frame.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QImage(frame.data, width, height, bytesPerLine,
                                 QImage.Format_RGB888).scaled(self.label.width(), self.label.height())
                self.label.setPixmap(QPixmap.fromImage(q_image))
                self.f += 1
            else:
                self.cap.release()
                self.timer_camera.stop()  # 停止计时器


    # def detection(self):
    #     self.detectFlag = not self.detectFlag  # 取反
    #     if self.detectFlag == True:
    #         self.btn_detect.setStyleSheet("QPushButton{background:green;}")
    #     else:
    #         self.btn_detect.setStyleSheet("QPushButton{background:red;}")
    # #        self.label_num.setText("There are 5 people.")

    def format(self, all_bboxes):
        detections = []
        for cls_ind in all_bboxes:
            category_id = self.cls_map_id[cls_ind]
            for bbox in all_bboxes[cls_ind]:
                score = bbox[4]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                bbox_out = [[x1, y1], [x1, y2], [x2, y1], [x2, y2]]

                detection = {
                    "class": category_id,
                    "bbox": bbox_out,
                    "center": [(x2 + x1) / 2, (y1 + y2) / 2],
                    "size": [x2 - x1, y2 - y1],
                    "score": float("{:.2f}".format(score))
                }
                if score >= 0.02 and x1 > 0 and y2 < 576:
                    detections.append(detection)
        return detections

if __name__ == "__main__":
    opt = opts().parse()
    # rets = demo(opt)
    app = QtWidgets.QApplication(sys.argv)
    my = Video(opt)
    my.show()
    sys.exit(app.exec_())