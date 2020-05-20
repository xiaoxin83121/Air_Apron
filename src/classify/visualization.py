from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
import json
from classify.data_augment import generate_test


def object_loss(file_names):
    # 目标检测损失函数可视化
    with open(file_names) as log_f:
        lines = log_f.readlines()
        losses = {'loss': [], 'hm_loss': [], 'wh_loss': [], 'off_loss': []}
        for line in lines:
            split = line.split('|')
            loss = float(split[1][0:-1].split(' ')[-1])
            hm_loss = float(split[2][0:-1].split(' ')[-1])
            wh_loss = float(split[3][0:-1].split(' ')[-1])
            off_loss = float(split[4][0:-1].split(' ')[-1])
            losses['loss'].append(loss)
            losses['hm_loss'].append(hm_loss)
            losses['wh_loss'].append(wh_loss)
            losses['off_loss'].append(off_loss)
        # plt.plot(losses['loss'])
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.title('resnet-18-loss')
        plt.plot(losses['loss'][1:-1])
        plt.show()
        # print('loss={} | hm_loss={} | wh_loss={} | off_loss={}'.format(loss, hm_loss, wh_loss, off_loss))


def event_loss(exp_id):
    # 事件检测损失函数可视化
    base_path = './models/rnn/'
    log_path = os.path.join(base_path, exp_id, 'log.txt')
    with open(log_path, 'r') as log_f:
        lines = log_f.readlines()
        lines = lines[1: -2]
        epoch_loss = []
        epoch_content = []
        for line in lines:
            split = line.split(':')
            epoch_list = split[1].split('/')
            epoch = int(epoch_list[0].split(' ')[-1])
            iter = int(epoch_list[1].split(' ')[-1])
            loss = float(split[2].split('=')[1])
            if iter == 0 and epoch != 0:
                avg_loss = np.mean(epoch_content)
                epoch_loss.append(avg_loss)
                epoch_content = []
            epoch_content.append(loss)
        # print(epoch_loss)
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.plot(epoch_loss)
        plt.show()


def plane_analyse():
    """
    分析飞机的相关部件之间的位置
    存储head，engine_left, engine_right位置坐标和尺寸在整体大框架内的比例
    """
    file_path = "../../data/VOC2007_new/Annotations"
    file_list = os.listdir(file_path)
    head_ws = []
    head_hs = []
    head_size_ws = []
    head_size_hs = []
    engine_l_ws = []
    engine_l_hs = []
    engine_r_ws = []
    engine_r_hs = []
    engine_size_ws = []
    engine_size_hs = []
    for filename in file_list:
        index = int(filename.split('.')[0])
        sample = generate_test(file_path, index)
        plane_dict = {'plane': [], 'engine': [], 'head': []}
        for inp in sample:
            cls = inp['class']
            if cls in plane_dict.keys():
                plane_dict[cls].append(inp)
        if len(plane_dict['plane']) < 1:
            pass
        else:
            plane = plane_dict['plane'][0]
            left_top = plane['bbox'][0]
            size = plane['size']
            # head
            if len(plane_dict['head']) < 1:
                pass
            else:
                head = plane_dict['head'][0]
                head_w = (head['center'][0]-left_top[0]) / size[0]
                head_h = (head['center'][1]-left_top[1]) / size[1]
                if head_w < 0.55 and head_w > 0.35:
                    head_ws.append(head_w)
                if head_w < 1 and head_w > 0:
                    head_hs.append(head_h)

                size_w = (head['size'][0]) / size[0]
                size_h = (head['size'][1]) / size[1]
                if size_w > 0 and size_w < 1:
                    head_size_ws.append(size_w)
                if size_h > 0 and size_h < 1:
                    head_size_hs.append(size_h)

            # engine
            if len(plane_dict['engine']) < 1:
                pass
            elif len(plane_dict['engine']) == 1:  # 单引擎
                engine = plane_dict['engine'][0]
                engine_w = (engine['center'][0]-left_top[0]) / size[0]
                engine_h = (engine['center'][1]-left_top[1]) / size[1]
                if engine['center'][0] < plane['center'][0]:
                    if engine_w > 0  and engine_w < 1:
                        engine_l_ws.append(engine_w)
                    if engine_h > 0 and engine_h < 1:
                        engine_l_hs.append(engine_h)
                else:
                    if engine_w > 0 and engine_w < 1:
                        engine_r_ws.append(engine_w)
                    if engine_h > 0 and engine_h < 1:
                        engine_r_hs.append(engine_h)

                size_w = (engine['size'][0]) / size[0]
                size_h = (engine['size'][1]) / size[1]
                if size_w > 0 and size_w < 1:
                    engine_size_ws.append(size_w)
                if size_h > 0 and size_h < 1:
                    engine_size_hs.append(size_h)
            else:  # 双引擎
                engine_left = plane_dict['engine'][0] if plane_dict['engine'][0]['center'][0] < \
                                                         plane_dict['engine'][1]['center'][0] else \
                                                         plane_dict['engine'][1]
                engine_right = plane_dict['engine'][0] if plane_dict['engine'][0]['center'][0] >= \
                                                         plane_dict['engine'][1]['center'][0] else \
                                                         plane_dict['engine'][1]
                engine_l_w = (engine_left['center'][0]-left_top[0]) / size[0]
                engine_l_h = (engine_left['center'][1]-left_top[1]) / size[1]
                engine_r_w = (engine_right['center'][0] - left_top[0]) / size[0]
                engine_r_h = (engine_right['center'][1] - left_top[1]) / size[1]
                if engine_l_w > 0 and engine_l_w < 1:
                    engine_l_ws.append(engine_l_w)
                if engine_l_h > 0 and engine_l_h < 1:
                    engine_l_hs.append(engine_l_h)
                if engine_r_w > 0 and engine_r_w < 1:
                    engine_r_ws.append(engine_r_w)
                if engine_r_h > 0 and engine_r_h < 1:
                    engine_r_hs.append(engine_r_h)

                size_l_w = (engine_left['size'][0]) / size[0]
                size_l_h = (engine_left['size'][1]) / size[1]
                if size_l_w > 0 and size_l_w < 1:
                    engine_size_ws.append(size_l_w)
                if size_l_h > 0 and size_l_h < 1:
                    engine_size_hs.append(size_l_h)

                size_r_w = (engine_right['size'][0]) / size[0]
                size_r_h = (engine_right['size'][1]) / size[1]
                if size_r_w > 0 and size_r_w < 1:
                    engine_size_ws.append(size_r_w)
                if size_r_h > 0 and size_r_h < 1:
                    engine_size_hs.append(size_r_h)

    plt.xlabel('num')
    plt.ylabel('percentage')
    plt.plot(engine_l_ws)
    plt.show()
    print(np.mean(np.array(engine_l_ws)))






event_loss('5000_64_2_0.002_10')
# object_loss('../../data/log.txt')
# plane_analyse()