from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
import json


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
        plt.plot(epoch_loss)
        plt.show()


def precision_recall():
    pass


def plane_analyse():
    pass


def single_bbox_joggle():
    pass



event_loss('epoch_1000_modify_0')
# object_loss('../../data/log.txt')