from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from classify.pre_process import single_process, mul_process, safe_area, cal_distance
from classify.data_augment import generate_dataset, generate_test, merge
from classify.rnn_classify import rnn_train, rnn_eval, rnn_demo
from classify.vote_classify import Vote_Net
MAX_SIZE = 10
WINDOWS = 3
INP_SIZE = 55
OUT_SIZE = 3

"""
event_dict record classes of event, and every event has a triple state:0 for begin;1 for end;2 for in_status
event_warning: 0:events ; 1:warning; 2:preparing
type: 0:EVENT_DICT;  1:0-警告，1-提醒，2-安全; 2:0-起飞准备，1-降落准备
state: 0: begin; 1: in; 2: end
"""

EVENT_DICT = {
    '0': '飞机与空中扶梯连接',
    '1': '飞机与加油车连接',
    '2': '飞机与牵引车连接',
    '3': '客车运送乘客',
    '4': '乘客上飞机',
    '5': '飞机与生活用车连接'
}


def classify_train(dir):
    net_paras = [[INP_SIZE, OUT_SIZE, 10, 5, 0, 0.5], [INP_SIZE, OUT_SIZE, 10, 4, 1, 0.5]]
    samples, labels, length = generate_dataset(dir)
    print(labels)
    sequence_size = 10
    sequence_dic = {'samples_seg':[], 'labels_seg':[]}
    for i in range(length - sequence_size):
        samples_seg = samples[i:i+sequence_size]
        labels_seg = labels[i+sequence_size]
        sequence_dic['samples_seg'].append(samples_seg)
        sequence_dic['labels_seg'].append(labels_seg)

    # train rnn_net
    rnn_train(sequence_dic['samples_seg'], sequence_dic['labels_seg'], 'models/rnn/',
              5000, INP_SIZE, OUT_SIZE)
    # eval rnn_net

    # train vote_net
    vn = Vote_Net(net_paras=net_paras)
    # vn.train(sequence_dic['samples_seg'], sequence_dic['labels_seg'], 'classify/models/vote/',
    #          5000, INP_SIZE, OUT_SIZE)
    # eval vate_net

if __name__ == "__main__":
    # test_inps = generate_test('C:/Users/13778/workshop/gitrepos/Air_Apron/data/VOC2007/Annotations/', 100)
    # sgl = single_process(test_inps)
    # print(sgl)
    # mul = mul_process(test_inps)
    # print(mul)
    classify_train('../../data/VOC2007_new/')

