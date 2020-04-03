from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
from classify.pre_process import single_process, mul_process, safe_area, cal_distance
from classify.data_augment import generate_dataset
from classify.rnn_classify import rnn_train, rnn_eval, rnn_demo
from classify.vote_classify import Vote_Net
MAX_SIZE = 10
WINDOWS = 3

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


def ver_dis(point, line_set):
    # vertical distance from point to
    diss = []
    for line in line_set:
        k = (line[0][1] - line[1][1]) / (line[0][0] - line[1][0])
        A = -k
        C = k * line[0][0] - line[0][1]
        dis = abs(A * point[0] + point[1] + C) / \
            math.sqrt(A**2 + 1)
        diss.append(dis)
    return min(diss), max(diss)

def merge(inputs):
    # para:inputs is the result of object detection per frame
    mul = mul_process(inputs)
    sgl = single_process(inputs)
    sa = safe_area(inputs)
    # bus*2 & queue*2 & person*5
    bus_list = mul['bus']
    queue_list = mul['queue']
    pb =[[-1, -1], [-1, -1]]
    pq = [[-1, -1], [-1, -1]]
    pp = [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]
    dps_min = [-1, -1, -1, -1, -1]
    dps_max = [-1, -1, -1, -1, -1]
    for i in range(min(2, len(bus_list))):
        pb[i] = bus_list[i]['center']
    for i in range(min(2, len(queue_list))):
        pq[i] = queue_list[i]['center']
    for i in range(min(5, len(mul['person_ids']))):
        pp[i] = mul['person'][mul['person_ids'][i]]['center']
        dmin, dmax = ver_dis(pp[i], sa)
        dps_min[i] = dmin
        dps_max[i] = dmax
    res = {
        'is_plane': 1 if sgl['is_plane'] else 0, 'is_oil_car': 1 if sgl['is_oil_car'] else 0,
        'is_stair': 1 if sgl['is_stair'] else 0, 'is_tractor': 1 if sgl['is_tractor'] else 0,
        'is_person': 1 if mul['is_person'] else 0, 'is_queue': 1 if mul['is_queue'] else 0,
        'is_bus': 1 if mul['is_bus'] else 0,
        'plane2oil': cal_distance(sgl['plane'], sgl['oil_car']) if sgl['is_plane'] and sgl['is_oil_car'] else -1,
        'plane2stair': cal_distance(sgl['plane'], sgl['stair']) if sgl['is_plane'] and sgl['is_stair'] else -1,
        'plane2tractor': cal_distance(sgl['plane'], sgl['tractor']) if sgl['is_plane'] and sgl['is_tractor'] else -1,
        'dps_min1': dps_min[0], 'dps_min2': dps_min[1], 'dps_min3': dps_min[2],
        'dps_min4': dps_min[3], 'dps_min5': dps_min[4],
        'dps_max1': dps_max[0], 'dps_max2': dps_max[1], 'dps_max3': dps_max[2],
        'dps_max4': dps_max[3], 'dps_max5': dps_max[4]
    }
    pos_res = {
        'pb1': pb[0], 'pb2': pb[1],
        'pq1': pq[0], 'pq2': pq[1],
        'pp1': pp[0], 'pp2': pp[1], 'pp3': pp[2], 'pp4': pp[3], 'pp5': pp[4],
        'pplane': sgl['plane']['center'], 'po': sgl['oil_car']['center'],
        'ps': sgl['stair']['center'], 'pt': sgl['tractor']['center']
    }
    return res, pos_res


def classify_train():
    net_paras = [[46, 4, 10, 5, 0, 0.5], [46, 4, 10, 4, 1, 0.5]]
    samples, labels = generate_dataset('data/VOC2007/')
    length = len(labels)
    sequence_size = 10
    sequence_dic = {'samples_seg':[], 'labels_seg':[]}
    for i in range(length - sequence_size):
        samples_seg = samples[i:i+sequence_size]
        labels_seg = labels[i:i+sequence_size]
        sequence_dic['samples_seg'].append(samples_seg)
        sequence_dic['labels_seg'].append(labels_seg)

    # train rnn_net
    rnn_train(sequence_dic['samples_seg'], sequence_dic['labels_seg'], 'classify/models/rnn/', 5000, 46, 4)
    # eval rnn_net

    # train vote_net
    vn = Vote_Net(net_paras=net_paras)
    vn.train(5000, 'classify/models/vote/', sequence_dic['samples_seg'], sequence_dic['labels_seg'])
    # eval vate_net

if __name__ == "__main__":
    classify_train()



