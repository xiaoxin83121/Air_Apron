from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

"""
数据增广
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import math
import pandas as pd
from classify.pre_process import single_process, mul_process, cal_distance, safe_area


def generate_dataset(dir):
    # 从data中获取事件检测所用的数据集
    samples = []
    labels = []
    file_path = os.path.join(dir, 'sequence.csv')
    sequence = pd.read_csv(file_path).values
    length = len(sequence)
    for i in range(length):
        sample = generate_test('data/VOC2007/Annotations/', i)
        res, pos_res = merge(sample, single_process(sample))
        sample = res.values()
        for p in pos_res:
            sample.append(p[0])
            sample.append(p[1])
        label = sequence[i]
        samples.append(sample)
        labels.append(label)
    #TODO: normalize
    return samples, labels


def data_augument():
    # 对数据集进行数据增广
    pass


def generate_test(dirs, indexs):
    # para:index 列表，存了读哪些xml文件; int, 存读哪个文件
    res = dict()
    if isinstance(indexs, list):
        for i in indexs:
            filename = str(i) + '.xml'
            filepath = os.path.join(dirs, filename)
            domtree = ET.parse(filepath)
            root = domtree.getroot()
            objs = root.findall('object')
            dic_index = []
            for obj in objs:
                cls = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                size = [xmax - xmin, ymax - ymin]
                center = [xmax - size[0] // 2, ymax - size[1] // 2]
                bbox = [[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]]
                dic = {'class': cls, 'bbox': bbox, 'center': center, 'size': size}
                dic_index.append(dic)
            res[str(i)] = dic_index
        return res
    else:
        filename = str(indexs) + '.xml'
        filepath = os.path.join(dirs, filename)
        domtree = ET.parse(filepath)
        root = domtree.getroot()
        objs = root.findall('object')
        dic_index = []
        for obj in objs:
            cls = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            size = [xmax - xmin, ymax - ymin]
            center = [xmax - size[0] // 2, ymax - size[1] // 2]
            bbox = [[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]]
            dic = {'class': cls, 'bbox': bbox, 'center': center, 'size': size}
            dic_index.append(dic)
        return dic_index


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

def merge(inputs, sgl):
    # para:inputs is the result of object detection per frame
    mul = mul_process(inputs)
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
        'dps_max4': dps_max[3], 'dps_max5': dps_max[4],
        'horizon': sgl['plane']['horizon']
    }
    pos_res = {
        'pb1': pb[0], 'pb2': pb[1],
        'pq1': pq[0], 'pq2': pq[1],
        'pp1': pp[0], 'pp2': pp[1], 'pp3': pp[2], 'pp4': pp[3], 'pp5': pp[4],
        'pplane': sgl['plane']['center'], 'po': sgl['oil_car']['center'],
        'ps': sgl['stair']['center'], 'pt': sgl['tractor']['center']
    }
    return res, pos_res

