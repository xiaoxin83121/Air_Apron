from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


"""
数据增广
"""
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import math
import pandas as pd
from classify.pre_process import single_process, mul_process, cal_distance, safe_area



def res2vec(res, pos_res, size_res):
    dis_std = 587
    center_std = 1024
    sample = list(res.values())
    pos_res = list(pos_res.values())
    size_res = list(size_res.values())
    for p in pos_res:
        sample.append(p[0])
        sample.append(p[1])
    for s in size_res:
        sample.append(s[0])
        sample.append(s[1])
    # split_dict means normalization regulation: value means cut_off
    # 0 for 0-1，no need to process; 1 for arctan, add pi/2 and divided by pi
    # 3 and 4 devided by size; 3 for 1 dimension and 4 for 2 dimension
    split_list = [8, 9, 23, 51, 61]
    sample[split_list[0]] = ( sample[split_list[0]] + math.pi / 2 ) / math.pi
    for i in range(split_list[1], split_list[2]):
        sample[i] = sample[i] / dis_std
    for i in range(split_list[2], split_list[3]):
        sample[i] = sample[i] / center_std
    for i in range(split_list[3], split_list[4]):
        sample[i] = sample[i] / dis_std
    return sample

def generate_dataset(dir, file_name):
    # 从data中获取事件检测所用的数据集
    samples = []
    labels = []
    file_path = os.path.join(dir, file_name)
    sequence = pd.read_csv(file_path).values
    frame_sequence = set()
    for s in sequence:
        label = [int(siter) for siter in s[0].split('\t')]
        # if label[0] not in frame_sequence:
        frame_sequence.add(label[0])
        one_hot = []
        for l in label[1:9]:
            if l == 0:
                one_hot += [0, 0]
            elif l == 1:
                one_hot += [0, 1]
            elif l == 2:
                one_hot += [1, 0]
            elif l == 3:
                one_hot += [1, 1]
        labels.append(one_hot)
    # print(len(frame_sequence))
    length = len(frame_sequence)
    for i in range(length):
        sample = generate_test(os.path.join(dir, 'Annotations'), i)
        res, pos_res, size_res = merge(sample, single_process(sample))
        sample = res2vec(res, pos_res, size_res)
        samples.append(sample)
    return samples, labels, length


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
        try:
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
        except Exception as e:
            return []


def ver_dis(point, line_set):
    # vertical distance from point to lines
    diss = []
    for line in line_set['area']:
        if line[0][0] - line[1][0] <= 1:
            dis = abs(point[0] - line[0][0])
        else:
            k = (line[0][1] - line[1][1]) / (line[0][0] - line[1][0])
            A = -k
            C = k * line[0][0] - line[0][1]
            dis = abs(A * point[0] + point[1] + C) / \
                math.sqrt(A**2 + 1)
        diss.append(dis)
    return min(diss) if not (not diss) else -1, max(diss) if not (not diss) else -1

def merge(inputs, sgl):
    # para:inputs is the result of object detection per frame
    mul = mul_process(inputs)
    sa = safe_area(inputs)
    # bus*2 & queue*2 & person*5
    bus_list = mul['bus']
    queue_list = mul['queue']
    pb =[[0, 0], [0, 0]]
    pq = [[0, 0], [0, 0]]
    pp = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    dps_min = [0, 0, 0, 0, 0]
    dps_max = [0, 0, 0, 0, 0]
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
        'is_stair': 1 if sgl['is_stair'] else 0, 'is_traction': 1 if sgl['is_traction'] else 0,
        'is_cargo': 1 if sgl['is_cargo'] else 0,
        'is_person': 1 if mul['is_person'] else 0, 'is_queue': 1 if mul['is_queue'] else 0,
        'is_bus': 1 if mul['is_bus'] else 0,
        'horizon': sgl['plane']['horizon'], # [arctan -pi/2~pi/2]
        'plane2oil': cal_distance(sgl['plane'], sgl['oil_car']) if sgl['is_plane'] and sgl['is_oil_car'] else 0,
        'plane2stair': cal_distance(sgl['plane'], sgl['stair']) if sgl['is_plane'] and sgl['is_stair'] else 0,
        'plane2traction': cal_distance(sgl['plane'], sgl['traction']) if sgl['is_plane'] and sgl['is_traction'] else 0,
        'plane2cargo': cal_distance(sgl['plane'], sgl['cargo']) if sgl['is_plane'] and sgl['is_cargo'] else 0,
        # person to safe_area distance
        'dps_min1': dps_min[0], 'dps_min2': dps_min[1], 'dps_min3': dps_min[2],
        'dps_min4': dps_min[3], 'dps_min5': dps_min[4],
        'dps_max1': dps_max[0], 'dps_max2': dps_max[1], 'dps_max3': dps_max[2],
        'dps_max4': dps_max[3], 'dps_max5': dps_max[4]
    }
    pos_res = {
        'pb1': pb[0], 'pb2': pb[1],
        'pq1': pq[0], 'pq2': pq[1],
        'pp1': pp[0], 'pp2': pp[1], 'pp3': pp[2], 'pp4': pp[3], 'pp5': pp[4],
        'pplane': sgl['plane']['center'] if sgl['is_plane'] else [0, 0],
        'po': sgl['oil_car']['center'] if sgl['is_oil_car'] else [0, 0],
        'ps': sgl['stair']['center'] if sgl['is_stair'] else [0, 0],
        'pt': sgl['traction']['center'] if sgl['is_traction'] else [0, 0],
        'pc': sgl['cargo']['center'] if sgl['is_cargo'] else [0, 0]
    }
    size_res = {
        'sp': sgl['plane']['size'] if sgl['is_plane'] else [0, 0],
        'so': sgl['oil_car']['size'] if sgl['is_oil_car'] else [0, 0],
        'ss': sgl['stair']['size'] if sgl['is_stair'] else [0, 0],
        'st': sgl['traction']['size'] if sgl['is_traction'] else [0, 0],
        'sc': sgl['cargo']['size'] if sgl['is_cargo'] else [0, 0]
    }
    return res, pos_res, size_res


def data_augument(seq_dir, anno_dir, csv_name):
    # 对数据集进行数据增广
    seq_path = os.path.join(seq_dir, csv_name)
    sequence = pd.read_csv(seq_path).values
    frame_sequence = set()
    labels = []
    samples = []
    for s in sequence:
        label = [int(siter) for siter in s[0].split('\t')]
        frame_sequence.add(label[0])
        one_hot = []
        for l in label[1:9]:
            if l == 0:
                one_hot += [0, 0]
            elif l == 1:
                one_hot += [0, 1]
            elif l == 2:
                one_hot += [1, 0]
            elif l == 3:
                one_hot += [1, 1]
        labels.append(one_hot)
    length = len(frame_sequence)
    state = 0
    for i in range(length):
        sample = generate_test(anno_dir, i)
        sgl = single_process(sample)
        res, pos_res, size_res = merge(sample, sgl)
        if res['is_bus'] == 1:
            state = 1
        if res['is_stair'] == 0:
            state = 2
        if res['is_plane'] == 0:
            state = 0
        interim_vec = generate_interim_vector(state, sample)
        # print('iv={} sample={}'.format(interim_vec, sample))
        sample = recur_sample_label(interim_vec, sample)
        res, pos_res, size_res = merge(sample, single_process(sample))
        sample = res2vec(res, pos_res, size_res)
        samples.append(sample)
    # print(sequence)
    return {'samples': samples, 'labels': labels, 'length':length}


def generate_interim_vector(state, sample):
    """generate interim vector to make sure changes of samples and labels
    interim_vec_sample = [{'on_off':1, 'center_bias':[a,b], 'size_bias':[a,b], }...]
    add_vec_sample = [{'class':'stair', }] ## ignore
    """
    # 不同阶段的噪声控制normal参数 [avg=0, var]
    state_p_dict = [
        [4, 36],
        [1, 4],
        [9, 64]
    ]
    on_off_dict = [
        0.975, 0.99, 0.985
    ]

    interim_vec_sample = []
    length = len(sample)
    guassian_center = np.random.normal(0, state_p_dict[state][0], (length, 2))
    guassian_size = np.random.normal(0, state_p_dict[state][1], (length, 2))
    on_off = []
    for r in np.random.random(length):
        on_off.append(1 if r > on_off_dict[state] else 0)
    # print(guassian_center)
    # print(guassian_size)
    # print(on_off)
    for iter in range(length):
        iter_dic = dict()
        iter_dic['on_off'] = on_off[iter]
        iter_dic['center_bias'] = guassian_center[iter]
        iter_dic['size_bias'] = guassian_size[iter]
        interim_vec_sample.append(iter_dic)
    return interim_vec_sample


def recur_sample_label(interim_vec, samples):
    # print(samples)
    iter = 0
    while(iter < len(samples)):
        vec = interim_vec[iter]
        sample = samples[iter]
        if vec['on_off'] == 1:
            del samples[iter]
        else:
            center_bias = vec['center_bias']
            size_bias = vec['size_bias']
            sample['center'] = [ int(sample['center'][0]+center_bias[0]), int(sample['center'][1]+center_bias[1]) ]
            sample['size'] = [ int(sample['size'][0]+size_bias[0]), int(sample['size'][1]+size_bias[1]) ]
            iter += 1
    return samples


if __name__ == "__main__":
    # test function
    # samples, labels = generate_dataset('../../data/VOC2007_new')
    # print('len_s={} len_l={}'.format(len(samples), len(labels)))
    res = data_augument(seq_dir='../../data/VOC2007_new', anno_dir='../../data/VOC2007_new/Annotations',
                        csv_name='sequence1.csv')
    # print(res)