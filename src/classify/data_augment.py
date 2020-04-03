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
import pandas as pd
from classify.train import merge


def generate_dataset(dir):
    # 从data中获取事件检测所用的数据集
    samples = []
    labels = []
    file_path = os.path.join(dir, 'sequence.csv')
    sequence = pd.read_csv(file_path).values
    length = len(sequence)
    for i in range(length):
        sample = generate_test('data/VOC2007/Annotations/', i)
        res, pos_res = merge(sample)
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


