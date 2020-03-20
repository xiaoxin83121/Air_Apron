from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import os
from pycocotools import coco
import numpy as np


class PascalVOC(data.dataset):
    num_classes = 12
    default_resolution = [1920, 1080]
    # copy
    mean = np.array([0.485, 0.456, 0.406],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(PascalVOC, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'VOC2007')
        self.img_dir = os.path.join(self.data_dir, 'JPEGImages')
        self.annot_path = os.path.join(self.data_dir, 'Annotations')
        self.max_objs = 5
        self.class_name = ['__background__', 'plane', 'head', 'wheel', 'wings', 'stair',
                           'oil_car', 'person', 'cone', 'engine', 'traction', 'bus', 'queue']

        self.opt = opt
        self.split = split
        self.coco = coco.COCO(self.annot_path)
        self.num_samples = len(os.listdir(self.annot_path))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        # 生成一个图片及其他信息的result
        if self.split == 'train':
            pass
        else:
            pass
