from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import json
import cv2
import torch.utils.data as data
import os

"""
# TODO: process data
"""
live_path = '../../data/Live_demo_20200117/'

video_text_path = live_path+ 'video_text/'
video_annotation_path = live_path + 'video_annotation/'
video_path = live_path + 'video/'


class PascalVOC(data.dataset):
    num_classes = 23
    default_resolution = [1080, 1440]

    def __init__(self, opt, split):
        super(PascalVOC, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'VOC2007')
        self.img_dir = os.path.join(self.data_dir, 'JPEGImages')
        self.annot_path = os.path.join(self.data_dir, 'Annotations')
        self.max_objs = 5
        self.class_name = ['__background__', '短袖上衣', '长袖上衣', '短袖衬衫']
