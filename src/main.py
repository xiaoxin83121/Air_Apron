from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import json

from classify.similarity import Frame_Queue
from classify.config import MAX_SIZE, WINDOWS
from classify.train import merge
from classify.data_augment import res2vec, generate_dataset
from classify.rnn_classify import rnn_demo
from demo import detection_demo, demo
from lib.opts import opts
from lib.dataset.Pascal import PascalVOC

# sys.path.append("C:/User/13778/workshop/gitrepos/Air_Apron/src/")
sys.path.append("/gs/home/tongchao/zc/Air_Apron/src/")

def main(opt):
    # public
    fq = Frame_Queue(max_size=MAX_SIZE, wind=WINDOWS)
    # get image from live or video

    # img in opt.demo
    # rets = demo(opt)
    # with open('../data/rets.json', "w") as f:
    #     json.dump(rets, f)
    json_f = open('../data/rets.json')
    rets = json.load(json_f)
    # print(rets[0])
    path = opt.demo
    count = 0
    sequence = []
    classes = []
    for ret in rets:
        fq.ins(ret)
        result = fq.get_result()
        res, pos_res, size_res = merge(ret, result)
        # print(result)
        sample = res2vec(res, pos_res, size_res)
        if count == 0:
            for i in range(MAX_SIZE):
                sequence.append(sample)
        else:
            sequence.pop()
            sequence.append(sample)
        # 加入到分类网络中
        classification = rnn_demo(sample=[sequence], save_dir='classify/models/rnn/epoch_2000', latest_iter=2000)
        count += 1
        classes.append(classification)

    # print(classes)
    # eval
    _, labels, length = generate_dataset("../data/VOC2007_new/", "sequence_1.csv")
    res = classes.copy()

    for i in range(length):
        label = []
        for j in range(8):
            label.append(2*labels[i][j*2]+labels[i][j*2+1])
            if j == 7:
                label[7] = 2 if label[7] == 3 else label[7]
            res[i][j] = 1 if classes[i][j]==label[j] else 0
            # calculate percentage
    percentages = []
    r = np.array(res)
    for i in range(8):
        splice = r[:, i]
        percentages.append(np.mean(splice))
    print(percentages)



if __name__ == "__main__":
    opt = opts().parse()
    main(opt)
