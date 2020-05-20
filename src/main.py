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
from classify.pre_process import single_process

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
    predictions = []
    for ret in rets:
        fq.ins(ret)
        result = fq.get_result()
        # print("out_side={}".format(single_process(ret)['oil_car']))
        # print("frame {} : res= {} ret={} is_res={} is_ret={}".format(count,
        #             result['oil_car'], single_process(ret)['oil_car'], result['is_oil_car'],
        #                                                              single_process(ret)['is_oil_car']))
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
        classification, prediction = rnn_demo(sample=[sequence], save_dir='classify/models/rnn/epoch_1000_modify_0',
                                              latest_iter=2000)
        print(classification)
        count += 1
        classes.append(classification)
        predictions.append(prediction)

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
    print("acu={} mean={}".format(percentages, np.mean(np.array(percentages))))

    precisions = []
    recalls = []
    p = np.array(predictions)
    l = np.array(labels)
    for i in range(16):
        p_splice = p[:, i]
        l_splice = l[:, i]
        length = len(p_splice)
        tp, fp, fn, tn = (0, 0, 0, 0)
        for j in range(length):
            if p_splice[j] == 1 and l_splice[j] == 1:
                tp += 1
            elif p_splice[j] == 1 and l_splice[j] == 0:
                fp += 1
            elif p_splice[j] == 0 and l_splice[j] == 1:
                tn += 1
            else:
                fn += 1
        # print("tp={}, fp={}, fn={}, tn={}".format(tp, fp, fn, tn))
        precisions.append(tp / (tp + fp) if tp+fp != 0 else 0)
        recalls.append(tp / (tp + fn) if tp+fn != 0 else 0)
    print("prec={} mean={}".format(precisions, np.mean(np.array(precisions))))
    print('rec={} mean={}'.format(recalls, np.mean(np.array(recalls))))


if __name__ == "__main__":
    opt = opts().parse()
    main(opt)
