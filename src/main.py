from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from classify.similarity import Frame_Queue
from classify.config import MAX_SIZE, WINDOWS
from classify.train import merge
from classify.data_augment import res2vec
from classify.rnn_classify import rnn_demo
from classify.vote_classify import vote
from tools.logger import Logger
from demo import detection_demo, demo
from lib.Detector import Detector
from lib.opts import opts

# sys.path.append("C:/User/13778/workshop/gitrepos/Air_Apron/src/")
sys.path.append("/gs/home/tongchao/zc/Air_Apron/src/")

def main(opt):
    # public
    fq = Frame_Queue(max_size=MAX_SIZE, wind=WINDOWS)
    # get image from live or video

    # while循环
    # img in opt.demo
    rets = demo(opt)
    count = 0
    sequence = []
    for ret in rets:
        fq.ins(ret)
        result = fq.get_result()
        res, pos_res, size_res = merge(ret, result)
        sample = res2vec(res, pos_res, size_res)
        if count == 0:
            for i in range(MAX_SIZE):
                sequence.append(sample)
        else:
            sequence.pop()
            sequence.append(sample)
        # 加入到分类网络中
        classification = rnn_demo(sample=sequence, save_dir='classify/models/rnn/epoch_2000', latest_iter=2000)

    # eval

if __name__ == "__main__":
    opt = opts().init()
    main(opt)
