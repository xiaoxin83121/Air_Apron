from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from classify.similarity import Frame_Queue
from classify.train import MAX_SIZE, WINDOWS, merge
from classify.rnn_classify import rnn_demo
from classify.vote_classify import vote
from tools.logger import Logger
from demo import detection_demo
from lib.Detector import Detector
from lib.opts import opts

# sys.path.append("C:/User/13778/workshop/gitrepos/Air_Apron/src/")
sys.path.append("/gs/home/tongchao/zc/Air_Apron/src/")

def main(opt):
    # public
    detector = Detector(opt)
    fq = Frame_Queue(max_size=MAX_SIZE, wind=WINDOWS)
    # get image from live or video
    img = None
    rets, time_str = detection_demo(detector, img)
    fq.ins(rets)
    res = fq.get_result()
    res, pos_res = merge(rets, res)
    sample = res.values()
    sample.append(p for p in pos_res)
    # 加入到分类网络中
    res_rnn = rnn_demo(sample=sample, save_dir='classify/models/rnn/', latest_iter=5000)
    res_vote = vote(sample=sample, save_dir='classify/models/vote/')
    print("rnn result is {}, and vote result is {}".format(res_rnn, res_vote))


if __name__ == "__main__":
    opt = opts.init()
    main(opt)
