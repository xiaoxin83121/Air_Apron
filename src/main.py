from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from classify.similarity import Frame_Queue
from classify.config import MAX_SIZE, WINDOWS
from classify.train import merge
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
    state = 0
    # while循环
    img = None
    rets, time_str = detection_demo(detector, img)
    fq.ins(rets)
    res = fq.get_result()
    res, pos_res, size_res = merge(rets, res)
    if res['is_bus'] == 1:
        state = 1
    if res['is_stair'] == 0:
        state = 2
    if res['is_plane'] == 0:
        state = 0
    sample = res.values()
    for p in pos_res:
        sample.append(p[0])
        sample.append(p[1])
    for s in size_res:
        sample.append(s[0])
        sample.append(s[1])

    # 加入到分类网络中
    # state中，第一阶段以bus出现位分界点；第二阶段以stair消失位分界点
    res_rnn = rnn_demo(sample=sample, save_dir='classify/models/rnn/', latest_iter=5000)
    res_vote = vote(sample=sample, save_dir='classify/models/vote/')
    print("rnn result is {}, and vote result is {}".format(res_rnn, res_vote))


if __name__ == "__main__":
    opt = opts.init()
    main(opt)
