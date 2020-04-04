from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from classify.similarity import Frame_Queue
from classify.train import MAX_SIZE, WINDOWS, merge
from tools.logger import Logger
from demo import detection_demo
from lib.Detector import Detector
from lib.opts import opts


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


if __name__ == "__main__":
    opt = opts.init()
    main(opt)
