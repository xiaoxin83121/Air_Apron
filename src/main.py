from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from classify.similarity import Frame_Queue
from classify.train import MAX_SIZE, WINDOWS, merge
from tools.logger import Logger

def main(opt):
    # public
    fq = Frame_Queue(max_size=MAX_SIZE, wind=WINDOWS)
    # get image from live or video



    # TODO: 轮询式获取数据
    # example
    inp = None
    vec = merge(inp)
