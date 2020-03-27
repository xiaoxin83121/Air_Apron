from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from classify.similarity import Frame_Queue
MAX_SIZE = 10
WINDOWS = 3

def merge(inputs):
    # merge the results
    pass

def classify_main():
    fq = Frame_Queue(max_size=MAX_SIZE, wind=WINDOWS)
    # TODO: 轮询式获取数据
    # example
    inp = None
    vec = merge(inp)
    # TODO: 加入网络训练
