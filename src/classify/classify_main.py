from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from classify.similarity import Frame_Queue
MAX_SIZE = 10
WINDOWS = 3

"""
event_dict record classes of event, and every event has a triple state:0 for begin;1 for end;2 for in_status
"""

EVENT_DICT = {
    '0': '飞机与空中扶梯连接',
    '1': '飞机与加油车连接',
    '2': '飞机与牵引车连接',
    '3': '客车运送乘客',
    '4': '乘客上飞机',
    '5': '起飞准备',  # state = 2
    '6': '下客准备',
    '7': '飞机与生活用车连接'
}


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
