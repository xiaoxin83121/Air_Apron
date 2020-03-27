from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

"""
位置相似度检测算法
"""
from classify.pre_process import single_process, cal_distance
move_Distance = 10


class Frame_Queue(object):
    # 维护一个视频帧信息队列
    def __init__(self, max_size, wind):
        super(Frame_Queue, self).__init__()
        self.max_size = max_size
        self.wind = wind
        self.q = list()  # 整个queue
        self.cache = list()  # 最近的WINDOWS个窗口
        self.objs = ['tractor', 'oil_car', 'plane', 'stair']

    def ins(self, inputs):
        res = single_process(inputs)
        if len(self.q) >= self.max_size:
            self.q.pop(0)
        self.q.append(res)
        if len(self.cache) >= self.wind:
            self.q.pop(0)
        self.cache.append(res)
        if len(self.q) > self.wind:
            for obj in self.objs:
                self.split_deal(obj)

    def split_deal(self, split):
        if not self.windows_empty(split):
            center = self.cal_means(split)
            self.q[len(self.q)-1][split]['center'][0] = int(center[0] * (len(self.q)-1 / len(self.q)) + \
                                                            self.q[len(self.q)-1][split]['center'][0] * \
                                                            (1 / len(self.q)))
            self.q[len(self.q) - 1][split]['center'][1] = int(center[1] * (len(self.q) - 1 / len(self.q)) + \
                                                              self.q[len(self.q) - 1][split]['center'][1] *
                                                            (1 / len(self.q)))

    def windows_empty(self, split):
        # return False means not all items in windows are empty
        is_str = 'is_' + split
        for i in range(len(self.q)-self.wind, len(self.q)):
            if self.q[i][is_str]:
                return False
        return True

    def cal_means(self, split):
        # split='tractor' or 'plane' etc;  n means len(self.q) - 1
        # only work when windows_empty==False
        count = 0  # 多少帧有这个目标
        sum_width, sum_height = [0, 0]
        for i in range(len(self.q) - 1):
            is_str = 'is_' + split
            if self.q[i][is_str]:  # False帧不计算
                count += 1
                sum_width += self.q[i][split]['center'][0]
                sum_height += self.q[i][split]['center'][1]
        center = [sum_width // count, sum_height // count]

    def is_move(self, split):
        # 判断当前有没有动
        # after self.ins
        for i in range(self.wind):
            if cal_distance(self.q[len(self.q)-self.wind+i], self.cache[i]) <= move_Distance:
                return False
        return True

    def fresh(self, split):
        # work when self.is_move==True; refresh the self.q
        self.q[len(self.q)-1][split] = self.cache[len(self.cache)-1][split]

    def get_result(self):
        # get the final result
        move_res = {}
        for obj in self.objs:
            move_res.update({obj+'_move':self.is_move(split=obj)})
            if self.is_move(split=obj):
                self.fresh(split=obj)
        return self.q[len(self.q)-1].update(move_res)

