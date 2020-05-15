from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import classify.config as config
import copy

"""
位置相似度检测算法
"""
from classify.pre_process import single_process, cal_distance



class Frame_Queue(object):
    # 维护一个视频帧信息队列
    def __init__(self, max_size, wind):
        super(Frame_Queue, self).__init__()
        self.max_size = max_size
        self.wind = wind
        self.q = list()  # 整个queue
        self.cache = list()  # 最近的WINDOWS个窗口
        self.objs = ['traction', 'oil_car', 'plane', 'stair', 'cargo']

    def ins(self, inputs):
        res = single_process(inputs)
        # print("1={}".format(res['oil_car']))
        res1 = copy.deepcopy(res)
        if len(self.q) >= self.max_size:
            self.q.pop(0)
        self.q.append(res1)
        if len(self.cache) >= self.wind:
            self.cache.pop(0)
        self.cache.append(res)
        if len(self.q) > self.wind:
            for obj in self.objs:
                self.split_deal(obj)
        # print("2={}".format(res['oil_car']))
        # print(self.q[-1]['oil_car'])
        # print("2={}".format(self.cache[-1]['oil_car']))

    def split_deal(self, split):
        is_str = 'is_' + split
        if not self.windows_empty(split):  # 如果该split临近窗口内不全为空
            width, height, size_0, size_1, count = self.cal_means(split)
            if self.q[-1][is_str]:  # 如果当前帧存在split
                self.q[-1][split]['center'][0] = int(width * (count / (count+1)) + \
                                                                self.q[-1][split]['center'][0] * \
                                                                (1 / (count+1) ))
                self.q[-1][split]['center'][1] = int(height * (count / (count+1)) + \
                                                                  self.q[-1][split]['center'][1] *
                                                                (1 / (count+1)))
                self.q[-1][split]['size'][0] = int(size_0 * (count / (count + 1)) + \
                                                     self.q[-1][split]['size'][0] * \
                                                     (1 / (count + 1)))
                self.q[-1][split]['size'][1] = int(size_1 * (count / (count + 1)) + \
                                                     self.q[-1][split]['size'][1] *
                                                     (1 / (count + 1)))
            else:  # 当前帧不存在split
                self.q[-1][split]['center'][0] = width
                self.q[-1][split]['center'][1] = height
                self.q[-1][split]['size'][0] = size_0
                self.q[-1][split]['size'][1] = size_1
                self.q[-1][is_str] = True
            # if split == 'oil_car':
            #     print("w={} h={} c={} ct={}".format(width, height, count, self.q[-1][split]))
        else:
            # self.q[-1][split]['center'] = [0, 0]
            # self.q[-1][split]['size'] = [0, 0]
            self.q[-1][is_str] = False

    def windows_empty(self, split):
        # return False means not all items in windows are empty
        is_str = 'is_' + split
        # for i in range(len(self.q)-self.wind, len(self.q)):
        #     if self.q[i][is_str]:
        #         return False
        # return True
        for i in range(0, len(self.cache)):
            if self.cache[i][is_str] and self.cache[i][split]['score'] > 0.1:
                return False
        return True

    def cal_means(self, split):
        # split='traction' or 'plane' etc;  n means len(self.q) - 1
        # only work when windows_empty==False
        count = 0  # 多少帧有这个目标
        sum_width, sum_height = [0, 0]
        size_0, size_1 = [0, 0]
        for i in range(len(self.q) - 1):
            is_str = 'is_' + split
            if self.q[i][is_str]:  # False帧不计算
                count += 1
                sum_width += self.q[i][split]['center'][0]
                sum_height += self.q[i][split]['center'][1]
                size_0 += self.q[i][split]['size'][0]
                size_1 += self.q[i][split]['size'][1]
        if count != 0:
            sum_width = sum_width // count
            sum_height = sum_height // count
            size_0 = size_0 // count
            size_1 = size_1 // count
        return sum_width, sum_height, size_0, size_1, count
        # return center

    def is_move(self, split):
        # 判断当前有没有动
        # after self.ins
        if len(self.q) < self.wind:
            return False
        for i in range(self.wind):
            # TODO: 存在q或者是cache该split下center是[0,0]的问题
            if cal_distance(self.q[len(self.q)-self.wind+i][split], self.cache[i][split]) <= config.move_Distance:
                return False
        return True

    def fresh(self, split):
        # work when self.is_move==True; refresh the self.q
        self.q[-1][split] = self.cache[-1][split]

    def get_result(self):
        # get the final result
        # 存在一种is_move=True 但empty_wind=False, q[-1]为修正值；cache[-1]为空。此时不能fresh
        # for obj in self.objs:
        #     if self.is_move(split=obj) and self.cache[-1]['is_'+obj]:
        #         self.fresh(split=obj)
        return self.q[-1]

