from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import cv2

input_w, input_h = (1920, 1080)
Distance = 200
alpha = 0.5
person_Distance = 10
People_Num = 3

"""
要完成的任务是：飞机的姿态检测，对input进行预处理
v1.0 理想情况
inputs should be below per frame: [{
    'class': 'person',
    'bbox': [[x1, y1], [x1, y2], [x2, y1], [x2, y2]],
    'center': [a, b], 
    'size': [w, h]
}]
"""


def cal_size(elem):
    return elem['size'][0] * elem['size'][1]


def cal_distance(elem1, elem2):
    c1 = elem1['center']
    c2 = elem2['center']
    return math.sqrt( (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 )


def center_in(center, corner1, corner2):
    return center[0] > corner1[0] and center[0] < corner2[0] and center[1] > corner1[1] and center[1] < corner2[1]


def find_max(inputs, split):
    _lists = []
    for inp in inputs:
        if inp['class']==split:
            _lists.append(inp)
    sorted_list = sorted(_lists, key=cal_size, reverse=True)
    return sorted_list[0]


def find_lowest(inputs):
    # return the lowest object in pic
    _list = [inp['center'][1] for inp in inputs]
    index = _list.index(min(_list))
    return index


def dfs(id, item_list, uncount_set):  # TODO:Debug
    # given an id, return the remain_uncount_set and the already count id_list
    ids = set()
    for i in uncount_set:
        if cal_distance(item_list[id], item_list[i]) <= person_Distance and id != i:
            uncount_set.remove(i)
            rids, uncount_set = dfs(i, item_list, uncount_set)
            ids = set.union(rids, ids)
    ids.add(id)
    return ids, uncount_set


def queue_size(ids, item_list):
    left = []
    up = []
    right = []
    down = []
    for id in ids:
        left.append(item_list[id]['bbox'][0][0])
        up.append(item_list[id]['bbox'][0][1])
        right.append(item_list[id]['bbox'][3][0])
        down.append(item_list[id]['bbox'][3][1])
    return [ max(right) - min(left), max(down) - min(up)]


def plane_pose(inputs):
    """
    飞机姿态信息处理
    """
    plane_dict = {'plane': [], 'engine': [], 'wings': [], 'wheel': [], 'head': []}
    for inp in inputs:
        cls = inp['class']
        if cls in plane_dict.keys():
            plane_dict[cls].append(inp)
    # print(plane_dict)

    # 判断飞机主体是否存在
    if len(plane_dict['plane']) < 1:
        # print("No plane detected")
        plane = None
        plane_exist = False
    else:
        # 找到size最大的检测框
        plane_sorted = sorted(plane_dict['plane'], key=cal_size, reverse=True)
        plane = plane_sorted[0]
        plane_exist = True

    # 认为head的识别率会比较高
    if len(plane_dict['head']) < 1:
        if not plane_exist:
            plane_exist = False
            # print("No plane and No head!")
            center = [-1, -1]
        # 否则按照原先的plane位置
    else:
        head_sorted = sorted(plane_dict['head'], key=cal_size, reverse=True)
        head = head_sorted[0]
        if plane_exist:
            center = head['center']
            left_up = plane['bbox'][0]
            right_down = plane['bbox'][3]
            if not center_in(center, left_up, right_down):
                plane_exist = False
            else:
                new_width = max(center[0] - left_up[0], right_down[0] - center[0])
                new_height = max(center[1] - left_up[1], right_down[1] - center[1])
                plane['center'] = center if cal_distance(head, plane) >= Distance else \
                    [(1-alpha)*center[0]+alpha*plane['center'][0], (1-alpha)*center[1]+alpha*plane['center'][1]]
                plane['size'] = [new_width * 2, new_height * 2]


    horizon = -1
    ground = [-1, -1]
    if plane_exist:
        if len(plane_dict['engine']) >= 2:
            engine1 = plane_dict['engine'][0]['center']
            engine2 = plane_dict['engine'][1]['center']
            dividend = ( abs(engine1[1] - engine2[1]) ) / ( abs(engine1[0] - engine2[0]) )
            horizon = math.atan(dividend)
        if len(plane_dict['wheel']) >= 1:
            # 找到画面最下方的轮
            index = find_lowest(plane_dict['wheel'])
            ground = plane_dict['wheel'][index]['center']

    res = {
        'class': 'plane',
        'center': plane['center'],
        'size': plane['size'],
        'horizon': horizon,
        'ground': ground
    } if plane_exist==True else\
    {
        'class':'', 'center': [], 'size': [], 'horizon': -1, 'ground': ground
        }

    return res, plane_exist


def single_process(inputs):
    cls = set()
    for inp in inputs:
        cls.add(inp['class'])
    # print(cls)
    empty_dict = {'class':'', 'bbox':[], 'center':[0, 0], 'size': [0, 0]}
    is_oilcar = True if 'oil_car' in cls else False
    is_stair = True if 'stair' in cls else False
    is_traction = True if 'traction' in cls else False
    plane, is_plane = plane_pose(inputs)
    res = {
        'is_oil_car':is_oilcar, 'oil_car':find_max(inputs, 'oil_car') if is_oilcar else empty_dict,
        'is_stair': is_stair, 'stair': find_max(inputs, 'stair') if is_stair else empty_dict,
        'is_traction': is_traction, 'traction': find_max(inputs, 'traction') if is_traction else empty_dict,
        'is_plane': is_plane, 'plane': plane,
    }
    return res


def mul_process(inputs):
    # 针对bus，，person和queue之间
    cls = set()
    person_list = []
    queue_list = []
    bus_list = []
    person_ids = []
    # count = 0
    for inp in inputs:
        cls.add(inp['class'])
        if inp['class'] == 'person':
            person_list.append(inp)
        if inp['class'] == 'queue':
            queue_list.append(inp)
        if inp['class'] == 'bus':
            bus_list.append(inp)
    # print(cls)
    is_bus = True if 'bus' in cls else False
    is_person = True if 'person' in cls else False
    is_queue = True if 'queue' in cls else False

    uncount_set = set()
    for i in range(len(person_list)):
        uncount_set.add(i)
    seed = 0
    if not (not uncount_set):
        seed = uncount_set.pop()
    while not (not uncount_set):
        ids, uncount_set = dfs(seed, person_list, uncount_set)
        if len(ids) >= People_Num:
            width, height = [0,0]
            for id in ids:
                width += person_list[id]['center'][0]
                height += person_list[id]['center'][1]
            center = [width//len(ids), height//len(ids)]
            s = queue_size(ids, person_list)
            dic = {
                'class': 'queue',
                'center':  center,
                'size': s,
                'bbox': [[center[0] - s[0]/2, center[1] - s[1]/2],
                         [center[0] - s[0]/2, center[1] + s[1]/2],
                         [center[0] + s[0]/2, center[1] - s[1]/2],
                         [center[0] + s[0]/2, center[1] + s[1]/2]]
            }
            queue_list.append(dic)
        else:
            for id in ids:
                person_ids.append(id)
        seed = uncount_set.pop()

    is_queue = is_queue and len(queue_list) > 0
    res = {
        'is_bus': is_bus, 'bus': bus_list,
        'is_person': is_person, 'person': person_list, 'person_ids': person_ids,
        'is_queue': is_queue, 'queue': queue_list
    }
    return res


def safe_area(inputs):
    # bus or person in safe_area
    # return: is_safe=0 可能出现风险；is_safe=1 不会出现风险
    cone_list = []
    for inp in inputs:
        if inp['class'] == 'cone':
            cone_list.append(inp)

    # 形成一个cone的区域闭包
    steady_list = single_process(inputs)
    if steady_list['is_plane']:
        cone_set = list()
        plane_center = steady_list['plane']['center']
        ground_center = steady_list['plane']['ground']
        center = [(plane_center[0] + ground_center[0]) // 2, (plane_center[1] + ground_center[1]) // 2]
        # 将检测到的cone和center做对称，计算相应的cone集合
        for cone in cone_list:
            cone_set.append(cone['center'])
        for cone in cone_list:
            vcone_center = [ center[0] - (cone['center'][0]-center[0]), cone['center'][1]]
            vcone = {'center':vcone_center}
            flag = True
            for c in cone_list:
                if cal_distance(vcone, c) <= 50:
                    flag = False
                    break
            if flag:
                cone_set.append(vcone_center)

        # 整理出区域, 线段格式
        line_set = list()
        cone_set_sorted = sorted(cone_set, key=lambda x:x[0])
        length = len(cone_set)
        if length <= 1:
            return {'is_safe':1, 'area':[]}
        else:
            for i in range(length-1):
                line_set.append([cone_set_sorted[i], cone_set_sorted[i+1]])
            line_set.append([[cone_set_sorted[0][0], 0], cone_set_sorted[0]])
            line_set.append([[cone_set_sorted[-1][0], 0], cone_set_sorted[-1]])
            return {'is_safe':0, 'area': line_set}
    else:
        return {'is_safe':1, 'area':[]}

