"""
config.py
work for experiment
"""

### data_augument.py

# normalization
dis_std = 587   # 距离与框尺寸最大值定义
center_std = 1024   # 中心点最大值定义

# 不同阶段的噪声控制normal参数 [avg=0, var]
state_p_dict = [   # 不同阶段的数据增广中二维正态分布参数
        [4, 36],
        [1, 4],
        [9, 64]]
on_off_dict = [   # 懒惰删除中阈值参数
        0.975, 0.99, 0.985]

### pre_process.py
Distance = 200  # 机鼻与飞机中心点距离
alpha = 0.5   # 飞机位置加权
person_Distance = 10   # 人判定为人群的距离阈值
People_Num = 3   # 人判定为人群的人数阈值

### rnn_classify.py
hidden_size = 32
num_layers = 2
learning_rate = 0.02
sigmoid_threshold = 0.5

### similarity.py
move_Distance = 10   # 判定移动的阈值

### train.py

# similarity
MAX_SIZE = 10   # 与sequence_size相同
WINDOWS = 3   # 对比窗口

INP_SIZE = 61
OUT_SIZE = 16
train_percent = 0.7
sequence_size = 10
num_epochs = 1000


EVENT_DICT = {
    '0': '飞机与空中扶梯连接',
    '1': '飞机与加油车连接',
    '2': '飞机与客舱车连接',
    '3': '飞机与牵引车连接',
    '4': '客车运送乘客上飞机',
    '5': '起飞准备',
    '6': '牵引车牵引进跑道',
}

def config2dict():
    return {
        'dis_std': dis_std, 'center_std': center_std,
        'state_p_dict' : state_p_dict, 'on_off_dict': on_off_dict,
        'Distance': Distance, 'alpha': alpha, 'person_Distance': person_Distance, 'People_Num': People_Num,
        'hidden_size': hidden_size,  'num_layers': num_layers, 'learning_rate': learning_rate,
        'sigmoid_threshold': sigmoid_threshold,
        'move_Distance': move_Distance, 'MAX_SIZE': MAX_SIZE, 'WINDOWS': WINDOWS,
        'INP_SIZE': INP_SIZE, 'OUT_SIZE': OUT_SIZE, 'train_percent': train_percent,
        'sequence_size': sequence_size , 'num_epochs': num_epochs
    }