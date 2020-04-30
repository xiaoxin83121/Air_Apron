from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch.utils.data
import torch
from classify.pre_process import single_process, mul_process, safe_area, cal_distance
from classify.data_augment import generate_dataset, generate_test, merge, data_augument
from classify.rnn_classify import rnn_train, rnn_eval, rnn_demo, RNN_Trainer
from classify.vote_classify import Vote_Net
MAX_SIZE = 10
WINDOWS = 3
INP_SIZE = 55
OUT_SIZE = 3

"""
event_dict record classes of event, and every event has a triple state:0 for begin;1 for end;2 for in_status
event_warning: 0:events ; 1:warning; 2:preparing
type: 0:EVENT_DICT;  1:0-警告，1-提醒，2-安全; 2:0-起飞准备，1-降落准备
state: 0: begin; 1: in; 2: end
"""

EVENT_DICT = {
    '0': '飞机与空中扶梯连接',
    '1': '飞机与加油车连接',
    '2': '飞机与牵引车连接',
    '3': '客车运送乘客',
    '4': '乘客上飞机',
    '5': '飞机与生活用车连接'
}

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, length, sequence_size):
        self.sequences = []
        self.labels = []
        self.length = length - sequence_size
        for i in range(self.length):
            sequence_seg = sequences[i:i+sequence_size]
            label_seg = labels[i+sequence_size]
            self.sequences.append(sequence_seg)
            self.labels.append(label_seg)
        # print(self.sequences)

    def __getitem__(self, item):
        s = self.sequences[item]
        l = self.labels[item]
        return {'s': s, 'l': l}

    def __len__(self):
        return self.length

def classify_train(dir):
    net_paras = [[INP_SIZE, OUT_SIZE, 10, 5, 0, 0.5], [INP_SIZE, OUT_SIZE, 10, 4, 1, 0.5]]
    samples, labels, length = generate_dataset(dir)
    # print(samples)
    sequence_size = 10
    train_dataset = SequenceDataset(samples, labels, length, sequence_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)

    # sequence_dic = {'samples_seg':[], 'labels_seg':[]}
    # for i in range(length - sequence_size):
    #     samples_seg = samples[i:i+sequence_size]
    #     labels_seg = labels[i+sequence_size]
    #     sequence_dic['samples_seg'].append(samples_seg)
    #     sequence_dic['labels_seg'].append(labels_seg)

    res = data_augument(seq_dir='../../data/VOC2007_new', anno_dir='../../data/VOC2007_new/Annotations',
                        csv_name='sequence1.csv')
    # test_dataset = SequenceDataset(res['samples'], res['labels'], res['length'], sequence_size)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # sequence_dic_test = {'samples_seg': [], 'labels_seg': []}
    # for i in range(res['length'] - sequence_size):
    #     samples_seg_test = res['samples'][i:i + sequence_size]
    #     labels_seg_test = res['labels'][i + sequence_size]
    #     sequence_dic_test['samples_seg'].append(samples_seg_test)
    #     sequence_dic_test['labels_seg'].append(labels_seg_test)

    # train rnn_net
    trainer = RNN_Trainer(INP_SIZE, OUT_SIZE, 'models/rnn/')
    num_epochs = 500
    for epoch in range(num_epochs):
        loss = trainer.train(train_loader, epoch)
        print("Epoch {}: Loss={:4f}".format(epoch, loss))

    # rnn_pkl = rnn_train(sequence_dic['samples_seg'], sequence_dic['labels_seg'], 'models/rnn/',
    #           1000, INP_SIZE, OUT_SIZE)
    # eval rnn_net
    # results = rnn_eval(sequence_dic_test['samples_seg'], sequence_dic_test['labels_seg'],
    #                    rnn_pkl, INP_SIZE, OUT_SIZE)
    # print(results)



    # train vote_net
    vn = Vote_Net(net_paras=net_paras)
    # vn.train(sequence_dic['samples_seg'], sequence_dic['labels_seg'], 'classify/models/vote/',
    #          5000, INP_SIZE, OUT_SIZE)
    # eval vate_net

if __name__ == "__main__":
    # test_inps = generate_test('C:/Users/13778/workshop/gitrepos/Air_Apron/data/VOC2007/Annotations/', 100)
    # sgl = single_process(test_inps)
    # print(sgl)
    # mul = mul_process(test_inps)
    # print(mul)
    classify_train('../../data/VOC2007_new/')

