from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch.utils.data
import torch
import time
import random
import json
import classify.config as config
from classify.pre_process import single_process, mul_process, safe_area, cal_distance
from classify.data_augment import generate_dataset, generate_test, merge, data_augument
from classify.rnn_classify import rnn_train, rnn_eval, rnn_demo, RNN_Trainer
from classify.vote_classify import Vote_Net

json_filename = '../../data/sequence.json'

"""
event_dict record classes of event, and every event has a triple state:
0 for begin;1 for end;2 for in_status；3 for no_event
warning status: 0 for safe; 1 for warning; 2 for alert
the input vector has a frame id and eight status key
"""

# EVENT_DICT = {
#     '0': '飞机与空中扶梯连接',
#     '1': '飞机与加油车连接',
#     '2': '飞机与牵引车连接',
#     '3': '客车运送乘客',
#     '4': '乘客上飞机',
#     '5': '飞机与生活用车连接'
# }


class CLogger():
    def __init__(self, exp_id):
        save_dir = os.path.join('models/rnn/', exp_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # save the opt as opt_log_file
        self.log = open(os.path.join(save_dir,'log.txt'), 'w')
        self.start_line = True

    def write(self, text):
        if self.start_line:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            self.log.write('{}: {}'.format(time_str, text))
        else:
            self.log.write(text)
        self.start_line = False
        if '\n' in text:
            self.start_line = True
            self.log.flush()

    def close(self):
        self.log.close()

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, length, sequence_size):
        self.trainset = []
        self.testset = []
        self.sequences = []
        self.labels = []
        self.length = length - sequence_size
        for i in range(self.length):
            sequence_seg = sequences[i:i+sequence_size]
            label_seg = labels[i+sequence_size]
            self.sequences.append(sequence_seg)
            self.labels.append(label_seg)

    def __getitem__(self, item):
        s = self.sequences[item]
        l = self.labels[item]
        return {'s': s, 'l': l}

    def __len__(self):
        return self.length

    def add(self, sequences, labels, length, sequence_size):
        add_length = length - sequence_size
        self.length += add_length
        for i in range(add_length):
            sequence_seg = sequences[i:i + sequence_size]
            label_seg = labels[i + sequence_size]
            self.sequences.append(sequence_seg)
            self.labels.append(label_seg)

    def split(self):
        train_percent = config.train_percent # 7:3划分数据集
        num = range(self.length)
        data_count = int(self.length * train_percent)
        train_list = random.sample(num, data_count)
        for i in num:
            if i in train_list:
                self.trainset.append({'s':self.sequences[i], 'l': self.labels[i]})
            else:
                self.testset.append({'s': self.sequences[i], 'l': self.labels[i]})
        res_dic = {'train': self.trainset, 'train_len': len(self.trainset), 'test': self.testset,
                   'test_len':len(self.testset)}
        with open(json_filename, "w") as f:
            json.dump(res_dic, f)
            print("refresh sequence dataset!")


class SonSequenceDataSet(torch.utils.data.Dataset):
    def __init__(self, length, dataset):
        self.length = length
        self.dataset = dataset

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.dataset[item]


def classify(split, fresh_dataset=False, exp_id=''):
    # config
    net_paras = [[config.INP_SIZE, config.OUT_SIZE, 10, 5, 0, 0.5],
                 [config.INP_SIZE, config.OUT_SIZE, 10, 4, 1, 0.5]]
    # train_data
    if fresh_dataset:
        samples, labels, length = generate_dataset('../../data/VOC2007_1/', 'sequence_1.csv')
        samples2, labels2, length2 = generate_dataset('../../data/VOC2007_2/', 'sequence_2.csv')
        samples3, labels3, length3 = generate_dataset('../../data/VOC2007_3/', 'sequence_3.csv')
        res_1 = data_augument(seq_dir='../../data/VOC2007_1', anno_dir='../../data/VOC2007_1/Annotations',
                     csv_name='sequence_1.csv')
        res_2 = data_augument(seq_dir='../../data/VOC2007_2', anno_dir='../../data/VOC2007_2/Annotations',
                     csv_name='sequence_2.csv')

        sequence_size = config.sequence_size
        dataset = SequenceDataset(samples, labels, length, sequence_size)
        dataset.add(samples2, labels2, length2, sequence_size)
        dataset.add(samples3, labels3, length3, sequence_size)
        dataset.add(res_1['samples'], res_1['labels'], res_1['length'], sequence_size)
        dataset.add(res_2['samples'], res_2['labels'], res_2['length'], sequence_size)
        dataset.split()
    try:
        load_f = open(json_filename)
        res_dict = json.load(load_f)
    except Exception as e:
        print('fresh_datast should be True if json_file does not exist' )

    train_dataset = SonSequenceDataSet(res_dict['train_len'], res_dict['train'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # test_data
    # res = data_augument(seq_dir='../../data/VOC2007_1', anno_dir='../../data/VOC2007_1/Annotations',
    #                     csv_name='sequence_1.csv')
    # test_dataset = SequenceDataset(res['samples'], res['labels'], res['length'], sequence_size)
    test_dataset = SonSequenceDataSet(res_dict['test_len'], res_dict['test'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    save_dir = os.path.join("models/rnn", exp_id)
    trainer = RNN_Trainer(config.INP_SIZE, config.OUT_SIZE, save_dir)
    logger = CLogger(exp_id)
    logger.write(config.config2dict())
    logger.write('\n')
    # train rnn_net
    num_epochs = config.num_epochs
    if split == 'train':
        start_time = time.time()
        for epoch in range(num_epochs):
            loss = trainer.train(train_loader, epoch, logger, num_epochs)
            print("Epoch: {} end".format(epoch))
        logger.write("total time use:{}\n".format(time.time()-start_time))

    res = trainer.eval(test_loader, path=os.path.join(save_dir, str(num_epochs)+".pkl"))
    logger.write(res['percentages'])
    print(res['percentages'])
    print("pres={}".format(res['precisions']))
    print("recall={}".format(res['recalls']))
    # rnn_pkl = rnn_train(sequence_dic['samples_seg'], sequence_dic['labels_seg'], 'models/rnn/',
    #           1000, INP_SIZE, OUT_SIZE)
    # eval rnn_net
    # results = rnn_eval(sequence_dic_test['samples_seg'], sequence_dic_test['labels_seg'],
    #                    rnn_pkl, INP_SIZE, OUT_SIZE)
    # print(results)

    # train vote_net
    # vn = Vote_Net(net_paras=net_paras)
    # vn.train(sequence_dic['samples_seg';p'], sequence_dic['labels_seg'], 'classify/models/vote/',
    #          5000, INP_SIZE, OUT_SIZE)
    # eval vate_net
    logger.close()


if __name__ == "__main__":
    # classify('test', fresh_dataset=False ,exp_id='2000_64_2_0.02_5')
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    classify('train', fresh_dataset=False, exp_id='2000_64_3_0.02_5')


