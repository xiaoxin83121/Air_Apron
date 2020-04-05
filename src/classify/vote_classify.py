from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

"""
net_paras:[[inp_size, out_size, hidden_size, n_stacks, is_dropout, drop_P],...]
"""

class NNet(nn.Module):
    def __init__(self, net_para):
        super(NNet, self).__init__()
        self.inp_size = net_para[0]
        self.out_size = net_para[1]
        self.hidden_size = net_para[2]
        self.n_stacks = net_para[3]
        self.is_dropout = net_para[4]
        self.drop_p = net_para[5]
        self.inp_layer = nn.Linear(self.inp_size, self.hidden_size)
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_layer = nn.Linear(self.hidden_size, self.out_size)

    def forward(self, x):
        x = self.inp_layer(x)
        for _ in range(self.n_stacks):
            x = self.hidden_layer(x)
        if self.is_dropout:
            x = nn.Dropout(self.drop_p)
        x = self.out_layer(x)
        return F.relu(x) # shape: batch_size*out_size

class Vote_Net(object):
    def __init__(self, net_paras):
        super(Vote_Net, self).__init__()
        self.net_paras = net_paras
        self.n = len(net_paras)
        self.nets = [NNet(net_para=net_para) for net_para in net_paras]

    def train(self, sample_batch, label_batch, save_dir, iter_num, inp_size=47, out_size=4):
        # batch.shape = [batch_size, item_size]
        samples = np.array(sample_batch, dtype=np.float32)
        labels = np.array(label_batch, dtype=np.int8)
        labels = labels.transpose()
        x = Variable(torch.Tensor(samples).type(torch.FloatTensor))
        # y = Variable(torch.Tensor(labels).type(torch.IntTensor))

        for i in range(self.n):
            optimizer = torch.optim.Adam(self.nets[i].parameters(), lr=0.02)
            loss_func = nn.CrossEntropyLoss()

            for iter in range(iter_num):
                out = self.nets[i](x)
                out = np.array(out).transpose()

                losses = []
                for i in range(out_size):
                    losses.append(loss_func(out[i], labels[i]))
                loss = np.mean(losses)
                print('round-{}={}'.format(iter, loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (iter + 1) % 5000 == 0:
                    torch.save(self.nets[i], save_dir + 'net_'+ str(i) +'_' + str(iter + 1) + '.pkl')

    def val(self, index, sample_batch, label_batch, save_dir, latest_iter, inp_size=47, out_size=4):
        file_name = 'net_'+str(index)+'_'+str(latest_iter)+'.pkl'
        model = torch.load(os.path.join(save_dir, file_name))
        samples = np.array(sample_batch, dtype=np.float32)
        labels = np.array(label_batch, dtype=np.int8)
        labels = labels.transpose()
        x = Variable(torch.Tensor(samples).type(torch.FloatTensor))
        # y = Variable(torch.Tensor(labels).type(torch.IntTensor))

        preds = model(x)
        results = []
        for i in range(out_size):
            pred = preds[i]
            label = labels[i]
            res = []
            for j in range(len(pred)):
                res.append(1 if pred[i] == label[i] else 0)
            results.append(np.mean(res))
        return results

def vote(sample, save_dir):
    preds = []
    pos = 0
    # 与class有关
    sample = np.array(sample, dtype=np.float32)
    x = Variable(torch.Tensor(sample).type(torch.FloatTensor))
    files = os.listdir(save_dir)
    for file in files:
        model = torch.load(os.path.join(save_dir, file))
        pred = model(x)
        preds.append(pred)
    preds = np.array(preds)
    return np.argmax(np.bincount(preds))





