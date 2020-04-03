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
        return F.relu(x)

class Vote_Net(object):
    def __init__(self, net_paras):
        super(Vote_Net, self).__init__()
        self.net_paras = net_paras
        self.n = len(net_paras)
        self.nets = [NNet(net_para=net_para) for net_para in net_paras]

    def train(self, iter_num, save_dir, sample_batch, label_batch):
        # batch.shape = [batch_size, item_size]
        samples = np.array(sample_batch, dtype=np.float32)
        labels = np.array(label_batch, dtype=np.int8)
        x = Variable(torch.Tensor(samples).type(torch.FloatTensor))
        y = Variable(torch.Tensor(labels).type(torch.IntTensor))

        for i in range(self.n):
            optimizer = torch.optim.Adam(self.nets[i].parameters(), lr=0.02)
            loss_func = nn.MSELoss()

            for iter in range(iter_num):
                out = self.nets[i](x)
                loss = loss_func(out, y)
                print('round-{}={}'.format(iter, loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (iter + 1) % 1000 == 0:
                    torch.save(self.nets[i], save_dir + 'net_'+ str(i) +'_' + str(iter + 1) + '.pkl')

    def val(self, index, sample_batch, label_batch, save_dir, latest_iter):
        file_name = 'net_'+str(index)+'_'+str(latest_iter)+'.pkl'
        model = torch.load(os.path.join(save_dir, file_name))
        samples = np.array(sample_batch, dtype=np.float32)
        labels = np.array(label_batch, dtype=np.int8)

        x = Variable(torch.Tensor(samples).type(torch.FloatTensor))
        y = Variable(torch.Tensor(labels).type(torch.IntTensor))

        preds = model(x)
        count = 0
        for i in range(sample_batch.shape[0]):
            if preds[i] == labels[i]:
                count +=1
        return count / sample_batch.shape[0]

    def vote(self, save_dir, latest_iter, sample):
        preds = []
        pos = 0
        # 与class有关
        sample = np.array(sample, dtype=np.float32)
        x = Variable(torch.Tensor(sample).type(torch.FloatTensor))
        for i in range(self.n):
            file_name = 'net_' + str(i) + '_' + str(latest_iter) + '.pkl'
            model = torch.load(os.path.join(save_dir, file_name))
            pred = model(x)
            preds.append(pred)
            pos += 1
        if pos > self.n //2:
            return 1
        return 0





