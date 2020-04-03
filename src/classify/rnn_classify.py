from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import os

class Rnn(nn.Module):
    def __init__(self, inp_size, out_size):
        super(Rnn, self).__init__()
        self.inp_size = inp_size
        self.out_size = out_size
        self.rnn = nn.RNN(
            input_size=inp_size,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )

        self.linear = nn.Linear(32, out_size)

    def forward(self, x, h_n):
        r_out, h_n = self.rnn(x, h_n)

        outs = []
        for step in range(r_out.size(1)):
            outs.append(self.linear(r_out[:, step, :]))

        return torch.stack(outs, dim=self.out_size), h_n

def rnn_train(inps, labels, save_dir, iter_num=1000, inp_size=10, out_size=2):
    rnn = Rnn(inp_size=inp_size, out_size=out_size)

    # inps和labels == np(batch_size, time_inp, feature_size)
    # eg: 2个batch:一个batch5个sque:一个sque5个dim
    sample = np.array(inps, dtype=np.float32)
    label = np.array(labels, dtype=np.int8)

    x = Variable(torch.Tensor(sample).type(torch.FloatTensor))
    y = Variable(torch.Tensor(label).type(torch.IntTensor))

    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.02)
    loss_func = nn.MSELoss()

    h_n = None
    for iter in range(iter_num):
        prediction, h_n = rnn(x, h_n)
        h_n = h_n.data

        loss = loss_func(prediction, y)
        print('round-'+str(iter) + '=' + str(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iter+1)%1000 == 0:
            torch.save(rnn, save_dir+'rnn_'+str(iter+1)+'.pkl')


def rnn_eval(sample_batch, label_batch, save_dir, latest_iter):
    model = torch.load(save_dir+'rnn_'+str(latest_iter)+'.pkl')
    samples = np.array(sample_batch, dtype=np.float32)
    labels = np.array(label_batch, dtype=np.int8)
    x = Variable(torch.Tensor(samples).type(torch.FloatTensor))
    y = Variable(torch.Tensor(labels).type(torch.IntTensor))

    preds = model(x)
    count = 0
    for i in range(sample_batch.shape[0]):
        if preds[i] == labels[i]:
            count += 1
    return count / sample_batch.shape[0]


def rnn_demo(save_dir, latest_iter, sample):
    model = torch.load(save_dir, 'rnn_'+str(latest_iter)+'.pkl')
    sample = np.array(sample, dtype=np.float32)
    x = Variable(torch.Tensor(sample).type(torch.FloatTensor))
    return model(x)
