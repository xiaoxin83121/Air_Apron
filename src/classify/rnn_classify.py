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
        self.rnn = nn.LSTM(
            input_size=inp_size,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )

        self.linear = nn.Linear(32, out_size)

    def forward(self, x, h_n):
        r_out, h_n = self.rnn(x, h_n)
        # r_out.shape = batch_size*time_inp*hidden_size
        outs = []
        for step in range(r_out.size(1)):
            outs.append(self.linear(r_out[:, step, :])) # shape: batch_size*hidden_size -> batch_size*out_size

        return torch.stack(outs, dim=self.out_size), h_n # shape: out_size*batch_size

def rnn_train(inps, labels, save_dir, iter_num=1000, inp_size=10, out_size=2):
    rnn = Rnn(inp_size=inp_size, out_size=out_size)

    # inps和labels == np(batch_size, time_inp, feature_size)
    # eg: 2个batch:一个batch5个sque:一个sque5个dim
    sample = np.array(inps, dtype=np.float32)
    label = np.array(labels, dtype=np.int8)
    label = label.transpose()

    x = Variable(torch.Tensor(sample).type(torch.FloatTensor))
    # y = Variable(torch.Tensor(label).type(torch.IntTensor))

    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.02)
    # loss_func = nn.MultiLabelSoftMarginLoss()
    # loss_func = nn.MultiLabelMarginLoss()
    loss_func = nn.CrossEntropyLoss()

    h_n = None
    for iter in range(iter_num):
        prediction, h_n = rnn(x, h_n)
        h_n = h_n.data

        losses = []
        for i in range(out_size):
            losses.append(loss_func(prediction[i], label[i]))
        loss = np.mean(losses)
        print('round-'+str(iter) + '=' + str(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iter+1)%1000 == 0:
            torch.save(rnn, save_dir+'rnn_'+str(iter+1)+'.pkl')


def rnn_eval(sample_batch, label_batch, save_dir, latest_iter, inp_size=47, out_size=4):
    model = torch.load(save_dir+'rnn_'+str(latest_iter)+'.pkl')
    samples = np.array(sample_batch, dtype=np.float32)
    labels = np.array(label_batch, dtype=np.int8)
    labels = labels.transpose()
    x = Variable(torch.Tensor(samples).type(torch.FloatTensor))
    # y = Variable(torch.Tensor(labels).type(torch.IntTensor))

    preds, _ = model(x)
    results = []
    for i in range(out_size):
        pred = preds[i]
        label = labels[i]
        res = []
        for j in range(len(pred)):
            res.append(1 if pred[i]==label[i] else 0)
        results.append(np.mean(res))
    return results


def rnn_demo(sample, save_dir, latest_iter):
    # input should be sequence
    model = torch.load(save_dir, 'rnn_'+str(latest_iter)+'.pkl')
    sample = np.array(sample, dtype=np.float32)
    x = Variable(torch.Tensor(sample).type(torch.FloatTensor))
    preds, _ = model(x)
    return [pred[-1] for pred in preds]
