from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import os
from torchsummary import summary

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
        # outs = []
        # for step in range(r_out.size(1)):
        #     out = self.linear(r_out[:, step, :]).detach().numpy()
        #     outs.append(out) # shape: batch_size*hidden_size -> batch_size*out_size
        outs = self.linear(r_out[:, -1, :])
        # return torch.stack(outs, dim=self.out_size), h_n # shape: out_size*batch_size
        return outs, h_n # outs = batch_size*out_size

class RNN_Trainer(object):
    def __init__(self, inp_size, out_size, save_dir):
        self.rnn = Rnn(inp_size=inp_size, out_size=out_size)
        self.rnn = self.rnn.cuda()
        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=0.02)
        self.loss_func = nn.MultiLabelSoftMarginLoss()
        self.save_dir = save_dir

    def train(self, dataloader, iter_num):
        losses = []
        for iter, batch in enumerate(dataloader):
            for i in range(len(batch['s'])):
                for j in range(len(batch['s'][i])):
                    batch['s'][i][j] = batch['s'][i][j].detach().numpy().tolist()
            for i in range(len(batch['l'])):
                batch['l'][i] = batch['l'][i].detach().numpy().tolist()
            sample = np.transpose(batch['s'], (2, 0, 1))
            label = np.transpose(batch['l'], (1, 0))
            x = Variable(torch.tensor(sample, dtype=torch.float32)).cuda()
            h_n = None
            prediction, h_n = self.rnn(x, h_n)
            l = torch.tensor(label, dtype=torch.float32).cuda()
            loss = self.loss_func(prediction, l)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("Epoch {}/Iter {}: Loss={}".format(iter_num, iter, loss))
            # sum_loss += loss
            losses.append(loss)
            if iter_num + 1 == 500:
                torch.save(self.rnn, self.save_dir + 'rnn_' + str(iter_num + 1) + '.pkl')

        return losses

    def eval(self, dataloader, path=""):
        rnn = self.rnn if path=="" else torch.load(path)
        for iter, batch in enumerate(dataloader):
            for i in range(len(batch['s'])):
                for j in range(len(batch['s'][i])):
                    batch['s'][i][j] = batch['s'][i][j].detach().numpy().tolist()
            for i in range(len(batch['l'])):
                batch['l'][i] = batch['l'][i].detach().numpy().tolist()
            sample = np.transpose(batch['s'], (2, 0, 1))
            label = np.transpose(batch['l'], (1, 0))
            x = Variable(torch.tensor(sample, dtype=torch.float32)).cuda()
            h_n = None
            prediction, h_n = rnn(x, h_n)
            prediction = prediction.detach().cpu().numpy()
            # compare prediction with label










def rnn_train(inps, labels, save_dir, iter_num=1000, inp_size=10, out_size=2):
    rnn = Rnn(inp_size=inp_size, out_size=out_size)
    rnn = rnn.cuda()

    # inps == np(batch_size, time_inp, feature_size)
    # labels = np(batch_size, out_size)
    # eg: 2个batch:一个batch5个sque:一个sque5个dim
    sample = np.array(inps, dtype=np.float32)
    label = np.array(labels, dtype=np.int8)
    x = Variable(torch.tensor(sample, dtype=torch.float32)).cuda()
    # y = torch.Tensor(label).type(torch.IntTensor)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.02)
    loss_func = nn.MultiLabelSoftMarginLoss()

    h_n = None
    for iter in range(iter_num):
        prediction, h_n = rnn(x, h_n)
        # prediction = torch.transpose(prediction, 0, 1)
        # print(prediction.shape)
        # losses = []
        # for i in range(out_size):
        #     # print('p={} | l={}\n'.format(prediction[i].shape, label[i].shape))
        #     p = torch.tensor(prediction[i], dtype=torch.float32)
        #     l = torch.tensor(label[i], dtype=torch.long)
        #     losses.append(loss_func(p, l))
        # p = torch.tensor(prediction, dtype=torch.float32)
        l = torch.tensor(label, dtype=torch.float32).cuda()
        loss = loss_func(prediction, l)
        # loss = np.mean(losses)
        print('round-'+str(iter) + '=' + str(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iter+1) % iter_num == 0:
            torch.save(rnn, save_dir+'rnn_'+str(iter+1)+'.pkl')
            return save_dir+'rnn_'+str(iter_num)+'.pkl'


def rnn_eval(sample_batch, label_batch, save_dir, inp_size=47, out_size=4):
    model = torch.load(save_dir)
    samples = np.array(sample_batch, dtype=np.float32)
    labels = np.array(label_batch, dtype=np.int8).transpose()
    # labels = labels.transpose()
    x = Variable(torch.tensor(samples, dtype=torch.float32))
    # y = Variable(torch.Tensor(labels).type(torch.IntTensor))

    preds, _ = model(x) # preds = batch_size* out_size
    preds = preds.detach().numpy().transpose()
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
    x = Variable(torch.tensor(sample, dtype=torch.float32))
    preds, _ = model(x)
    return preds.detach().numpy()
