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
import classify.config as config

class Rnn(nn.Module):
    def __init__(self, inp_size, out_size):
        super(Rnn, self).__init__()
        self.inp_size = inp_size
        self.out_size = out_size
        self.rnn = nn.LSTM(
            input_size=inp_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
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
        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=config.learning_rate)
        # self.loss_func = nn.MultiLabelSoftMarginLoss()
        self.loss_func = nn.BCELoss()
        self.save_dir = save_dir

    def train(self, dataloader, iter_num, logger, num_epoch):
        losses = []
        for iter, batch in enumerate(dataloader):
            for i in range(len(batch['s'])):
                for j in range(len(batch['s'][i])):
                    batch['s'][i][j] = batch['s'][i][j].detach().numpy().tolist()
            for i in range(len(batch['l'])):
                batch['l'][i] = batch['l'][i].detach().numpy().tolist()
            sample = np.transpose(batch['s'], (2, 0, 1))
            label = np.transpose(batch['l'], (1, 0))
            # print(label.shape)
            x = Variable(torch.tensor(sample, dtype=torch.float32)).cuda()
            h_n = None
            prediction, h_n = self.rnn(x, h_n)
            l = torch.tensor(label, dtype=torch.float32).cuda()
            # print(prediction.detach().cpu().numpy().shape)
            prediction = nn.Sigmoid()(prediction)
            loss = self.loss_func(prediction, l)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            logger.write("Epoch {}/Iter {}: Loss={}\n".format(iter_num, iter, loss))
            # sum_loss += loss
            losses.append(loss)
            if iter_num + 1 == num_epoch:
                torch.save(self.rnn, os.path.join(self.save_dir , str(num_epoch) + '.pkl'))

        return losses

    def eval(self, dataloader, path=""):
        rnn = self.rnn if path=="" else torch.load(path)
        rets = []
        for iter, batch in enumerate(dataloader):
            for i in range(len(batch['s'])):
                for j in range(len(batch['s'][i])):
                    batch['s'][i][j] = batch['s'][i][j].detach().numpy().tolist()
            for i in range(len(batch['l'])):
                batch['l'][i] = batch['l'][i].detach().numpy().tolist()
            sample = np.transpose(batch['s'], (2, 0, 1))
            label = np.transpose(batch['l'], (1, 0)).tolist()[0]
            x = Variable(torch.tensor(sample, dtype=torch.float32)).cuda()
            h_n = None
            prediction, h_n = rnn(x, h_n)
            prediction = nn.Sigmoid()(prediction)
            prediction = prediction.detach().cpu().numpy().tolist()[0]
            # print(prediction)
            for i in range(len(prediction)):
                prediction[i] = 1 if prediction[i] >= config.sigmoid_threshold else 0
            ret = []
            for i in range(8):
                # print("p={} | l={}".format(prediction[2*i : 2*i+2], label[2*i : 2*i+2]))
                ret.append(1 if prediction[2*i : 2*i+2]==label[2*i : 2*i+2] else 0)
            rets.append(ret)

        # calculate percentage
        percentages = []
        r = np.array(rets)
        for i in range(8):
            splice = r[:, i]
            percentages.append(np.mean(splice))
        return {'rets': rets, 'percentages': percentages}









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


def rnn_eval(path, res):
    pass


def rnn_demo(sample, save_dir, latest_iter):
    # input should be sequence
    model = torch.load(os.path.join(save_dir, str(latest_iter)+'.pkl'))
    sample = np.array(sample, dtype=np.float32)
    x = Variable(torch.tensor(sample, dtype=torch.float32)).cuda()
    h_n = None
    prediction, _ = model(x, h_n)
    prediction = nn.Sigmoid()(prediction)
    prediction = prediction.detach().cpu().numpy().tolist()[0]
    # print(prediction)
    for i in range(len(prediction)):
        prediction[i] = 1 if prediction[i] >= config.sigmoid_threshold else 0

    # 转为正常值
    rets = []
    for i in range(8):
        rets.append(prediction[i*2]*2 + prediction[i*2+1])
    rets[7] = 2 if rets[7] == 3 else rets[7]
    return rets

