from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import time
import progress.bar as Bar

from lib.Loss import TrainLoss
from utils import decode

class ModelWithLoss(nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        output = self.model(batch['input'])
        loss, loss_stats = self.loss(output, batch)
        return output[-1], loss, loss_stats

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

class Trainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.mode = model
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)

    def set_device(self, gpus, chunk_sizes, device):
        # ignore data parallel
        self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            # if len(self.opt.gpus) > 1:
            #     model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        print('{}/{}'.format(opt.task, opt.exp_id))
        end = time.time()
        # with torch.no_grad():
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            print('{phase}: [{0}][{1}/{2}]'.format(epoch, iter_id, num_iters, phase=phase))
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                print('|{} {:.4f} '.format(l, avg_loss_stats[l].avg))
            if not opt.hide_data_time:
                print('|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time))
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}'.format(opt.task, opt.exp_id))
            else:
                pass

            if opt.debug > 0:
                self.debug(batch, output, iter_id)

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        # ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = decode.ctdet_decode(
            output['hm'], output['wh'], reg=reg, cat_spec_wh=self.opt.cat_spec_with, K=self.opt.K
        )
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = decode.ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1]
        )
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        loss = TrainLoss(opt)
        return loss_states, loss

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)


