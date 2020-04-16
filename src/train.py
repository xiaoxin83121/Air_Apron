from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import torch
import torch.utils.data
from torchsummary import summary
from lib.opts import opts
from lib.dataset.Pascal import PascalVOC
from tools.logger import Logger
from lib.models import create_model, load_model, save_model
from lib.trainer import Trainer


"""
TODO:
verify
"""
# sys.path.append("C:/User/13778/workshop/gitrepos/Air_Apron/src/")
sys.path.append("/gs/home/tongchao/zc/Air_Apron/src/")


def train(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = PascalVOC
    opt =opts().update_dataset_info_and_set_heads(opt, Dataset)
    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >=0 else 'cpu')

    print('Creating Models......')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step
        )

    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting Up DataSet......')

    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'), batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    if opt.test:
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.save_dir)
    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'), batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True
    )

    # for iter, batch in enumerate(train_loader):
    #     if iter == 1:
    #         print(batch['hm'].shape)
    #         break

    print("Start Training......")
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            # logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                # logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        # decrease the learn_rate per para:lr_step steps
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()


if __name__ == "__main__":
    opt = opts().parse()
    train(opt)
