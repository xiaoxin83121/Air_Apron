from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import torch


class Logger(object):
    def __init__(self, opt):
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)

        if not os.path.exists(opt.debug_dir):
            os.makedirs(opt.debug_dir)

        time_str = time.strftime('%Y-%m-%d-%H-%M')

        # save the opt as opt_log_file
        args = dict((name, getattr(opt, name)) for name in dir(opt)
                    if not name.startswith('_'))
        file_name = os.path.join(opt.save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> torch version: {}\n'.format(torch.__version__))
            opt_file.write('==> cudnn version: {}\n'.format(torch.backends.cudnn.version()))
            opt_file.write('==> Cmd:\n')
            opt_file.write(str(sys.argv))
            opt_file.write('\n==> Opt:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))

        log_dir = opt.save_dir + '/log_{}'.format(time_str)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not os.path.exists(os.path.dirname(log_dir)):
            os.mkdir(os.path.dirname(log_dir))
        # open the log file
        self.log = open(log_dir+'/log.txt', 'w')
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



