import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch_optimizer import Adafactor
import glob
import os 
import time
from datetime import datetime
import logging

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    '''
    setup logger
    '''

    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H-%M-%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, '{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def get_optimizer(named_params, opt):
    logger = logging.getLogger('base')

    trainable_params = []
    for k, v in named_params:
        if v.requires_grad:
            trainable_params.append(v)
        else:
            logger.warning(f'{k} will not be optimized')

    opt_name = opt['type']
    if opt_name == 'adamw':
        optimizer_function = optim.AdamW
    if opt_name == 'adam':
        optimizer_function = optim.Adam
    elif opt_name == 'adafactor':
        optimizer_function = Adafactor
    else:
        raise ValueError(f'{opt_name} optimizer not supported')
    
    return optimizer_function(trainable_params, **opt['opt_params'])


def get_scheduler(optimizer, opt):
    sched_type = opt['type']

    if not sched_type:
        return 

    if sched_type == 'step':
        scheduler = lrs.StepLR
    if sched_type == 'cyclic':
        scheduler = lrs.CyclicLR
    else:
        raise ValueError(f'{sched_type} scheduler type not supported')
    
    return scheduler(optimizer, **opt['sched_params'])


# https://github.com/XPixelGroup/BasicSR
class AvgTimer:
    def __init__(self, window=200):
        self.window = window
        self.current_time = 0
        self.total_time = 0
        self.count = 0
        self.avg_time = 0
        self.start()

    def start(self):
        self.start_time = self.tic = time.time()

    def record(self):
        self.count += 1
        self.toc = time.time()
        self.current_time = self.toc - self.tic
        self.total_time += self.current_time
        self.avg_time = self.total_time / self.count

        # reset
        if self.count > self.window:
            self.count = 0
            self.total_time = 0

        self.tic = time.time()

    def get_current_time(self):
        return self.current_time

    def get_avg_time(self):
        return self.avg_time

