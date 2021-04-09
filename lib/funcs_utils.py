import os
import sys
import time
import math
import numpy as np
import cv2
import shutil
from collections import OrderedDict

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from core.config import cfg


def lr_check(optimizer, epoch):
    base_epoch = 5
    if False and epoch <= base_epoch:
        lr_warmup(optimizer, cfg.TRAIN.lr, epoch, base_epoch)

    for param_group in optimizer.param_groups:
        curr_lr = param_group['lr']
    print(f"Current epoch {epoch}, lr: {curr_lr}")


def lr_warmup(optimizer, lr, epoch, base):
    lr = lr * (epoch / base)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        self.acc += time.time() - self.t0  # cacluate time diff

    def reset(self):
        self.acc = 0

    def print(self):
        return round(self.acc, 2)


def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()


def stop():
    sys.exit()


def check_data_pararell(train_weight):
    new_state_dict = OrderedDict()
    for k, v in train_weight.items():
        name = k[7:]  if k.startswith('module') else k  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(model):
    optimizer = None
    if cfg.TRAIN.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.lr,
            momentum=cfg.TRAIN.momentum,
            weight_decay=cfg.TRAIN.weight_decay,
            nesterov=cfg.TRAIN.nesterov
        )
    elif cfg.TRAIN.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=cfg.TRAIN.lr
        )
    elif cfg.TRAIN.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.lr
        )

    return optimizer


def get_scheduler(optimizer):
    scheduler = None
    if cfg.TRAIN.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.lr_step, gamma=cfg.TRAIN.lr_factor)
    elif cfg.TRAIN.scheduler == 'platue':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.TRAIN.lr_factor, patience=10, min_lr=1e-5)

    return scheduler


def save_checkpoint(states, epoch, is_best=None):
    file_name = f'checkpoint{epoch}.pth.tar'
    output_dir = cfg.checkpoint_dir
    if states['epoch'] == cfg.TRAIN.end_epoch:
        file_name = 'final.pth.tar'
    torch.save(states, os.path.join(output_dir, file_name))

    if is_best:
        torch.save(states, os.path.join(output_dir, 'best.pth.tar'))


def load_checkpoint(load_dir, epoch=0, pick_best=False):
    try:
        print(f"Fetch model weight from {load_dir}")
        checkpoint = torch.load(load_dir, map_location='cuda')
        return checkpoint
    except Exception as e:
        raise ValueError("No checkpoint exists!\n", e)


def save_plot(data_list, epoch, title='Train Loss'):
    f = plt.figure()

    plot_title = '{} epoch {}'.format(title, epoch)
    file_ext = '.pdf'
    save_path = '_'.join(title.split(' ')).lower() + file_ext

    plt.plot(np.arange(1, len(data_list) + 1), data_list, 'b-', label=plot_title)
    plt.legend()
    plt.title(plot_title)
    plt.xlabel('epoch')
    plt.xlim(left=0, right=len(data_list) + 1)
    plt.xticks(np.arange(0, len(data_list) + 1, 1.0), fontsize=5)

    min_value = np.asarray(data_list).min()
    plt.annotate('%0.2f' % min_value, xy=(1, min_value), xytext=(8, 0),
                 arrowprops=dict(arrowstyle="simple", connectionstyle="angle3"),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')

    f.savefig(os.path.join(cfg.graph_dir, save_path))
    plt.close(f)
