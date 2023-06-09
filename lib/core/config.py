import os
import os.path as osp
import shutil

import yaml
from easydict import EasyDict as edict
import datetime


def init_dirs(dir_list):
    for dir in dir_list:
        if os.path.exists(dir) and os.path.isdir(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)


cfg = edict()


""" Directory """
cfg.cur_dir = osp.dirname(os.path.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, '../../')
cfg.data_dir = osp.join(cfg.root_dir, 'data')
cfg.smpl_dir = osp.join(cfg.root_dir, 'smplpytorch')
cfg.mano_dir = osp.join(cfg.root_dir, 'manopth')
KST = datetime.timezone(datetime.timedelta(hours=9))
save_folder = 'exp_' + str(datetime.datetime.now(tz=KST))[5:-16]
save_folder = save_folder.replace(" ", "_")
save_folder_path = 'experiment/{}'.format(save_folder)

cfg.output_dir = osp.join(cfg.root_dir, save_folder_path)
cfg.graph_dir = osp.join(cfg.output_dir, 'graph')
cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
cfg.res_dir = osp.join(cfg.output_dir, 'result')
cfg.checkpoint_dir = osp.join(cfg.output_dir, 'checkpoint')

print("Experiment Data on {}".format(cfg.output_dir))
init_dirs([cfg.output_dir, cfg.graph_dir, cfg.vis_dir, cfg.checkpoint_dir])

""" Dataset """
cfg.DATASET = edict()
cfg.DATASET.train_list = ['Human36M', 'COCO', 'MuCo']
cfg.DATASET.test_list = ['PW3D']
cfg.DATASET.input_joint_set = 'coco'
cfg.DATASET.target_joint_set = 'human36'
cfg.DATASET.workers = 16
cfg.DATASET.use_gt_input = True

""" Model """
cfg.MODEL = edict()
cfg.MODEL.name = 'pose2mesh_net'
cfg.MODEL.input_shape = (384, 288)
cfg.MODEL.normal_loss_weight = 1e-1
cfg.MODEL.edge_loss_weight = 20
cfg.MODEL.joint_loss_weight = 1e-3
cfg.MODEL.posenet_pretrained = False
cfg.MODEL.posenet_path = './experiment/exp_08-20_13:48/checkpoint'


""" Train Detail """
cfg.TRAIN = edict()
cfg.TRAIN.print_freq = 10
cfg.TRAIN.batch_size = 64
cfg.TRAIN.shuffle = True
cfg.TRAIN.begin_epoch = 1
cfg.TRAIN.end_epoch = 40
cfg.TRAIN.edge_loss_start = 15
cfg.TRAIN.scheduler = 'step'
cfg.TRAIN.lr = 1e-3
cfg.TRAIN.lr_step = [30]
cfg.TRAIN.lr_factor = 0.1
cfg.TRAIN.optimizer = 'rmsprop'

""" Augmentation """
cfg.AUG = edict()
cfg.AUG.flip = False
cfg.AUG.rotate_factor = 0  # 30

""" Test Detail """
cfg.TEST = edict()
cfg.TEST.batch_size = 64
cfg.TEST.shuffle = False
cfg.TEST.weight_path = './experiment/pose2mesh_cocoJ_gt_train_human36_coco'
cfg.TEST.vis = False


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in cfg[k]:
            cfg[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.safe_load(f))
        for k, v in exp_config.items():
            if k in cfg:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        cfg[k][0] = (tuple(v))
                    else:
                        cfg[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


