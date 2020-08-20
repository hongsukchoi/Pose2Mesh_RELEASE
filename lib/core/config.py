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


config = edict()


""" Directory """
config.cur_dir = osp.dirname(os.path.abspath(__file__))
config.root_dir = osp.join(config.cur_dir, '../../')
config.data_dir = osp.join(config.root_dir, 'data')
config.smpl_dir = osp.join(config.root_dir, 'smplpytorch')
config.mano_dir = osp.join(config.root_dir, 'manopth')
KST = datetime.timezone(datetime.timedelta(hours=9))
save_folder = 'exp_' + str(datetime.datetime.now(tz=KST))[5:-16]
save_folder = save_folder.replace(" ", "_")
save_folder_path = 'experiment/{}'.format(save_folder)

config.output_dir = osp.join(config.root_dir, save_folder_path)
config.graph_dir = osp.join(config.output_dir, 'graph')
config.vis_dir = osp.join(config.output_dir, 'vis')
config.res_dir = osp.join(config.output_dir, 'result')
config.checkpoint_dir = osp.join(config.output_dir, 'checkpoint')

print("Experiment Data on {}".format(config.output_dir))
init_dirs([config.output_dir, config.graph_dir, config.vis_dir, config.checkpoint_dir])

""" Dataset """
config.DATASET = edict()
config.DATASET.train_list = ['Human36M', 'COCO', 'MuCo']
config.DATASET.test_list = ['PW3D']
config.DATASET.input_joint_set = 'coco'
config.DATASET.target_joint_set = 'human36'
config.DATASET.workers = 16
config.DATASET.use_gt_input = True

""" Model """
config.MODEL = edict()
config.MODEL.name = 'pose2mesh_net'
config.MODEL.input_shape = (384, 288)
config.MODEL.normal_loss_weight = 1e-1
config.MODEL.edge_loss_weight = 20
config.MODEL.joint_loss_weight = 1e-3
config.MODEL.posenet_pretrained = False
config.MODEL.posenet_path = './experiment/exp_08-20_13:48/checkpoint'


""" Train Detail """
config.TRAIN = edict()
config.TRAIN.print_freq = 10
config.TRAIN.batch_size = 64
config.TRAIN.shuffle = True
config.TRAIN.begin_epoch = 1
config.TRAIN.end_epoch = 40
config.TRAIN.edge_loss_start = 15
config.TRAIN.scheduler = 'step'
config.TRAIN.lr = 1e-3
config.TRAIN.lr_step = [30]
config.TRAIN.lr_factor = 0.1
config.TRAIN.optimizer = 'rmsprop'

""" Augmentation """
config.AUG = edict()
config.AUG.flip = False
config.AUG.rotate_factor = 0  # 30

""" Test Detail """
config.TEST = edict()
config.TEST.batch_size = 64
config.TEST.shuffle = False
config.TEST.weight_path = './experiment/pose2mesh_cocoJ_gt_train_human36_coco'
config.TEST.vis = False


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


