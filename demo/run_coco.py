import os
import os.path as osp
import numpy as np
import cv2
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import __init_path
import models
from core.config import cfg 
from funcs_utils import get_optimizer, load_checkpoint, get_scheduler, count_parameters, stop, lr_check
from models import project_net
from vis import vis_2d_pose, vis_3d_pose
from COCO import COCOHuman
from renderer import Renderer
from vis import vis_2d_keypoints


class Optimizer:
    def __init__(self, sample_num, dataset, load_dir=''):
        self.dataset = dataset
        self.dataset.datalist = self.dataset.datalist[:sample_num]
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
        self.model = eval(f'models.{cfg.MODEL.name}.get_model')(num_joint=self.dataset.joint_num, graph_L=self.dataset.graph_L)
        checkpoint = load_checkpoint(load_dir=load_dir)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.cuda()

        self.print_freq = 10

        self.coco_J_regressor = torch.Tensor(self.dataset.smpl.joint_regressor_coco).cuda()
        self.project_net = project_net.get_model().cuda()
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.project_net.parameters(), lr=0.1)

    def fit(self):
        self.model.eval()
        self.project_net.train()

        result = {'data': []}
        running_loss = 0.0
        loader = tqdm(self.loader)
        for i, (input_joint, target_joint, img_path, trans) in enumerate(loader):
            input_joint, target_joint, trans = input_joint.cuda().float(), target_joint.cuda().float(), trans.cuda().float()
            start_idx = img_path[0].find('val2017')
            img_name = img_path[0][start_idx:]

            # estimate mesh, pose
            pred_mesh, lifted_joint = self.model(input_joint)
            pred_mesh = pred_mesh[:, self.dataset.graph_perm_reverse[:self.dataset.smpl.face.max() + 1], :]
            pred_coco_joint = torch.matmul(self.coco_J_regressor[None, :, :], pred_mesh)
            # vis_3d_save(pred_smpl_joint[0].detach().cpu().numpy(), self.dataset.smpl_skeleton, prefix='pred3d')

            out = {}
            # assume batch=1
            for j in range(0, 1500):
                # projection
                pred_2d_joint = self.project_net(pred_coco_joint.detach())

                loss = self.criterion(pred_2d_joint, target_joint[:, :17, :])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if j == 500:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = 0.05
                if j == 1000:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = 0.001

            out['mesh'] = pred_mesh[0].detach().cpu().numpy().tolist()
            out['cam_param'] = self.project_net.cam_param[0].detach().cpu().numpy().tolist()
            out['trans'] = trans[0].detach().cpu().numpy().tolist()
            out['pred2d_pose'] = target_joint[0].detach().cpu().numpy().tolist()
            out['idx'] = str(i)
            out['img_path'] = img_name
            result['data'].append(out)

            running_loss += loss.detach().item()

            loader.set_description(f'loss: {loss.detach().item():.4f} ')

        return result


def render(result, img_dir, out_path, focal, img_res, dataset):
    renderer = Renderer(focal_length=focal, img_res=img_res, faces=dataset.smpl.face)
    data = result['data']

    for i in range(len(data)):
        d = data[i]
        pred_vertices, pred2d_pose, cam_param, trans, img_path, save_idx = \
            np.array(d['mesh']), np.array(d['pred2d_pose']), np.array(d['cam_param']), np.array(d['trans']), d['img_path'], d['idx']

        camera_translation = np.array([cam_param[1], cam_param[2], 2 * focal / (img_res * cam_param[0] + 1e-9)])
        outfile = osp.join(out_path, save_idx)

        img_path = osp.join(img_dir, img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)[:, :, ::-1]
        img = cv2.warpAffine(img, trans, (img_res, img_res), flags=cv2.INTER_LINEAR)

        img_shape = 255 * renderer(pred_vertices, camera_translation, img / 255.)
        tmpimg = img.copy().astype(np.uint8)
        tmpkps = np.zeros((3, len(pred2d_pose)))
        tmpkps[0, :], tmpkps[1, :], tmpkps[2, :] = pred2d_pose[:, 0], pred2d_pose[:, 1], 1
        img_2dpose = vis_2d_keypoints(tmpimg, tmpkps, dataset.coco_skeleton)

        # Save reconstructions
        cv2.imwrite(outfile + '_shape.png', img_shape[:, :, ::-1])
        cv2.imwrite(outfile + '_2dpose.png', img_2dpose[:, :, ::-1])
        cv2.imwrite(outfile + '_img.png', img[:, :, ::-1])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Render Pose2Mesh output')
    parser.add_argument('--gpu', type=str, default='0', help='assign gpu number')
    parser.add_argument('--res', type=int, default=500, help='assign image resolution for visualization of mesh projection')
    parser.add_argument('--sample_num', type=int, default=10, help='# of samples to visualize')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # fix config
    cfg.DATASET.target_joint_set = 'coco'
    cfg.MODEL.posenet_pretrained = False

    FOCAL_LENGTH = 1500
    IMG_RES = args.res
    img_path = './data/COCO/images'
    output_path = './demo/result'
    model_chk_path = './experiment/exp_07-15_14:50/checkpoint'

    dataset = COCOHuman(IMG_RES)
    optimizer = Optimizer(args.sample_num, dataset, load_dir=model_chk_path)
    result = optimizer.fit()

    render(result, img_path, output_path, FOCAL_LENGTH, IMG_RES, dataset)