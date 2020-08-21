import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
import transforms3d
import scipy.sparse
from pycocotools.coco import COCO

from funcs_utils import stop, save_obj
from graph_utils import build_coarse_graphs

from core.config import cfg 
from smpl import SMPL
from coord_utils import cam2pixel, process_bbox, get_bbox, rigid_align
from aug_utils import augm_params, j2d_processing, j3d_processing

from vis import vis_2d_pose, vis_3d_pose


class SURREAL(torch.utils.data.Dataset):
    def __init__(self, data_split, args):
        dataset_name = 'SURREAL'
        self.data_split = 'train' if data_split == 'train' else 'val'
        self.debug = args.debug
        self.data_path = osp.join(cfg.data_dir, dataset_name, 'data')
        self.img_path = osp.join(cfg.data_dir, dataset_name, 'images', self.data_split)

        self.mesh_model = SMPL()
        self.smpl_vertex_num = 6890
        self.smpl_joint_num = 24
        self.smpl_joints_name = self.mesh_model.joints_name
        self.smpl_flip_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
        self.smpl_skeleton = (
            (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
            (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15))
        self.smpl_root_joint_idx = self.mesh_model.root_joint_idx
        self.joint_regressor_smpl = self.mesh_model.layer['neutral'].th_J_regressor

        self.datalist = self.load_data()
        det_file_path = osp.join(self.data_path, f'hrnet_output_on_{self.data_split}set.json')
        self.datalist_pose2d_det = self.load_pose2d_det(det_file_path)
        print("Check lengths of annotation and detection output: ", len(self.datalist), len(self.datalist_pose2d_det))

        self.input_joint_name = cfg.DATASET.input_joint_set  # 'coco'
        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(self.input_joint_name)

        self.graph_Adj, self.graph_L, self.graph_perm, self.graph_perm_reverse = \
            build_coarse_graphs(self.mesh_model.face, self.joint_num, self.skeleton, self.flip_pairs, levels=9)

    def get_joint_setting(self, joint_category='human36'):
        joint_num = eval(f'self.{joint_category}_joint_num')
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')

        return joint_num, skeleton, flip_pairs

    def get_smpl_coord(self, smpl_param):
        pose, shape, trans, gender = smpl_param['pose'], smpl_param['shape'], smpl_param['trans'], smpl_param['gender']
        smpl_pose = torch.FloatTensor(pose).view(1, -1);
        smpl_shape = torch.FloatTensor(shape).view(1, -1);  # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        smpl_trans = torch.FloatTensor(trans).view(1,-1) # translation vector

        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.mesh_model.layer[gender](smpl_pose, smpl_shape, smpl_trans)
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3);
        smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1, 3)

        # incorporate face keypoints
        # smpl_face_kps_coord = smpl_mesh_coord[self.face_kps_vertex, :].reshape(-1, 3)
        # smpl_joint_coord = np.concatenate((smpl_joint_coord, smpl_face_kps_coord))

        # m -> mm
        smpl_mesh_coord, smpl_joint_coord = smpl_mesh_coord * 1000, smpl_joint_coord * 1000

        return smpl_mesh_coord, smpl_joint_coord

    def load_pose2d_det(self, det_path):
        datalist = []

        with open(det_path) as f:
            data = json.load(f)
            for item in sorted(data, key=lambda d: d['image_id']):
                datalist.append({
                    'img_id': item['image_id'],
                    'annot_id': item['annotation_id'],
                    'img_joint':np.array(item['keypoints'], dtype=np.float32)
                })

        datalist = sorted(datalist, key=lambda d: d['img_id'])

        if self.data_split == 'train':
            # fix gpu bug
            datalist.append(datalist[0])
        return datalist

    def load_data(self):
        print('Load annotations of SURREAL')

        db = COCO(osp.join(self.data_path, self.data_split + '.json'))

        datalist = []
        for iid in db.imgs.keys():
            img = db.imgs[iid]
            img_id = img['id']
            img_width, img_height = img['width'], img['height']
            img_name = img['file_name']
            cam_param = {k: np.array(v) for k, v in img['cam_param'].items()}

            ann_id = db.getAnnIds(img_id)
            ann = db.loadAnns(ann_id)[0]
            smpl_param = ann['smpl_param']
            joint_cam = np.array(ann['joint_cam'], dtype=np.float32).reshape(-1,3)

            bbox = process_bbox(ann['bbox'])
            if bbox is None: continue

            datalist.append({
                'img_id': img_id,
                'img_name': img_name,
                'img_shape': (img_height, img_width),
                'cam_param': cam_param,
                'bbox': bbox,
                'smpl_param': smpl_param
            })

            if self.debug and len(datalist) > 1000:
                break

        datalist = sorted(datalist, key=lambda d: d['img_id'])
        if self.data_split == 'train':
            # fix gpu bug
            datalist.append(datalist[0])
        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_id, img_name, cam_param, bbox, smpl_param, img_shape =data['img_id'], data['img_name'], data['cam_param'], data['bbox'], data[
            'smpl_param'], data['img_shape']
        flip, rot = augm_params(is_train=(self.data_split == 'train'))

        # smpl coordinates
        smpl_mesh_coord_cam, smpl_joint_coord_cam = self.get_smpl_coord(smpl_param)
        smpl_coord_cam = np.concatenate((smpl_mesh_coord_cam, smpl_joint_coord_cam))
        smpl_coord_img = cam2pixel(smpl_coord_cam, cam_param['focal'], cam_param['princpt'])
        joint_coord_img = smpl_coord_img[self.smpl_vertex_num:][:, :2]

        # vis_2d_pose(joint_coord_img, img_path, self.smpl_skeleton, prefix='gt')

        # root relative cam coord
        smpl_coord_cam = smpl_coord_cam - smpl_coord_cam[self.smpl_vertex_num + self.smpl_root_joint_idx]
        mesh_coord_cam = smpl_coord_cam[:self.smpl_vertex_num];
        joint_coord_cam = smpl_coord_cam[self.smpl_vertex_num:];

        if not cfg.DATASET.use_gt_input:
            # train / test with 2d dection input
            det_data = self.datalist_pose2d_det[idx]
            assert img_id == det_data['img_id']
            joint_coord_img = det_data['img_joint']

        # vis_2d_pose(joint_coord_img, img_path, self.smpl_skeleton, prefix='det')
        # vis_3d_pose(joint_coord_cam, self.smpl_skeleton, joint_set_name='smpl')

        # make new bbox
        bbox = get_bbox(joint_coord_img)
        bbox = process_bbox(bbox.copy())

        # aug
        joint_coord_img, trans = j2d_processing(joint_coord_img.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]),
                                          bbox, rot, flip, self.flip_pairs)
        joint_coord_cam = j3d_processing(joint_coord_cam, rot, flip, self.flip_pairs)

        #  -> 0~1
        joint_coord_img = joint_coord_img[:, :2]
        joint_coord_img /= np.array([[cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]]])

        # normalize loc&scale
        mean, std = np.mean(joint_coord_img, axis=0), np.std(joint_coord_img, axis=0)
        joint_coord_img = (joint_coord_img.copy() - mean) / std

        if cfg.MODEL.name == 'pose2mesh_net':
            # default valid
            mesh_valid = np.ones((len(mesh_coord_cam), 1), dtype=np.float32)
            reg_joint_valid = np.ones((len(joint_coord_cam), 1), dtype=np.float32)
            lift_joint_valid = np.ones((len(joint_coord_cam), 1), dtype=np.float32)

            inputs = {'pose2d': joint_coord_img}
            targets = {'mesh': mesh_coord_cam / 1000, 'lift_pose3d': joint_coord_cam, 'reg_pose3d': joint_coord_cam}
            meta = {'mesh_valid': mesh_valid, 'lift_pose3d_valid': lift_joint_valid, 'reg_pose3d_valid': reg_joint_valid}

        elif cfg.MODEL.name == 'posenet':
            # default valid
            joint_valid = np.ones((len(joint_coord_cam), 1), dtype=np.float32)
            return joint_coord_img, joint_coord_cam, joint_valid

        return inputs, targets, meta

    def compute_joint_err(self, pred_joint, target_joint):
        # root align joint
        pred_joint, target_joint = pred_joint - pred_joint[:, :1, :], target_joint - target_joint[:, :1, :]

        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()

        joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error

    def compute_both_err(self, pred_mesh, target_mesh, pred_joint, target_joint):
        # root joint align
        pred_mesh, target_mesh = pred_mesh - pred_joint[:, :1, :], target_mesh - target_joint[:, :1, :]
        pred_joint, target_joint = pred_joint - pred_joint[:, :1, :], target_joint - target_joint[:, :1, :]

        pred_mesh, target_mesh = pred_mesh.detach().cpu().numpy(), target_mesh.detach().cpu().numpy()
        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()

        mesh_mean_error = np.power((np.power((pred_mesh - target_mesh), 2)).sum(axis=2), 0.5).mean()
        joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error, mesh_mean_error

    def evaluate_joint(self, outs):
        print('Evaluation start...')
        sample_num = len(outs)

        mpjpe = np.zeros((sample_num, self.joint_num))  # pose error
        pampjpe = np.zeros((sample_num, self.joint_num))  # pose error

        for n in range(sample_num):
            out = outs[n]

            # render materials
            pose_coord_out, pose_coord_gt = out['joint_coord'], out['joint_coord_target']

            # root joint alignment
            pose_coord_out, pose_coord_gt = pose_coord_out - pose_coord_out[:1], pose_coord_gt - pose_coord_gt[:1]

            # pose error calculate
            mpjpe[n] = np.sqrt(np.sum((pose_coord_out - pose_coord_gt) ** 2, 1))
            # perform rigid alignment
            pose_coord_out = rigid_align(pose_coord_out, pose_coord_gt)
            pampjpe[n] = np.sqrt(np.sum((pose_coord_out - pose_coord_gt) ** 2, 1))

        # total pose error (H36M joint set)
        tot_err = np.mean(mpjpe)
        eval_summary = 'MPJPE (mm)    >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

        tot_err = np.mean(pampjpe)
        eval_summary = 'PA-MPJPE (mm) >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

    def evaluate(self, outs):
        annots = self.datalist
        assert len(annots) == len(outs)
        sample_num = len(annots)
        eval_result = {'pose_error': [], 'mesh_error': []}
        for n in range(sample_num):
            out = outs[n]
            annot = annots[n]

            # root joint alignment
            mesh_coord_out, mesh_coord_gt = out['mesh_coord'], out['mesh_coord_target']
            joint_coord_out, joint_coord_gt = np.dot(self.joint_regressor_smpl, mesh_coord_out), np.dot(
                self.joint_regressor_smpl, mesh_coord_gt)
            mesh_coord_out = mesh_coord_out - joint_coord_out[:1]
            mesh_coord_target = mesh_coord_gt - joint_coord_gt[:1]
            joint_coord_out = joint_coord_out - joint_coord_out[:1]
            joint_coord_target = joint_coord_gt - joint_coord_gt[:1]

            # mesh error
            eval_result['mesh_error'].append(
                np.sqrt(np.sum((mesh_coord_out - mesh_coord_target) ** 2, 1)))  # meter -> milimeter

            # pose error
            eval_result['pose_error'].append(
                np.sqrt(np.sum((joint_coord_out - joint_coord_target) ** 2, 1)))  # meter -> milimeter

            vis = cfg.TEST.vis
            if vis:
                filename = annot['img_name'].split('/')[-1][:-4]  #str(n)
                save_obj(mesh_coord_out, self.mesh_model.face, osp.join(cfg.vis_dir, filename + '_pred.obj'))
                save_obj(mesh_coord_target, self.mesh_model.face, osp.join(cfg.vis_dir, filename + '_gt.obj'))

        print('Pose error (MPJPE): %.2f mm' % np.mean(eval_result['pose_error']))
        print('Mesh error (MPVPE): %.2f mm' % np.mean(eval_result['mesh_error']))




