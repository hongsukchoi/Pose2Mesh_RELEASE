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
from core.config import cfg 
from _mano import MANO

from graph_utils import build_coarse_graphs

from coord_utils import process_bbox, get_bbox
from aug_utils import j2d_processing
from funcs_utils import save_obj, stop
from vis import vis_3d_pose, vis_2d_pose


class FreiHAND(torch.utils.data.Dataset):
    def __init__(self, mode, args):
        dataset_name = 'FreiHAND'
        self.data_split = mode
        self.data_path = osp.join(cfg.data_dir, dataset_name, 'data')

        # MANO joint set
        self.mesh_model = MANO()
        self.face = self.mesh_model.face
        self.joint_regressor_mano = self.mesh_model.joint_regressor
        self.vertex_num = self.mesh_model.vertex_num
        self.joint_num = self.mesh_model.joint_num
        self.joints_name = self.mesh_model.joints_name
        self.skeleton = self.mesh_model.skeleton
        self.root_joint_idx = self.mesh_model.root_joint_idx
        self.joint_hori_conn = ((1,5), (5,9), (9,13), (13,17), (2,6),(6,10),(10,14), (14,18), (3,7), (7,11), (11,15), (15,19),(4,8),(8,12),(12,16),(16,20))
            # ((1,5,9,13,17),(2,6,10,14,18),(3,7,11,15,19),(4,8,12,16,20))

        self.datalist = self.load_data()
        det_path = osp.join(self.data_path, f'hrnet_output_on_{mode}set.json')
        self.datalist_pose2d_det = self.load_pose2d_det(det_path)
        print("Check lengths of annotation and detection output: ", len(self.datalist), len(self.datalist_pose2d_det))

        self.graph_Adj, self.graph_L, self.graph_perm, self.graph_perm_reverse = \
            build_coarse_graphs(self.mesh_model.face, self.joint_num, self.skeleton, self.joint_hori_conn, levels=6)

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

        return sorted(datalist, key=lambda d: d['img_id'])

    def load_data(self):
        print('Load annotations of FreiHAND ')
        if self.data_split == 'train':
            db = COCO(osp.join(self.data_path, 'freihand_train_coco.json'))
            with open(osp.join(self.data_path, 'freihand_train_data.json')) as f:
                data = json.load(f)

        else:
            db = COCO(osp.join(self.data_path, 'freihand_eval_coco.json'))
            with open(osp.join(self.data_path, 'freihand_eval_data.json')) as f:
                data = json.load(f)

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.data_path, img['file_name'])
            img_shape = (img['height'], img['width'])
            db_idx = str(img['db_idx'])

            if self.data_split == 'train':
                cam_param, mano_param, joint_cam = data[db_idx]['cam_param'], data[db_idx]['mano_param'], data[db_idx][
                    'joint_3d']
                joint_cam = np.array(joint_cam).reshape(-1, 3)
                bbox = process_bbox(np.array(ann['bbox']))
                if bbox is None: continue

            else:
                cam_param, scale = data[db_idx]['cam_param'], data[db_idx]['scale']
                cam_param['R'] = np.eye(3).astype(np.float32).tolist();
                cam_param['t'] = np.zeros((3), dtype=np.float32)  # dummy
                joint_cam = np.ones((self.joint_num, 3), dtype=np.float32)  # dummy
                mano_param = {'pose': np.ones((48), dtype=np.float32), 'shape': np.ones((10), dtype=np.float32)}

            datalist.append({
                'img_id': image_id,
                'img_path': img_path,
                'img_shape': img_shape,
                'joint_cam': joint_cam,
                'cam_param': cam_param,
                'mano_param': mano_param})

        return sorted(datalist, key=lambda d: d['img_id'])

    def get_mano_coord(self, mano_param, cam_param):
        pose, shape = mano_param['pose'], mano_param['shape']
        # mano parameters (pose: 48 dimension, shape: 10 dimension)
        mano_pose = torch.FloatTensor(pose).view(-1, 3)
        mano_shape = torch.FloatTensor(shape).view(1, -1)
        # camera rotation and translation
        R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3), np.array(cam_param['t'], dtype=np.float32).reshape(3)
        mano_trans = torch.from_numpy(t).view(-1, 3)

        # transform world coordinate to camera coordinate
        root_pose = mano_pose[self.root_joint_idx, :].numpy()
        angle = np.linalg.norm(root_pose)
        root_pose = transforms3d.axangles.axangle2mat(root_pose / angle, angle)
        root_pose = np.dot(R, root_pose)
        axis, angle = transforms3d.axangles.mat2axangle(root_pose)
        root_pose = axis * angle
        mano_pose[self.root_joint_idx] = torch.from_numpy(root_pose)
        mano_pose = mano_pose.view(1, -1)

        # get mesh and joint coordinates
        mano_mesh_coord, mano_joint_coord = self.mesh_model.layer(mano_pose, mano_shape, mano_trans)
        mano_mesh_coord = mano_mesh_coord.numpy().reshape(self.vertex_num, 3);
        mano_joint_coord = mano_joint_coord.numpy().reshape(self.joint_num, 3)

        return mano_mesh_coord, mano_joint_coord

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_id, img_path, img_shape, joint_cam, cam_param, mano_param = \
            data['img_id'], data['img_path'], data['img_shape'],  data['joint_cam'], data['cam_param'], data['mano_param']
        rot, flip = 0, 0

        # mano coordinates
        mano_mesh_cam, mano_joint_cam = self.get_mano_coord(mano_param, cam_param)
        mano_coord_cam = np.concatenate((mano_mesh_cam, mano_joint_cam))
        # cam -> image projection
        # focal, princpt = cam_param['focal'], cam_param['princpt']
        # joint_coord_img = cam2pixel(mano_joint_cam, focal, princpt)[:, :2]

        # root align cam mesh/joint
        mano_coord_cam = mano_coord_cam - mano_joint_cam[:1]
        mesh_coord_cam = mano_coord_cam[:self.vertex_num];
        joint_coord_cam = mano_coord_cam[self.vertex_num:];

        # use det
        det_data = self.datalist_pose2d_det[idx]
        assert img_id == det_data['img_id']
        joint_coord_img = det_data['img_joint']

        # make bbox
        bbox = get_bbox(joint_coord_img)
        bbox = process_bbox(bbox.copy())

        # aug
        joint_coord_img, trans = j2d_processing(joint_coord_img.copy(),
                                                (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]),
                                                bbox, rot, flip, None)
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
            return inputs, targets, meta

        elif cfg.MODEL.name == 'posenet':
            # default valid
            joint_valid = np.ones((len(joint_coord_cam), 1), dtype=np.float32)
            return joint_coord_img, joint_coord_cam, joint_valid

    def compute_joint_err(self, pred_joint, target_joint):
        return 0

    def compute_both_err(self, pred_mesh, target_mesh, pred_joint, target_joint):
        return 0, 0

    def evaluate_joint(self, outs):
        print('Evaluation start...')
        annots = self.datalist
        assert len(annots) == len(outs)
        sample_num = len(annots)

        mesh_output_save = []
        joint_output_save = []
        for n in range(sample_num):
            annot = annots[n]
            out = outs[n]

            joint_coord_out = out['joint_coord']
            mesh_coord_out = joint_coord_out  # dummy

            mesh_coord_out = mesh_coord_out - joint_coord_out[:1]
            joint_coord_out = joint_coord_out - joint_coord_out[:1]

            mesh_output_save.append(mesh_coord_out.tolist())
            joint_output_save.append(joint_coord_out.tolist())

            vis = False
            if vis:
                filename = str(n)
                save_obj(mesh_coord_out, self.mano.face, osp.join(cfg.output_dir, filename + '.obj'))

        output_save_path = osp.join(cfg.output_dir, 'pred.json')
        with open(output_save_path, 'w') as f:
            json.dump([joint_output_save, mesh_output_save], f)

    def evaluate(self, outs):
        print('Evaluation start...')
        annots = self.datalist
        assert len(annots) == len(outs)
        sample_num = len(annots)

        mesh_output_save = []
        joint_output_save = []
        for n in range(sample_num):
            annot = annots[n]
            out = outs[n]

            mesh_coord_out = out['mesh_coord']
            # joint_coord_out = coord_out_cam[self.vertex_num:,:]
            joint_coord_out = np.dot(self.joint_regressor_mano, mesh_coord_out)

            mesh_output_save.append(mesh_coord_out.tolist())
            joint_output_save.append(joint_coord_out.tolist())

            vis = cfg.TEST.vis
            if vis and n % 500 == 0:
                filename = str(n)
                save_obj(mesh_coord_out, self.mesh_model.face, osp.join(cfg.vis_dir, filename + '.obj'))

        output_save_path = osp.join(cfg.output_dir, 'pred.json')
        with open(output_save_path, 'w') as f:
            json.dump([joint_output_save, mesh_output_save], f)

