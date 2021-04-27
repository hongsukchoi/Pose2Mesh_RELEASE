import os
import os.path as osp
import numpy as np
import copy
import json
import cv2
import random
import math
import torch
import scipy.sparse
from pycocotools.coco import COCO

from core.config import cfg
from Human36M.noise_stats import error_distribution
from noise_utils import synthesize_pose
from smpl import SMPL
from coord_utils import process_bbox, get_bbox
from aug_utils import augm_params, j2d_processing, affine_transform, j3d_processing, flip_2d_joint
from vis import vis_3d_pose, vis_2d_pose


class MSCOCO(torch.utils.data.Dataset):
    def __init__(self, data_split, args):
        dataset_name = 'COCO'
        self.data_split = 'train'
        self.img_path = osp.join(cfg.data_dir, dataset_name, 'images')
        self.annot_path = osp.join(cfg.data_dir, dataset_name, 'annotations')
        self.fitting_thr = 3.0  # following I2L-MeshNet

        # SMPL joint set
        self.mesh_model = SMPL()
        self.smpl_root_joint_idx = self.mesh_model.root_joint_idx
        self.face_kps_vertex = self.mesh_model.face_kps_vertex
        self.smpl_vertex_num = 6890
        self.smpl_joint_num = 24
        self.smpl_flip_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
        self.smpl_skeleton = (
            (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
            (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15))

        # h36m skeleton
        self.human36_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.human36_joint_num = 17
        self.human36_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
                'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.human36_skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        self.human36_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.human36_root_joint_idx = self.human36_joints_name.index('Pelvis')
        self.human36_error_distribution = self.get_stat()
        self.joint_regressor_h36m = self.mesh_model.joint_regressor_h36m

        # COCO joint set
        self.coco_joint_num = 19  # 17 + 2, manually added pelvis and neck
        self.coco_joints_name = (
            'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
            'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
        self.coco_flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        self.coco_origin_skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15), (5, 6), (11, 12))
        self.coco_skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15),  # (5, 6), #(11, 12),
            (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 0))
        self.coco_root_joint_idx = self.coco_joints_name.index('Pelvis')
        self.joint_regressor_coco = self.mesh_model.joint_regressor_coco

        self.input_joint_name = cfg.DATASET.input_joint_set
        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(self.input_joint_name)
        self.datalist = self.load_data()

    def get_stat(self):
        ordered_stats = []
        for joint in self.human36_joints_name:
            item = list(filter(lambda stat: stat['Joint'] == joint, error_distribution))[0]
            ordered_stats.append(item)

        print("error stat joint num: ", len(ordered_stats))
        return ordered_stats

    def generate_syn_error(self):
        noise = np.zeros((self.human36_joint_num, 2), dtype=np.float32)
        weight = np.zeros(self.human36_joint_num, dtype=np.float32)
        for i, ed in enumerate(self.human36_error_distribution):
            noise[i, 0] = np.random.normal(loc=ed['mean'][0], scale=ed['std'][0])
            noise[i, 1] = np.random.normal(loc=ed['mean'][1], scale=ed['std'][1])
            weight[i] = ed['weight']

        prob = np.random.uniform(low=0.0, high=1.0, size=self.human36_joint_num)
        weight = (weight > prob)
        noise = noise * weight[:, None]

        return noise

    def get_joint_setting(self, joint_category='human36'):
        joint_num = eval(f'self.{joint_category}_joint_num')
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')

        return joint_num, skeleton, flip_pairs

    def load_data(self):
        print('Load annotations of COCO')
        db = COCO(osp.join(self.annot_path, 'person_keypoints_' + self.data_split + '2017.json'))
        with open(osp.join(self.annot_path, 'coco_smplify_train.json')) as f:
            smplify_results = json.load(f)

        datalist = []
        if self.data_split == 'train':
            for aid in db.anns.keys():
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                imgname = osp.join('train2017', img['file_name'])
                img_path = osp.join(self.img_path, imgname)
                width, height = img['width'], img['height']

                if ann['iscrowd'] or (ann['num_keypoints'] == 0):
                    continue

                # bbox
                bbox = process_bbox(ann['bbox'])
                if bbox is None: continue

                # joint coordinates
                joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                joint_valid = (joint_img[:, 2].copy().reshape(-1, 1) > 0).astype(np.float32)
                joint_img[:, 2] = 0

                if str(aid) in smplify_results:
                    dp_data = None
                    smplify_result = smplify_results[str(aid)]
                else:
                    continue

                datalist.append({
                    'img_path': img_path,
                    'img_shape': (height, width),
                    'bbox': bbox,
                    'joint_img': joint_img,  # [org_img_x, org_img_y, 0]
                    'joint_valid': joint_valid,
                    'dp_data': dp_data,
                    'smplify_result': smplify_result
                })

            return datalist

    def get_smpl_coord(self, smpl_param):
        pose, shape = smpl_param['pose'], smpl_param['shape']
        smpl_pose = torch.FloatTensor(pose).view(1,-1)
        smpl_shape = torch.FloatTensor(shape).view(1,-1)
        # smpl_trans = torch.FloatTensor(trans).view(1,-1)

        # change to mean shape if beta is too far from it
        smpl_shape[(smpl_shape.abs() > 3).any(dim=1)] = 0.

        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.mesh_model.layer['neutral'](smpl_pose, smpl_shape)

        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3);
        smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1, 3)

        # meter -> milimeter
        smpl_mesh_coord *= 1000;
        smpl_joint_coord *= 1000;
        return smpl_mesh_coord, smpl_joint_coord

    def add_pelvis_and_neck(self, joint_coord):
        lhip_idx = self.coco_joints_name.index('L_Hip')
        rhip_idx = self.coco_joints_name.index('R_Hip')
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis = pelvis.reshape((1, -1))

        lshoulder_idx = self.coco_joints_name.index('L_Shoulder')
        rshoulder_idx = self.coco_joints_name.index('R_Shoulder')
        neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        neck = neck.reshape((1, -1))

        joint_coord = np.concatenate((joint_coord, pelvis, neck))
        return joint_coord

    def get_joints_from_mesh(self, mesh, joint_set_name, cam_param):
        joint_coord_cam = None
        if joint_set_name == 'human36':
            joint_coord_cam = np.dot(self.joint_regressor_h36m, mesh)
        elif joint_set_name == 'coco':
            joint_coord_cam = np.dot(self.joint_regressor_coco, mesh)
            joint_coord_cam = self.add_pelvis_and_neck(joint_coord_cam)
        # projection
        s, t = np.array(cam_param['s'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32).reshape(2)
        joint_coord_img = (joint_coord_cam[:, :2] / 1000) * s + t.reshape(-1, 2)

        joint_coord_img = np.concatenate((joint_coord_img, np.ones((len(joint_coord_img), 1))), axis=1)
        return joint_coord_cam, joint_coord_img

    def get_fitting_error(self, bbox, coco_from_dataset, coco_from_smpl, coco_joint_valid):
        bbox = process_bbox(bbox.copy(), aspect_ratio=1.0)
        coco_from_smpl_xy1 = np.concatenate((coco_from_smpl[:,:2], np.ones_like(coco_from_smpl[:,0:1])),1)
        coco_from_smpl, _ = j2d_processing(coco_from_smpl_xy1, (64, 64), bbox, 0, 0, None)
        coco_from_dataset_xy1 = np.concatenate((coco_from_dataset[:,:2], np.ones_like(coco_from_smpl[:,0:1])),1)
        coco_from_dataset, trans = j2d_processing(coco_from_dataset_xy1, (64, 64), bbox, 0, 0, None)

        # vis
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # img = cv2.warpAffine(img, trans, (64, 64), flags=cv2.INTER_LINEAR)
        # vis_2d_pose(coco_from_smpl, img, self.coco_origin_skeleton, prefix='from smpl')
        # vis_2d_pose(coco_from_dataset, img, self.coco_origin_skeleton, prefix='from dataset')

        # mask joint coordinates
        coco_joint = coco_from_dataset[:,:2][np.tile(coco_joint_valid,(1,2))==1].reshape(-1,2)
        coco_from_smpl = coco_from_smpl[:,:2][np.tile(coco_joint_valid,(1,2))==1].reshape(-1,2)

        error = np.sqrt(np.sum((coco_joint - coco_from_smpl)**2,1)).mean()
        return error

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_shape, bbox, dp_data, smplify_result = data['img_shape'], data['bbox'], data['dp_data'], data['smplify_result']
        flip, rot = augm_params(is_train=(self.data_split == 'train'))
        # img_name = img_path.split('/')[-1][:-4]

        smpl_param, cam_param = smplify_result['smpl_param'], smplify_result['cam_param']
        # regress h36m, coco joints
        mesh_cam, joint_cam_smpl = self.get_smpl_coord(smpl_param)
        joint_cam_h36m, joint_img_h36m = self.get_joints_from_mesh(mesh_cam, 'human36', cam_param)
        joint_cam_coco, joint_img_coco = self.get_joints_from_mesh(mesh_cam, 'coco', cam_param)
        # vis_2d_pose(joint_img_h36m, img_path, self.human36_skeleton, prefix='h36m joint')
        # vis_2d_pose(joint_img_coco, img_path, self.coco_skeleton, prefix='coco joint')
        # vis_3d_pose(joint_cam_h36m, self.human36_skeleton, 'human36', gt=True)

        # root relative camera coordinate
        mesh_cam = mesh_cam - joint_cam_h36m[:1]
        joint_cam_coco = joint_cam_coco - joint_cam_coco[-2:-1]
        joint_cam_h36m = joint_cam_h36m - joint_cam_h36m[:1]

        if self.input_joint_name == 'coco':
            joint_img, joint_cam = joint_img_coco, joint_cam_coco
        elif self.input_joint_name == 'human36':
            joint_img, joint_cam = joint_img_h36m, joint_cam_h36m

        # make new bbox
        tight_bbox = get_bbox(joint_img)
        bbox = process_bbox(tight_bbox.copy())

        # aug
        joint_img, trans = j2d_processing(joint_img.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]), bbox, rot, 0, None)
        if not cfg.DATASET.use_gt_input:
            joint_img = self.replace_joint_img(joint_img, tight_bbox, trans)
        if flip:
            joint_img = flip_2d_joint(joint_img, cfg.MODEL.input_shape[1], self.flip_pairs)
        joint_cam = j3d_processing(joint_cam, rot, flip, self.flip_pairs)

        #  -> 0~1
        joint_img = joint_img[:, :2]
        joint_img /= np.array([[cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]]])

        # normalize loc&scale
        mean, std = np.mean(joint_img, axis=0), np.std(joint_img, axis=0)
        joint_img = (joint_img.copy() - mean) / std

        if cfg.MODEL.name == 'pose2mesh_net':
            # default valid
            mesh_valid = np.ones((len(mesh_cam), 1), dtype=np.float32)
            reg_joint_valid = np.ones((len(joint_cam_h36m), 1), dtype=np.float32)
            lift_joint_valid = np.ones((len(joint_cam), 1), dtype=np.float32)
            error = self.get_fitting_error(tight_bbox, data['joint_img'], joint_img_coco[:17], data['joint_valid'])
            if error > self.fitting_thr:
                mesh_valid[:], reg_joint_valid[:], lift_joint_valid[:] = 0, 0, 0

            inputs = {'pose2d': joint_img}
            targets = {'mesh': mesh_cam / 1000, 'lift_pose3d': joint_cam, 'reg_pose3d': joint_cam_h36m}
            meta = {'mesh_valid': mesh_valid, 'lift_pose3d_valid': lift_joint_valid, 'reg_pose3d_valid': reg_joint_valid}

            return inputs, targets, meta

        elif cfg.MODEL.name == 'posenet':
            # default valid
            joint_valid = np.ones((len(joint_cam), 1), dtype=np.float32)
            # compute fitting error
            error = self.get_fitting_error(tight_bbox, data['joint_img'], joint_img_coco[:17], data['joint_valid'])
            if error > self.fitting_thr:
                joint_valid[:, :] = 0

            return joint_img, joint_cam, joint_valid

    def replace_joint_img(self, joint_img, bbox, trans):
        if self.input_joint_name == 'coco':
            joint_img_coco = joint_img
            if self.data_split == 'train':
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                pt1 = affine_transform(np.array([xmin, ymin]), trans)
                pt2 = affine_transform(np.array([xmax, ymin]), trans)
                pt3 = affine_transform(np.array([xmax, ymax]), trans)
                area = math.sqrt(pow(pt2[0] - pt1[0], 2) + pow(pt2[1] - pt1[1], 2)) * math.sqrt(
                    pow(pt3[0] - pt2[0], 2) + pow(pt3[1] - pt2[1], 2))
                joint_img_coco[:17, :] = synthesize_pose(joint_img_coco[:17, :], area, num_overlap=0)
                return joint_img_coco

        elif self.input_joint_name == 'human36':
            joint_img_h36m = joint_img
            if self.data_split == 'train':
                joint_syn_error = (self.generate_syn_error() / 256) * np.array([cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]], dtype=np.float32)
                joint_img_h36m = joint_img_h36m[:, :2] + joint_syn_error
                return joint_img_h36m


