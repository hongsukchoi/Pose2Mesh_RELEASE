import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
from pycocotools.coco import COCO

from Human36M.noise_stats import error_distribution
from core.config import cfg 
from funcs_utils import stop
from noise_utils import synthesize_pose
from smpl import SMPL
from coord_utils import cam2pixel, process_bbox, get_bbox
from aug_utils import augm_params, j2d_processing, affine_transform, transform_joint_to_other_db, j3d_processing
from vis import vis_3d_pose, vis_2d_pose


class MuCo(torch.utils.data.Dataset):
    def __init__(self, data_split, args):
        dataset_name = 'MuCo'
        self.data_split = 'train'
        self.debug = args.debug
        self.img_dir = osp.join(cfg.data_dir, dataset_name, 'data')
        self.annot_path = osp.join(cfg.data_dir, dataset_name, 'data', 'MuCo-3DHP.json')
        self.smpl_param_path = osp.join(cfg.data_dir, dataset_name, 'data', 'smpl_param.json')
        self.fitting_thr = 45  # milimeter

        # MuCo joint set
        self.muco_joint_num = 21
        self.muco_joints_name = (
        'Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee',
        'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
        self.muco_flip_pairs = ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20))
        self.muco_skeleton = (
        (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20),
        (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))
        self.muco_root_joint_idx = self.muco_joints_name.index('Pelvis')

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
        self.human36_joints_name = (
        'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.human36_skeleton = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
        (2, 3), (0, 4), (4, 5), (5, 6))
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
        if self.data_split == 'train':
            db = COCO(self.annot_path)
            with open(self.smpl_param_path) as f:
                smpl_params = json.load(f)
        else:
            print('Unknown data subset')
            assert 0

        datalist = []
        for iid in db.imgs.keys():
            img = db.imgs[iid]
            img_id = img["id"]
            img_width, img_height = img['width'], img['height']
            imgname = img['file_name']
            img_path = osp.join(self.img_dir, imgname)
            focal = img["f"]
            princpt = img["c"]
            cam_param = {'focal': focal, 'princpt': princpt}

            # crop the closest person to the camera
            ann_ids = db.getAnnIds(img_id)
            anns = db.loadAnns(ann_ids)

            # sample closest subject
            root_depths = [ann['keypoints_cam'][self.muco_root_joint_idx][2] for ann in anns]
            closest_pid = root_depths.index(min(root_depths))
            pid_list = [closest_pid]

            # sample close subjects
            # for i in range(len(anns)):
            #     if i == closest_pid:
            #         continue
            #     picked = True
            #     for j in range(len(anns)):
            #         if i == j:
            #             continue
            #         dist = (np.array(anns[i]['keypoints_cam'][self.muco_root_joint_idx]) - np.array(
            #             anns[j]['keypoints_cam'][self.muco_root_joint_idx])) ** 2
            #         dist_2d = math.sqrt(np.sum(dist[:2]))
            #         dist_3d = math.sqrt(np.sum(dist))
            #         if dist_2d < 500 or dist_3d < 500:
            #             picked = False
            #     if picked:
            #         pid_list.append(i)

            for pid in pid_list:
                joint_cam = np.array(anns[pid]['keypoints_cam'])
                joint_img = np.array(anns[pid]['keypoints_img'])
                joint_img = np.concatenate([joint_img, joint_cam[:, 2:]], 1)
                joint_valid = np.ones((self.muco_joint_num, 3))

                bbox = process_bbox(anns[pid]['bbox'])
                if bbox is None: continue

                # check smpl parameter exist
                try:
                    smpl_param = smpl_params[str(ann_ids[pid])]
                    pose, shape, trans = np.array(smpl_param['pose']), np.array(smpl_param['shape']), np.array(smpl_param['trans'])
                    sum = pose.sum() + shape.sum() + trans.sum()
                    if np.isnan(sum):
                        continue
                except KeyError:
                    continue

                datalist.append({
                    'annot_id': ann_ids[pid],
                    'img_path': img_path,
                    'img_shape': (img_height, img_width),
                    'bbox': bbox,
                    'joint_img': joint_img,
                    'joint_cam': joint_cam,
                    'joint_valid': joint_valid,
                    'cam_param': cam_param,
                    'smpl_param': smpl_param
                })

            if self.debug and len(datalist) > 10000:
                break

        return datalist

    def get_smpl_coord(self, smpl_param):
        pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
        smpl_pose = torch.FloatTensor(pose).view(1, -1);
        smpl_shape = torch.FloatTensor(shape).view(1, -1);  # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        smpl_trans = torch.FloatTensor(trans).view(1,
                                                   3)  # translation vector from smpl coordinate to muco world coordinate
        smpl_shape[(smpl_shape.abs() > 3).any(dim=1)] = 0.

        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.mesh_model.layer['neutral'](smpl_pose, smpl_shape, smpl_trans)

        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3);
        smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1, 3)
        smpl_face_kps_coord = smpl_mesh_coord[self.face_kps_vertex, :].reshape(-1, 3)
        smpl_joint_coord = np.concatenate((smpl_joint_coord, smpl_face_kps_coord))

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
        f, c = np.array(cam_param['focal'], dtype=np.float32), np.array(cam_param['princpt'], dtype=np.float32)
        joint_coord_img = cam2pixel(joint_coord_cam, f, c)

        joint_coord_img[:, 2] = 1
        return joint_coord_cam, joint_coord_img

    def get_fitting_error(self, muco_joint, smpl_mesh):
        muco_joint = muco_joint.copy()
        muco_joint = muco_joint - muco_joint[self.muco_root_joint_idx, None, :]  # root-relative
        muco_joint_valid = np.ones((self.muco_joint_num, 3), dtype=np.float32)

        # transform to h36m joint set
        h36m_joint = transform_joint_to_other_db(muco_joint, self.muco_joints_name, self.human36_joints_name)
        h36m_joint_valid = transform_joint_to_other_db(muco_joint_valid, self.muco_joints_name,
                                                       self.human36_joints_name)
        h36m_joint = h36m_joint[h36m_joint_valid == 1].reshape(-1, 3)

        h36m_from_smpl = np.dot(self.joint_regressor_h36m, smpl_mesh)
        h36m_from_smpl = h36m_from_smpl[h36m_joint_valid == 1].reshape(-1, 3)
        h36m_from_smpl = h36m_from_smpl - np.mean(h36m_from_smpl, 0)[None, :] + np.mean(h36m_joint, 0)[None,
                                                                                :]  # translation alignment
        error = np.sqrt(np.sum((h36m_joint - h36m_from_smpl) ** 2, 1)).mean()
        return error

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, smpl_param, cam_param = data['img_path'], data['img_shape'], data['bbox'], data[
            'smpl_param'], data['cam_param']
        flip, rot = augm_params(is_train=(self.data_split == 'train'))

        # regress h36m, coco joints
        mesh_cam, joint_cam_smpl = self.get_smpl_coord(smpl_param)
        joint_cam_h36m, joint_img_h36m = self.get_joints_from_mesh(mesh_cam, 'human36', cam_param)
        joint_cam_coco, joint_img_coco = self.get_joints_from_mesh(mesh_cam, 'coco', cam_param)
        # vis_3d_save(joint_cam_h36m, self.human36_skeleton, prefix='3d_gt', gt=True)
        # vis_2d_joints(joint_img_h36m, img_path, self.human36_skeleton, prefix='h36m joint')
        # vis_2d_joints(joint_img_coco, img_path, self.coco_skeleton, prefix='coco joint')

        # root relative camera coordinate
        mesh_cam = mesh_cam - joint_cam_h36m[:1]
        joint_cam_coco = joint_cam_coco - joint_cam_coco[-2:-1]
        joint_cam_h36m = joint_cam_h36m - joint_cam_h36m[:1]

        if self.input_joint_name == 'coco':
            joint_img, joint_cam = joint_img_coco, joint_cam_coco
        elif self.input_joint_name == 'human36':
            joint_img, joint_cam = joint_img_h36m, joint_cam_h36m

        # make new bbox
        init_bbox = get_bbox(joint_img)
        bbox = process_bbox(init_bbox.copy())

        # aug
        joint_img, trans = j2d_processing(joint_img.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]),
                                          bbox, rot, flip, self.flip_pairs)
        joint_cam = j3d_processing(joint_cam, rot, flip, self.flip_pairs)

        if not cfg.DATASET.use_gt_input:
            joint_img = self.replace_joint_img(joint_img, bbox, trans)

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
            # if fitted mesh is too far from h36m gt, discard it
            error = self.get_fitting_error(joint_cam_h36m, mesh_cam)
            if error > self.fitting_thr:
                mesh_valid[:], reg_joint_valid[:], lift_joint_valid[:] = 0, 0, 0

            inputs = {'pose2d': joint_img}
            targets = {'mesh': mesh_cam / 1000, 'lift_pose3d': joint_cam, 'reg_pose3d': joint_cam_h36m}
            meta = {'mesh_valid': mesh_valid, 'lift_pose3d_valid': lift_joint_valid, 'reg_pose3d_valid': reg_joint_valid}

            return inputs, targets, meta

        elif cfg.MODEL.name == 'posenet':
            # default valid
            joint_valid = np.ones((len(joint_cam), 1), dtype=np.float32)
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
                joint_syn_error = (self.generate_syn_error() / 256) * np.array(
                    [cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]], dtype=np.float32)
                joint_img_h36m = joint_img_h36m[:, :2] + joint_syn_error
                return joint_img_h36m
