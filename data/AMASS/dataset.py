import os
import os.path as osp
import numpy as np
import glob
import copy
import json
import cv2
import scipy
import random
import math
import torch
import transforms3d

from Human36M.noise_stats import error_distribution
from core.config import cfg 
from funcs_utils import stop
from noise_utils import synthesize_pose
from smpl import SMPL
from coord_utils import cam2pixel, process_bbox, get_bbox, euler2mat
from aug_utils import augm_params, j2d_processing, affine_transform, j3d_processing, flip_2d_joint
from vis import vis_3d_pose, vis_2d_pose


class AMASS_CMU(torch.utils.data.Dataset):
    def __init__(self, data_split, args):
        dataset_name = 'AMASS'
        self.data_split = 'train'
        self.debug = args.debug

        self.data_path = osp.join(cfg.data_dir, dataset_name, 'data')
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
        self.human36_joint_num = 17
        self.human36_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
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
        print('Load annotations of AMASS')
        sub_dataset_list = glob.glob(f'{self.data_path}/*')
        h36m_cam_Rs = np.array([[-0.9153617 ,  0.40180838,  0.02574755],[ 0.05154812,  0.18037356, -0.9822465 ], [-0.39931902, -0.89778364, -0.18581952]], dtype=np.float32),\
                      np.array([[ 0.92816836,  0.37215385,  0.00224838],[ 0.08166409, -0.1977723 , -0.9768404 ], [-0.36309022,  0.9068559 , -0.2139576 ]], dtype=np.float32),\
                      np.array([[-0.91415495, -0.40277803, -0.04572295],[-0.04562341,  0.2143085 , -0.97569996], [ 0.4027893 , -0.8898549 , -0.21428728]], dtype=np.float32),\
                      np.array([[ 0.91415626, -0.40060705,  0.06190599],[-0.05641001, -0.2769532 , -0.9592262 ], [ 0.40141782,  0.8733905 , -0.27577674]], dtype=np.float32)

        datalist = []
        for sub in sub_dataset_list:
            sub_name = sub.split('/')[-1]

            if cfg.MODEL.name == 'pose2mesh_net':
                if 'CMU' not in sub_name:
                    continue
            elif cfg.MODEL.name == 'posenet':
                if 'CMU' not in sub_name and 'BML' not in sub_name:
                    continue

            sampling_ratio = self.get_subsampling_ratio(sub_name.lower())
            seq_name_list = glob.glob(f'{sub}/*')
            for seq in seq_name_list:
                file_list = glob.glob(f'{seq}/*_poses.npz')
                for file in file_list:
                    # data load
                    data = np.load(file)
                    poses = data['poses']  # (frame_num, 156)
                    dmpls = data['dmpls'] # (frame_num, 8)
                    trans = data['trans']  # (frame_num, 3)
                    betas = data['betas']  # (16,)
                    gender = data['gender']  # male

                    for frame_idx in range(len(poses)):
                        if frame_idx % sampling_ratio != 0:
                            continue
                        # get vertex and joint coordinates
                        pose = poses[frame_idx:frame_idx+1, :72]
                        beta = betas[None, :10]

                        # # set camera parameters
                        # x_rot, y_rot = math.pi/2, math.pi
                        # sR = torch.FloatTensor([x_rot, y_rot, 0])
                        # sR = euler2mat(sR).numpy()
                        #
                        # elevation = 0# random.uniform(- math.pi / 3, math.pi / 3)  # set random elevation here
                        # azimuth = random.uniform(- math.pi, + math.pi)  # set random azimuth here
                        #
                        # R = torch.FloatTensor([elevation, azimuth, 0])
                        # R = euler2mat(R).numpy()
                        # R = np.dot(R, sR)

                        # set camera parameters
                        for R in h36m_cam_Rs:
                            t = [0, 0, 10]
                            focal = [1500, 1500]
                            princpt = [500, 500]

                            cam_param = {'R': R, 't': t, 'focal': focal, 'princpt': princpt}
                            smpl_param = {'pose': pose, 'shape': beta}
                            img_shape = (princpt[0]*2+1, princpt[1]*2+1)  # (h, w)

                            datalist.append({
                                'smpl_param': smpl_param,
                                'cam_param': cam_param,
                                'img_shape': img_shape
                            })

                if self.debug:
                    break

        return datalist

    def get_subsampling_ratio(self, dataset_name):
        if dataset_name == 'cmu':
            return 60  # 120 -> 10 fps
        elif dataset_name == 'mpi_mosh':
            return 10
        elif dataset_name == 'bmlrub':
            return 10
        elif dataset_name == 'bmlmovi':
            return 10
        else:
            return 5

    def get_smpl_coord(self, smpl_param, cam_param):
        pose, shape = smpl_param['pose'], smpl_param['shape']
        smpl_pose = torch.FloatTensor(pose).view(-1, 3);
        smpl_shape = torch.FloatTensor(shape).view(1, -1);  # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3), np.array(cam_param['t'], dtype=np.float32).reshape( 3)  # camera rotation and translation

        # rotate global pose
        root_pose = smpl_pose[self.smpl_root_joint_idx, :].numpy()
        angle = np.linalg.norm(root_pose)
        root_pose = transforms3d.axangles.axangle2mat(root_pose / angle, angle)

        root_pose = np.dot(R, root_pose)
        axis, angle = transforms3d.axangles.mat2axangle(root_pose)
        root_pose = axis * angle
        smpl_pose[self.smpl_root_joint_idx] = torch.from_numpy(root_pose)
        smpl_pose = smpl_pose.view(1, -1)

        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.mesh_model.layer['neutral'](smpl_pose, smpl_shape)

        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3);
        smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1, 3)

        # translation
        smpl_mesh_coord += t.reshape(-1, 3)
        smpl_joint_coord += t.reshape(-1, 3)

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
        joint_coord_img = cam2pixel(joint_coord_cam / 1000, f, c)

        joint_coord_img[:, 2] = 1
        return joint_coord_cam, joint_coord_img

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        flip, rot = augm_params(is_train=(self.data_split == 'train'))

        # get smpl mesh, joints
        smpl_param, cam_param, img_shape = data['smpl_param'], data['cam_param'], data['img_shape']
        mesh_cam, joint_cam_smpl = self.get_smpl_coord(smpl_param, cam_param)

        # regress coco joints
        joint_cam_h36m, joint_img_h36m = self.get_joints_from_mesh(mesh_cam, 'human36', cam_param)
        joint_cam_coco, joint_img_coco = self.get_joints_from_mesh(mesh_cam, 'coco', cam_param)
        # debug vis
        # vis_3d_pose(joint_cam_coco, self.coco_skeleton, joint_set_name='coco', prefix=f'coco_joint_cam_{idx}')
        # img = np.zeros((int(img_shape[0]), int(img_shape[1]), 3))
        # vis_2d_pose(joint_img_coco, img, self.coco_skeleton, prefix='coco joint img')

        # root relative camera coordinate
        mesh_cam = mesh_cam - joint_cam_h36m[:1]
        joint_cam_coco = joint_cam_coco - joint_cam_coco[-2:-1]
        joint_cam_h36m = joint_cam_h36m - joint_cam_h36m[:1]

        # joint_cam is PoseNet target
        if self.input_joint_name == 'coco':
            joint_img, joint_cam = joint_img_coco, joint_cam_coco
        elif self.input_joint_name == 'human36':
            joint_img, joint_cam = joint_img_h36m, joint_cam_h36m

        # make new bbox
        tight_bbox = get_bbox(joint_img)
        bbox = process_bbox(tight_bbox.copy())

        # aug
        joint_img, trans = j2d_processing(joint_img.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]),
                                          bbox, rot, 0, None)
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