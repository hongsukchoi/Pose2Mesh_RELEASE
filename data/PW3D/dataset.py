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
from tqdm import tqdm

from coarsening import coarsen, laplacian, perm_index_reverse, lmax_L, rescale_L
from core.config import config as cfg
from display_utils import display_model

from funcs_utils import stop, save_obj
from graph_utils import build_graph, build_coarse_graphs
from smpl import SMPL
from coord_utils import world2cam, cam2pixel, process_bbox, get_center_scale, rigid_align, get_bbox
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from aug_utils import get_affine_transform, affine_transform, augm_params, j2d_processing, j3d_processing, rotate_2d, my3d_processing
from vis import vis_2d_pose, vis_3d_pose


class PW3D(torch.utils.data.Dataset):
    def __init__(self, data_split, args):
        dataset_name = 'PW3D'
        self.data_split = 'test'
        self.data_path = osp.join(cfg.data_dir, dataset_name, 'data')
        self.det_data_path = osp.join(self.data_path, 'hrnet_output_on_testset.json')
        # simple_output_on_testset.json
        self.img_path = osp.join(cfg.data_dir, dataset_name, 'imageFiles')

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
        self.joint_regressor_smpl = self.mesh_model.layer['neutral'].th_J_regressor

        # H36M joint set
        self.human36_root_joint_idx = 0
        self.human36_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.human36_skeleton = (
            (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
            (2, 3), (0, 4), (4, 5), (5, 6))
        self.joint_regressor_human36 = torch.Tensor(self.mesh_model.joint_regressor_h36m)

        # COCO joint set
        self.coco_joint_num = 19  # 17 + 2, manually added pelvis and neck
        self.coco_joints_name = (
            'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
            'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
        self.coco_flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        self.coco_skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15), #(5, 6), (11, 12))
            (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 0))
        self.joint_regressor_coco = torch.Tensor(self.mesh_model.joint_regressor_coco)

        input_joint_name = 'coco'
        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(input_joint_name)

        self.datalist = self.load_data()
        self.datalist_pose2d_det = self.load_pose2d_det(self.det_data_path)

        self.graph_Adj, self.graph_L, self.graph_perm, self.graph_perm_reverse = \
            build_coarse_graphs(self.mesh_model.face, self.joint_num, self.skeleton, self.flip_pairs, levels=9)

        print("Check lengths of annotation and detection output: ", len(self.datalist), len(self.datalist_pose2d_det))

    def load_pose2d_det(self, data_path):
        with open(data_path) as f:
            datalist = json.load(f)
            datalist = sorted(datalist, key=lambda x: x['annotation_id'])

            return datalist

    def get_joint_setting(self, joint_category='human36'):
        joint_num = eval(f'self.{joint_category}_joint_num')
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')

        return joint_num, skeleton, flip_pairs

    def build_adj(self):
        adj_matrix = np.zeros((self.joint_num, self.joint_num))
        for line in self.skeleton:
            adj_matrix[line] = 1
            adj_matrix[line[1], line[0]] = 1
        for lr in self.flip_pairs:
            adj_matrix[lr] = 1
            adj_matrix[lr[1], lr[0]] = 1

        return adj_matrix + np.eye(self.joint_num)

    def compute_graph(self, levels=9):
        joint_adj = self.build_adj()
        # Build graph
        mesh_adj = build_graph(self.mesh_model.face, self.mesh_model.face.max() + 1)
        graph_Adj, graph_L, graph_perm = coarsen(mesh_adj, levels=levels)

        input_Adj = scipy.sparse.csr_matrix(joint_adj)
        input_Adj.eliminate_zeros()
        input_L = laplacian(input_Adj, normalized=True)

        graph_L[-1] = input_L
        graph_Adj[-1] = input_Adj

        # Compute max eigenvalue of graph Laplacians, rescale Laplacian
        graph_lmax = []
        for i in range(levels):
            graph_lmax.append(lmax_L(graph_L[i]))
            graph_L[i] = rescale_L(graph_L[i], graph_lmax[i])

        return graph_Adj, graph_L, graph_perm, perm_index_reverse(graph_perm[0])

    def get_smpl_coord(self, smpl_param):
        pose, shape, trans, gender = smpl_param['pose'], smpl_param['shape'], smpl_param['trans'], smpl_param['gender']
        smpl_pose = torch.FloatTensor(pose).view(-1, 3);
        smpl_shape = torch.FloatTensor(shape).view(1, -1);
        # translation vector from smpl coordinate to 3dpw world coordinate
        smpl_trans = torch.FloatTensor(trans).view(-1, 3)

        smpl_pose = smpl_pose.view(1, -1)
        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.mesh_model.layer[gender](smpl_pose, smpl_shape, smpl_trans)

        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3);
        smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1, 3)

        # meter -> milimeter
        smpl_mesh_coord *= 1000;
        smpl_joint_coord *= 1000;
        return smpl_mesh_coord, smpl_joint_coord

    def load_data(self):
        print('Load annotations of 3DPW ')
        db = COCO(osp.join(self.data_path, '3DPW_' + self.data_split + '.json'))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']

            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            sequence_name = img['sequence']
            img_name = img['file_name']
            img_path = osp.join(self.img_path, sequence_name, img_name)
            cam_param = {k: np.array(v, dtype=np.float32) for k,v in img['cam_param'].items()}

            smpl_param = ann['smpl_param']
            bbox = process_bbox(np.array(ann['bbox']))
            if bbox is None: continue

            datalist.append({
                'annot_id': aid,
                'image_id': image_id,
                'img_path': img_path,
                'img_shape': (img_height, img_width),
                'cam_param': cam_param,
                'bbox': bbox,
                'smpl_param': smpl_param})

        datalist = sorted(datalist, key=lambda x: x['annot_id'])

        return datalist

    def add_pelvis_and_neck(self, joint_coord):
        lhip_idx = self.coco_joints_name.index('L_Hip')
        rhip_idx = self.coco_joints_name.index('R_Hip')
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis = pelvis.reshape((1, -1))

        lshoulder_idx = self.coco_joints_name.index('L_Shoulder')
        rshoulder_idx = self.coco_joints_name.index('R_Shoulder')
        neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        neck = neck.reshape((1,-1))

        joint_coord = np.concatenate((joint_coord, pelvis, neck))
        return joint_coord

    def get_coco_from_mesh(self, mesh_coord_cam, cam_param):
        # regress coco joints
        mesh = torch.Tensor(mesh_coord_cam)
        joint_coord_cam = torch.matmul(self.joint_regressor_coco, mesh)
        joint_coord_cam = joint_coord_cam.numpy()
        joint_coord_cam = self.add_pelvis_and_neck(joint_coord_cam)
        # projection
        f, c = cam_param['focal'], cam_param['princpt']
        joint_coord_img = cam2pixel(joint_coord_cam, f, c)

        joint_coord_img[:, 2] = 1
        return joint_coord_cam, joint_coord_img

    def get_h36mJ_from_mesh(self, mesh_coord_cam):
        mesh = torch.Tensor(mesh_coord_cam)
        joint_coord_cam = torch.matmul(self.joint_regressor_human36, mesh)
        joint_coord_cam = joint_coord_cam.numpy()

        return joint_coord_cam

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_id, img_path, img_shape = data['image_id'], data['img_path'], data['img_shape']
        cam_param, bbox, smpl_param = data['cam_param'].copy(), data['bbox'].copy(), data['smpl_param'].copy()
        rot, flip = 0, 0

        # Detection
        detection_data = copy.deepcopy(self.datalist_pose2d_det[idx])
        det_annot_id, det_joint_img_coco = detection_data['annotation_id'], np.array(detection_data['keypoints'], dtype=np.float32)
        joint_img_coco = self.add_pelvis_and_neck(det_joint_img_coco)

        # smpl coordinates
        mesh_cam, joint_cam_smpl = self.get_smpl_coord(smpl_param)

        # regress h36m, coco cam joints
        joint_cam_coco, gt_joint_img_coco = self.get_coco_from_mesh(mesh_cam, cam_param)
        joint_cam_h36m = self.get_h36mJ_from_mesh(mesh_cam)

        # root relative camera coordinate
        mesh_cam = mesh_cam - joint_cam_h36m[:1]
        joint_cam_coco = joint_cam_coco - joint_cam_coco[-2:-1]
        joint_cam_h36m = joint_cam_h36m - joint_cam_h36m[:1]

        # make new bbox
        bbox = get_bbox(gt_joint_img_coco)
        bbox = process_bbox(bbox.copy())

        if cfg.DATASET.use_gt_input:
            joint_img_coco = gt_joint_img_coco

        # aug
        joint_img_coco, trans = j2d_processing(joint_img_coco.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]),
                                               bbox, rot, flip, None)

        #  -> 0~1
        joint_img_coco = joint_img_coco[:, :2]
        joint_img_coco /= np.array([[cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]]])

        # normalize loc&scale
        mean, std = np.mean(joint_img_coco, axis=0), np.std(joint_img_coco, axis=0)
        joint_img_coco = (joint_img_coco.copy() - mean) / std

        inputs = {'pose2d': joint_img_coco}
        targets = {'mesh': mesh_cam / 1000, 'reg_pose3d': joint_cam_h36m}
        meta = {'dummy': np.ones(1, dtype=np.float32)}

        return inputs, targets, meta

        return joint_img_coco, joint_cam_coco, mesh_cam / 1000, joint_img_coco, joint_img_coco, joint_cam_h36m

    def evaluate_both(self, pred_mesh, target_mesh, pred_joint, target_joint):
        # root align joint
        pred_mesh = pred_mesh - pred_joint[:, :1, :]
        target_mesh = target_mesh - target_joint[:, :1, :]
        pred_joint = pred_joint - pred_joint[:, :1, :]
        target_joint = target_joint - target_joint[:, :1, :]

        pred_mesh, target_mesh = pred_mesh.detach().cpu().numpy(), target_mesh.detach().cpu().numpy()
        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()

        pred_joint, target_joint = pred_joint[:, self.human36_eval_joint, :], target_joint[:, self.human36_eval_joint,
                                                                              :]
        mesh_mean_error = np.power((np.power((pred_mesh - target_mesh), 2)).sum(axis=2), 0.5).mean()
        joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error, mesh_mean_error

    def evaluate(self, outs):
        print('Evaluation start...')
        annots = self.datalist
        # assert len(annots) == len(outs)
        # sample_num = len(annots)
        sample_num = len(outs)

        mpjpe_h36m = np.zeros((sample_num, len(self.human36_eval_joint))) # pose error
        pampjpe_h36m = np.zeros((sample_num, len(self.human36_eval_joint))) # pose error

        mpjpe_smpl = np.zeros((sample_num, self.smpl_joint_num)) # pose error
        mpvpe = np.zeros((sample_num, self.smpl_vertex_num)) # mesh error

        for n in range(sample_num):
            out = outs[n]
            annot = annots[n]

            mesh_coord_out, mesh_coord_gt = out['mesh_coord'], out['mesh_coord_target']
            joint_coord_out, joint_coord_gt = np.dot(self.joint_regressor_smpl, mesh_coord_out), np.dot(self.joint_regressor_smpl, mesh_coord_gt)
            # root joint alignment
            coord_out_cam = np.concatenate((mesh_coord_out, joint_coord_out))
            coord_out_cam = coord_out_cam - coord_out_cam[self.smpl_vertex_num + self.smpl_root_joint_idx]
            coord_gt_cam = np.concatenate((mesh_coord_gt, joint_coord_gt))
            coord_gt_cam = coord_gt_cam - coord_gt_cam[self.smpl_vertex_num + self.smpl_root_joint_idx]
 
            # pose error calculate
            pose_coord_out = coord_out_cam[self.smpl_vertex_num:,:]
            pose_coord_gt = coord_gt_cam[self.smpl_vertex_num:,:]
            mpjpe_smpl[n] = np.sqrt(np.sum((pose_coord_out - pose_coord_gt)**2,1))

            # mesh error calculate
            mesh_coord_out = coord_out_cam[:self.smpl_vertex_num,:]
            mesh_coord_gt = coord_gt_cam[:self.smpl_vertex_num,:]
            mpvpe[n] = np.sqrt(np.sum((mesh_coord_out - mesh_coord_gt)**2,1))

            # pose error of h36m calculate
            pose_coord_out_h36m = np.dot(self.mesh_model.joint_regressor_h36m, mesh_coord_out)
            pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[self.human36_root_joint_idx]
            pose_coord_out_h36m = pose_coord_out_h36m[self.human36_eval_joint, :]
            pose_coord_gt_h36m = np.dot(self.mesh_model.joint_regressor_h36m, mesh_coord_gt)

            pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[self.human36_root_joint_idx]
            pose_coord_gt_h36m = pose_coord_gt_h36m[self.human36_eval_joint, :]
            mpjpe_h36m[n] = np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m)**2,1))
            pose_coord_out_h36m = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m) # perform rigid alignment
            pampjpe_h36m[n] = np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m)**2,1))

            vis = cfg.TEST.vis
            if vis:
                mesh_to_save = mesh_coord_out / 1000
                obj_path = osp.join(cfg.vis_dir, f'3dpw_{n}.obj')
                save_obj(mesh_to_save, self.mesh_model.face, obj_path)

        # total pose error (H36M joint set)
        tot_err = np.mean(mpjpe_h36m)
        eval_summary = 'H36M MPJPE (mm)    >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

        tot_err = np.mean(pampjpe_h36m)
        eval_summary = 'H36M PA-MPJPE (mm) >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

        # total pose error (SMPL joint set)
        tot_err = np.mean(mpjpe_smpl)
        eval_summary = 'SMPL MPJPE (mm)    >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

        # total mesh error
        tot_err = np.mean(mpvpe)
        eval_summary = 'MPVPE (mm)         >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

