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

from core.config import cfg
from funcs_utils import stop, save_obj
from graph_utils import build_coarse_graphs
from smooth_utils import smooth_pose
from smpl import SMPL
from coord_utils import cam2pixel, process_bbox, rigid_align, get_bbox, compute_error_accel
from aug_utils import j2d_processing, j3d_processing, transform_joint_to_other_db
from vis import vis_2d_pose, vis_3d_pose


class PW3D(torch.utils.data.Dataset):
    def __init__(self, data_split, args):
        dataset_name = 'PW3D'
        self.data_split = data_split#'validation'
        self.data_path = osp.join(cfg.data_dir, dataset_name, 'data')
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
        self.openpose_joints_name = ('Nose', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip',  'L_Knee', 'L_Ankle', 'R_Eye', 'L_Eye', 'R_Ear',  'L_Ear', 'Pelvis')

        input_joint_name = 'coco'
        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(input_joint_name)

        self.datalist, self.video_indices = self.load_data()  # self.video_indexes: 37 video, and indices of each video

        # TEMP
        self.graph_Adj, self.graph_L, self.graph_perm, self.graph_perm_reverse = \
            build_coarse_graphs(self.mesh_model.face, self.joint_num, self.skeleton, self.flip_pairs, levels=9)
        print(self.graph_perm_reverse)

        print("3dpw data len: ", len(self.datalist))

    def get_joint_setting(self, joint_category='human36'):
        joint_num = eval(f'self.{joint_category}_joint_num')
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')

        return joint_num, skeleton, flip_pairs

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
        db = COCO(osp.join(self.data_path, '3DPW_latest_' + self.data_split + '.json'))

        # get detected 2d pose
        with open(osp.join(self.data_path,  f'darkpose_3dpw_{self.data_split}set_output.json')) as f:#
            pose2d_outputs = {}
            data = json.load(f)
            for item in data:
                annot_id = str(item['annotation_id'])
                pose2d_outputs[annot_id] = {'coco_joints': np.array(item['keypoints'], dtype=np.float32)[:, :3]}

        datalist = []
        custompose_count = 0
        for aid in db.anns.keys():
            aid = int(aid)
            ann = db.anns[aid]
            image_id = ann['image_id']

            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            sequence_name = img['sequence']
            img_name = img['file_name']

            img_path = osp.join(self.img_path, sequence_name, img_name)
            cam_param = {k: np.array(v, dtype=np.float32) for k,v in img['cam_param'].items()}

            smpl_param = ann['smpl_param']
            pid = ann['person_id']
            vid_name = sequence_name + str(pid)
            bbox = process_bbox(np.array(ann['bbox']))
            if bbox is None: continue

            openpose = np.array(ann['openpose_result'], dtype=np.float32).reshape(-1, 3)
            openpose = self.add_pelvis_and_neck(openpose, self.openpose_joints_name, only_pelvis=True)

            custompose = np.array(pose2d_outputs[str(aid)]['coco_joints'])
            custompose = self.add_pelvis_and_neck(custompose, self.coco_joints_name)
            custompose_count += 1

            datalist.append({
                'annot_id': aid,
                'person_id': pid,
                'image_id': image_id,
                'img_path': img_path,
                'vid_name': vid_name,
                'img_shape': (img_height, img_width),
                'cam_param': cam_param,
                'bbox': bbox,
                'smpl_param': smpl_param,
                'pred_pose2d': custompose
            })

        datalist = sorted(datalist, key=lambda x: (x['person_id'],x['img_path']))
        valid_names = np.array([data['vid_name'] for data in datalist])
        unique_names = np.unique(valid_names)
        video_indices = []
        for u_n in unique_names:
            indexes = valid_names == u_n
            video_indices.append(indexes)

        print("num custom pose: ", custompose_count)
        return datalist, video_indices

    def add_pelvis_and_neck(self, joint_coord, joints_name, only_pelvis=False):
        lhip_idx = joints_name.index('L_Hip')
        rhip_idx = joints_name.index('R_Hip')
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis = pelvis.reshape((1, -1))

        lshoulder_idx = joints_name.index('L_Shoulder')
        rshoulder_idx = joints_name.index('R_Shoulder')
        neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        neck = neck.reshape((1,-1))

        if only_pelvis:
            joint_coord = np.concatenate((joint_coord, pelvis))
        else:
            joint_coord = np.concatenate((joint_coord, pelvis, neck))
        return joint_coord

    def get_coco_from_mesh(self, mesh_coord_cam, cam_param):
        # regress coco joints
        mesh = torch.Tensor(mesh_coord_cam)
        joint_coord_cam = torch.matmul(self.joint_regressor_coco, mesh)
        joint_coord_cam = joint_coord_cam.numpy()
        joint_coord_cam = self.add_pelvis_and_neck(joint_coord_cam, self.coco_joints_name)
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
        annot_id, img_id, img_path, img_shape = data['annot_id'], data['image_id'], data['img_path'], data['img_shape']
        cam_param, bbox, smpl_param = data['cam_param'].copy(), data['bbox'].copy(), data['smpl_param'].copy()
        rot, flip = 0, 0

        # get coco img joints from detection
        joint_img_coco = data['pred_pose2d']
        # vis
        # img = cv2.imread(img_path)
        # vis_2d_pose(joint_img_coco, img, self.coco_skeleton, prefix='vis2dpose', bbox=None)
        # import pdb; pdb.set_trace()

        # smpl coordinates
        mesh_cam, joint_cam_smpl = self.get_smpl_coord(smpl_param)

        # regress h36m, coco cam joints
        joint_cam_coco, gt_joint_img_coco = self.get_coco_from_mesh(mesh_cam, cam_param)
        joint_cam_h36m = self.get_h36mJ_from_mesh(mesh_cam)

        # root relative camera coordinate
        mesh_cam = mesh_cam - joint_cam_h36m[:1]
        joint_cam_coco = joint_cam_coco - joint_cam_coco[-2:-1]
        joint_cam_h36m = joint_cam_h36m - joint_cam_h36m[:1]

        if cfg.DATASET.use_gt_input:
            joint_img_coco = gt_joint_img_coco

        # make new bbox
        bbox = get_bbox(joint_img_coco)
        bbox = process_bbox(bbox.copy())

        # aug
        joint_img_coco, trans = j2d_processing(joint_img_coco.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]),
                                               bbox, rot, flip, None)

        #  -> 0~1
        joint_img_coco = joint_img_coco[:, :2]
        joint_img_coco /= np.array([[cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]]])

        # normalize loc&scale
        mean, std = np.mean(joint_img_coco, axis=0), np.std(joint_img_coco, axis=0)
        joint_img_coco = (joint_img_coco.copy() - mean) / std

        if cfg.MODEL.name == 'pose2mesh_net':
            inputs = {'pose2d': joint_img_coco}
            targets = {'mesh': mesh_cam / 1000, 'reg_pose3d': joint_cam_h36m}
            meta = {'dummy': np.ones(1, dtype=np.float32)}

            return inputs, targets, meta

        elif cfg.MODEL.name == 'posenet':
            joint_valid = np.ones((len(joint_cam_coco), 1), dtype=np.float32)  # dummy
            return joint_img_coco, joint_cam_coco, joint_valid

    def compute_joint_err(self, pred_joint, target_joint):
        # root align joint, coco joint set
        pred_joint, target_joint = pred_joint - pred_joint[:, -2:-1, :], target_joint - target_joint[:, -2:-1, :]

        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()

        joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error

    def compute_both_err(self, pred_mesh, target_mesh, pred_joint, target_joint):
        # root align joint
        pred_mesh, target_mesh = pred_mesh - pred_joint[:, :1, :], target_mesh - target_joint[:,:1, :]
        pred_joint, target_joint = pred_joint - pred_joint[:, :1, :], target_joint - target_joint[:, :1, :]

        pred_mesh, target_mesh = pred_mesh.detach().cpu().numpy(), target_mesh.detach().cpu().numpy()
        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()

        pred_joint, target_joint = pred_joint[:, self.human36_eval_joint, :], target_joint[:, self.human36_eval_joint,
                                                                              :]
        mesh_mean_error = np.power((np.power((pred_mesh - target_mesh), 2)).sum(axis=2), 0.5).mean()
        joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error, mesh_mean_error

    def evaluate_joint(self, outs):
        print('Evaluation start...')
        annots = self.datalist
        assert len(annots) == len(outs)
        sample_num = len(annots)
        sample_num = len(outs)

        mpjpe = np.zeros((sample_num, self.coco_joint_num))  # pose error
        pa_mpjpe = np.zeros((sample_num, self.coco_joint_num))  # pose error

        for n in range(sample_num):
            out = outs[n]
            annot = annots[n]
            img_path = annot['img_path']

            joint_coord_out, joint_coord_gt = out['joint_coord'], out['joint_coord_target']
            # root joint alignment, coco joint set
            joint_coord_out = joint_coord_out - joint_coord_out[-2:-1]
            joint_coord_gt = joint_coord_gt - joint_coord_gt[-2:-1]

            # pose error calculate
            mpjpe[n] = np.sqrt(np.sum((joint_coord_out - joint_coord_gt) ** 2, 1))

            joint_coord_out_aligned = rigid_align(joint_coord_out, joint_coord_gt)
            pa_mpjpe[n] = np.sqrt(np.sum((joint_coord_out_aligned - joint_coord_gt) ** 2, 1))

        tot_err = np.mean(mpjpe)
        eval_summary = 'COCO MPJPE (mm)    >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

        tot_err = np.mean(pa_mpjpe)
        eval_summary = 'COCO PA-MPJPE (mm) >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

    def evaluate(self, outs):
        print('Evaluation start...')
        annots = self.datalist
        assert len(annots) == len(outs)
        sample_num = len(outs)

        mpjpe_h36m = np.zeros((sample_num, len(self.human36_eval_joint))) # pose error
        pampjpe_h36m = np.zeros((sample_num, len(self.human36_eval_joint))) # pose error

        mpjpe_smpl = np.zeros((sample_num, self.smpl_joint_num)) # pose error
        mpvpe = np.zeros((sample_num, self.smpl_vertex_num)) # mesh error
        pa_mpvpe = np.zeros((sample_num, self.smpl_vertex_num)) # mesh error

        pred_j3d, gt_j3d = [], []

        for n in range(sample_num):
            out = outs[n]
            annot = annots[n]
            img_path = annot['img_path']

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

            # mesh_coord_out_aligned = rigid_align(mesh_coord_out, mesh_coord_gt)
            # pa_mpvpe[n] = np.sqrt(np.sum((mesh_coord_out_aligned - mesh_coord_gt)**2,1))

            # pose error of h36m calculate
            pose_coord_out_h36m = np.dot(self.mesh_model.joint_regressor_h36m, mesh_coord_out)
            pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[self.human36_root_joint_idx]
            pose_coord_out_h36m = pose_coord_out_h36m[self.human36_eval_joint, :]
            pose_coord_gt_h36m = np.dot(self.mesh_model.joint_regressor_h36m, mesh_coord_gt)
            pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[self.human36_root_joint_idx]
            pose_coord_gt_h36m = pose_coord_gt_h36m[self.human36_eval_joint, :]

            pred_j3d.append(pose_coord_out_h36m); gt_j3d.append(pose_coord_gt_h36m)

            mpjpe_h36m[n] = np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m)**2,1))
            pose_coord_out_h36m_aligned = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m) # perform rigid alignment
            pampjpe_h36m[n] = np.sqrt(np.sum((pose_coord_out_h36m_aligned - pose_coord_gt_h36m)**2,1))

            vis = cfg.TEST.vis
            if vis and (n % 10):
                mesh_to_save = mesh_coord_out / 1000
                obj_path = osp.join(cfg.vis_dir, f'3dpw_{img_path}.obj')
                save_obj(mesh_to_save, self.mesh_model.face, obj_path)

        """
        print("--------Smoothed output errors--------")
        # print smoothed results
        # compute accel
        accel_error = []
        mpjpe_list = []
        pa_mpjpe_list = []
        pred_j3d, gt_j3d = np.array(pred_j3d), np.array(gt_j3d)
        for vid_idx in self.video_indices:
            pred, gt = pred_j3d[vid_idx], gt_j3d[vid_idx]
            pred = smooth_pose(pred, min_cutoff=0.004, beta=0.005)
            vid_acc_err = compute_error_accel(gt, pred)
            vid_acc_err = np.mean(vid_acc_err)
            accel_error.append(vid_acc_err)

            mpjpe = np.sqrt(np.sum((pred - gt)**2,2))
            mpjpe_list.append(np.mean(mpjpe))
            for idx in range(len(pred)):
                pa_pred = rigid_align(pred[idx], gt[idx])
                pa_mpjpe = np.sqrt(np.sum((pa_pred - gt[idx])**2,1))
                pa_mpjpe_list.append(pa_mpjpe)

        accel_error = np.mean(accel_error)
        eval_summary = 'H36M accel error (mm/s^2): tot: %.2f\n' % (accel_error)
        print(eval_summary)

        tot_err = np.mean(mpjpe_list)
        eval_summary = 'H36M MPJPE (mm)    >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

        tot_err = np.mean(pa_mpjpe_list)
        eval_summary = 'H36M PA-MPJPE (mm) >> tot: %.2f\n' % (tot_err)
        print(eval_summary)
        print("--------Original output errors--------")
        """

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

        # total mesh error
        tot_err = np.mean(pa_mpvpe)
        eval_summary = 'PA-MPVPE (mm)      >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

