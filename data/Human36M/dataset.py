import os.path as osp
import numpy as np
import math
import torch
import json
import copy
import transforms3d
import scipy.sparse
import cv2
from pycocotools.coco import COCO

from core.config import cfg 
from graph_utils import build_coarse_graphs
from noise_utils import synthesize_pose

from smpl import SMPL
from coord_utils import world2cam, cam2pixel, process_bbox, rigid_align, get_bbox
from aug_utils import affine_transform, j2d_processing, augm_params, j3d_processing, flip_2d_joint
from Human36M.noise_stats import error_distribution

from funcs_utils import save_obj, stop
from vis import vis_3d_pose, vis_2d_pose


class Human36M(torch.utils.data.Dataset):
    def __init__(self, mode, args):
        dataset_name = 'Human36M'
        self.debug = args.debug
        self.data_split = mode
        self.img_dir = osp.join(cfg.data_dir, dataset_name, 'images')
        self.annot_path = osp.join(cfg.data_dir, dataset_name, 'annotations')
        self.subject_genders = {1: 'female', 5: 'female', 6: 'male', 7: 'female', 8: 'male', 9: 'male', 11: 'male'}
        self.protocol = 2
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                            'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog',
                            'WalkTogether']
        self.fitting_thr = 25  # milimeter

        # SMPL joint set
        self.mesh_model = SMPL()
        self.smpl_root_joint_idx = self.mesh_model.root_joint_idx
        self.smpl_face_kps_vertex = self.mesh_model.face_kps_vertex
        self.smpl_vertex_num = 6890
        self.smpl_joint_num = 24
        self.smpl_flip_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
        self.smpl_skeleton = (
            (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
            (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15))
        self.joint_regressor_smpl = self.mesh_model.layer['neutral'].th_J_regressor

        # H36M joint set
        self.human36_joint_num = 17
        self.human36_joints_name = (
        'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.human36_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.human36_skeleton = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
        (2, 3), (0, 4), (4, 5), (5, 6))
        self.human36_root_joint_idx = self.human36_joints_name.index('Pelvis')
        self.human36_error_distribution = self.get_stat()
        self.human36_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.joint_regressor_human36 = self.mesh_model.joint_regressor_h36m

        # COCO joint set
        self.coco_joint_num = 19  # 17 + 2, manually added pelvis and neck
        self.coco_joints_name = (
            'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
            'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
        self.coco_flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        self.coco_skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15), #(5, 6), #(11, 12),
            (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 0))
        self.joint_regressor_coco = self.mesh_model.joint_regressor_coco

        self.input_joint_name = cfg.DATASET.input_joint_set  # 'coco'
        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(self.input_joint_name)

        self.datalist, skip_idx, skip_img_path = self.load_data()
        if self.data_split == 'test':
            det_2d_data_path = osp.join(cfg.data_dir, dataset_name, 'absnet_output_on_testset.json')
            self.datalist_pose2d_det = self.load_pose2d_det(det_2d_data_path, skip_img_path)
            print("Check lengths of annotation and detection output: ", len(self.datalist), len(self.datalist_pose2d_det))

        self.graph_Adj, self.graph_L, self.graph_perm, self.graph_perm_reverse = \
            build_coarse_graphs(self.mesh_model.face, self.joint_num, self.skeleton, self.flip_pairs, levels=9)

    def load_pose2d_det(self, data_path, skip_list):
        pose_list = []
        with open(data_path) as f:
            data = json.load(f)
            for img_path, pose2d in data.items():
                pose2d = np.array(pose2d, dtype=np.float32)
                if img_path in skip_list:
                    continue
                pose_list.append({'img_name': img_path, 'pose2d': pose2d})
        pose_list = sorted(pose_list, key=lambda x: x['img_name'])
        return pose_list

    def get_joint_setting(self, joint_category='human36'):
        joint_num = eval(f'self.{joint_category}_joint_num')
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')

        return joint_num, skeleton, flip_pairs

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5  # 50
        elif self.data_split == 'test':
            return 50 #
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            if self.protocol == 1:
                subject = [1, 5, 6, 7, 8, 9]
            elif self.protocol == 2:
                subject = [1, 5, 6, 7, 8]
        elif self.data_split == 'test':
            if self.protocol == 1:
                subject = [11]
            elif self.protocol == 2:
                subject = [9, 11]
        else:
            assert 0, print("Unknown subset")

        if self.debug:
            subject = subject[0:1]

        return subject

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

    def load_data(self):
        print('Load annotations of Human36M Protocol ' + str(self.protocol))
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()

        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}
        smpl_params = {}
        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'), 'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k, v in annot.items():
                    db.dataset[k] = v
            else:
                for k, v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'), 'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'), 'r') as f:
                joints[str(subject)] = json.load(f)
            # smpl parameter load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_smpl_param.json'), 'r') as f:
                smpl_params[str(subject)] = json.load(f)
        db.createIndex()

        skip_idx = []
        datalist = []
        skip_img_idx = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_name = img_path.split('/')[-1]

            # check subject and frame_idx
            frame_idx = img['frame_idx'];

            if frame_idx % sampling_ratio != 0:
                continue

            # check smpl parameter exist
            subject = img['subject'];
            action_idx = img['action_idx'];

            subaction_idx = img['subaction_idx'];
            frame_idx = img['frame_idx'];
            try:
                smpl_param = smpl_params[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)]
            except KeyError:
                skip_idx.append(image_id)
                skip_img_idx.append(img_path.split('/')[-1])
                continue

            smpl_param['gender'] = 'neutral'  # self.subject_genders[subject] # set corresponding gender

            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject)][str(cam_idx)]
            R, t, f, c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'],
                                                                              dtype=np.float32), np.array(
                cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}

            # project world coordinate to cam, image coordinate space
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)],
                                   dtype=np.float32)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, f, c)
            joint_vis = np.ones((self.human36_joint_num, 1))

            bbox = process_bbox(np.array(ann['bbox']))
            if bbox is None: continue

            datalist.append({
                'img_path': img_path,
                'img_name': img_name,
                'img_id': image_id,
                'bbox': bbox,
                'img_hw': (img['height'], img['width']),
                'joint_img': joint_img,  # [x_img, y_img, z_cam]
                'joint_cam': joint_cam,  # [X, Y, Z] in camera coordinate
                'joint_vis': joint_vis,
                'smpl_param': smpl_param,
                'cam_param': cam_param})

        datalist = sorted(datalist, key=lambda x: x['img_name'])

        return datalist, skip_idx, skip_img_idx

    def get_smpl_coord(self, smpl_param, cam_param):
        pose, shape, trans, gender = smpl_param['pose'], smpl_param['shape'], smpl_param['trans'], smpl_param['gender']
        # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        smpl_pose = torch.FloatTensor(pose).view(-1, 3)
        smpl_shape = torch.FloatTensor(shape).view(1, -1)
        # translation vector from smpl coordinate to h36m world coordinate
        trans = np.array(trans, dtype=np.float32).reshape(3)
        # camera rotation and translation
        R, t = np.array(cam_param['R'],dtype=np.float32).reshape(3, 3), np.array(cam_param['t'],dtype=np.float32).reshape(3)

        # change to mean shape if beta is too far from it
        smpl_shape[(smpl_shape.abs() > 3).any(dim=1)] = 0.

        # transform world coordinate to camera coordinate
        root_pose = smpl_pose[self.smpl_root_joint_idx, :].numpy()
        angle = np.linalg.norm(root_pose)
        root_pose = transforms3d.axangles.axangle2mat(root_pose / angle, angle)
        root_pose = np.dot(R, root_pose)
        axis, angle = transforms3d.axangles.mat2axangle(root_pose)
        root_pose = axis * angle
        smpl_pose[self.smpl_root_joint_idx] = torch.from_numpy(root_pose)

        smpl_pose = smpl_pose.view(1, -1)

        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.mesh_model.layer[gender](smpl_pose, smpl_shape)

        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3);
        smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1, 3)
        # smpl_face_kps_coord = smpl_mesh_coord[self.face_kps_vertex, :].reshape(-1, 3)
        # smpl_joint_coord = np.concatenate((smpl_joint_coord, smpl_face_kps_coord))

        # compenstate rotation (translation from origin to root joint was not cancled)
        smpl_trans = np.array(trans, dtype=np.float32).reshape(
            3)  # translation vector from smpl coordinate to h36m world coordinate
        smpl_trans = np.dot(R, smpl_trans[:, None]).reshape(1, 3) + t.reshape(1, 3) / 1000
        root_joint_coord = smpl_joint_coord[self.smpl_root_joint_idx].reshape(1, 3)
        smpl_trans = smpl_trans - root_joint_coord + np.dot(R, root_joint_coord.transpose(1, 0)).transpose(1, 0)

        # translation
        smpl_mesh_coord += smpl_trans; smpl_joint_coord += smpl_trans

        # meter -> milimeter
        smpl_mesh_coord *= 1000; smpl_joint_coord *= 1000;

        return smpl_mesh_coord, smpl_joint_coord

    def get_fitting_error(self, h36m_joint, smpl_mesh):
        h36m_joint = h36m_joint - h36m_joint[self.human36_root_joint_idx,None,:] # root-relative

        h36m_from_smpl = np.dot(self.joint_regressor_human36, smpl_mesh)
        # translation alignment
        h36m_from_smpl = h36m_from_smpl - np.mean(h36m_from_smpl,0)[None,:] + np.mean(h36m_joint,0)[None,:]
        error = np.sqrt(np.sum((h36m_joint - h36m_from_smpl)**2,1)).mean()
        return error

    def get_coco_from_mesh(self, mesh_coord_cam, cam_param):
        # regress coco joints
        joint_coord_cam = np.dot(self.joint_regressor_coco, mesh_coord_cam)
        joint_coord_cam = self.add_pelvis_and_neck(joint_coord_cam)
        # projection
        f, c = cam_param['focal'], cam_param['princpt']
        joint_coord_img = cam2pixel(joint_coord_cam, f, c)

        joint_coord_img[:, 2] = 1
        return joint_coord_cam, joint_coord_img

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

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_id, bbox, smpl_param, cam_param, img_shape = data['img_id'], data['bbox'].copy(), data['smpl_param'].copy(), data['cam_param'].copy(), data['img_hw']
        flip, rot = augm_params(is_train=(self.data_split == 'train'))

        # smpl coordinates
        mesh_cam, joint_cam_smpl = self.get_smpl_coord(smpl_param, cam_param)

        # regress coco joints
        joint_cam_coco, joint_img_coco = self.get_coco_from_mesh(mesh_cam, cam_param)
        # h36m joints from datasets
        joint_cam_h36m, joint_img_h36m = data['joint_cam'], data['joint_img'][:, :2]

        # root relative camera coordinate
        mesh_cam = mesh_cam - joint_cam_h36m[:1]
        # joint_cam_smpl = joint_cam_smpl - joint_cam_h36m[:1]
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
        joint_img, trans = j2d_processing(joint_img.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]), bbox, rot, 0, None)
        if not cfg.DATASET.use_gt_input:
            joint_img = self.replace_joint_img(idx, img_id, joint_img, tight_bbox, trans)
        if flip:
            joint_img = flip_2d_joint(joint_img, cfg.MODEL.input_shape[1], self.flip_pairs)
        joint_cam = j3d_processing(joint_cam, rot, flip, self.flip_pairs)


        # vis
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # new_img = cv2.warpAffine(img, trans, (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]), flags=cv2.INTER_LINEAR)
        # vis_2d_pose(joint_img, new_img, self.human36_skeleton, prefix='detection')
        # vis_3d_pose(joint_cam, self.human36_skeleton, joint_set_name='human36', gt=True)

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
                mesh_valid[:] = 0
                if self.input_joint_name == 'coco':
                    lift_joint_valid[:] = 0

            inputs = {'pose2d': joint_img}
            targets = {'mesh': mesh_cam / 1000, 'lift_pose3d': joint_cam, 'reg_pose3d': joint_cam_h36m}
            meta = {'mesh_valid': mesh_valid, 'lift_pose3d_valid': lift_joint_valid, 'reg_pose3d_valid': reg_joint_valid}

            return inputs, targets, meta

        elif cfg.MODEL.name == 'posenet':
            # default valid
            joint_valid = np.ones((len(joint_cam), 1), dtype=np.float32)

            # if fitted mesh is too far from h36m gt, discard it
            if self.input_joint_name == 'coco':
                error = self.get_fitting_error(joint_cam_h36m, mesh_cam)
                if (error > self.fitting_thr):
                    joint_valid[:] = 0

            return joint_img, joint_cam, joint_valid

    def replace_joint_img(self, idx, img_id, joint_img, bbox, trans):
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
            else:
                joint_img_coco = self.datalist_pose2d_det[img_id]
                joint_img_coco = self.add_pelvis_and_neck(joint_img_coco)
                for i in range(self.coco_joint_num):
                    joint_img_coco[i, :2] = affine_transform(joint_img_coco[i, :2].copy(), trans)
                return joint_img_coco

        elif self.input_joint_name == 'human36':
            joint_img_h36m = joint_img
            if self.data_split == 'train':
                joint_syn_error = (self.generate_syn_error() / 256) * np.array(
                    [cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]], dtype=np.float32)
                joint_img_h36m = joint_img_h36m[:, :2] + joint_syn_error
                return joint_img_h36m
            else:
                det_data = self.datalist_pose2d_det[idx]
                # assert img_name == det_data['img_name'], f"check: {img_name} / {det_data['img_name']}"
                joint_img_h36m = det_data['pose2d'][:, :2]
                for i in range(self.human36_joint_num):
                    joint_img_h36m[i, :2] = affine_transform(joint_img_h36m[i, :2].copy(), trans)
                return joint_img_h36m

    def compute_joint_err(self, pred_joint, target_joint):
        # root align joint
        pred_joint, target_joint = pred_joint - pred_joint[:, :1, :], target_joint - target_joint[:, :1, :]

        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()

        pred_joint, target_joint = pred_joint[:, self.human36_eval_joint, :], target_joint[:, self.human36_eval_joint, :]
        joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error

    def compute_both_err(self, pred_mesh, target_mesh, pred_joint, target_joint):
        # root align joint
        pred_mesh, target_mesh = pred_mesh - pred_joint[:, :1, :], target_mesh - target_joint[:, :1, :]
        pred_joint, target_joint = pred_joint - pred_joint[:, :1, :], target_joint - target_joint[:, :1, :]

        pred_mesh, target_mesh = pred_mesh.detach().cpu().numpy(), target_mesh.detach().cpu().numpy()
        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()

        pred_joint, target_joint = pred_joint[:, self.human36_eval_joint, :], target_joint[:, self.human36_eval_joint, :]
        mesh_mean_error = np.power((np.power((pred_mesh - target_mesh), 2)).sum(axis=2), 0.5).mean()
        joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error, mesh_mean_error

    def evaluate_joint(self, outs):
        print('Evaluation start...')
        annots = self.datalist
        assert len(annots) == len(outs)
        sample_num = len(annots)

        mpjpe = np.zeros((sample_num, len(self.human36_eval_joint)))
        pampjpe = np.zeros((sample_num, len(self.human36_eval_joint)))
        for n in range(sample_num):
            out = outs[n]
            annot = annots[n]

            # render materials
            pose_coord_out, pose_coord_gt = out['joint_coord'], annot['joint_cam']

            # root joint alignment
            pose_coord_out, pose_coord_gt = pose_coord_out - pose_coord_out[:1], pose_coord_gt - pose_coord_gt[:1]
            # sample eval joitns
            pose_coord_out, pose_coord_gt = pose_coord_out[self.human36_eval_joint, :], pose_coord_gt[self.human36_eval_joint, :]

            # pose error calculate
            mpjpe[n] = np.sqrt(np.sum((pose_coord_out - pose_coord_gt) ** 2, 1))
            # perform rigid alignment
            pose_coord_out = rigid_align(pose_coord_out, pose_coord_gt)
            pampjpe[n] = np.sqrt(np.sum((pose_coord_out - pose_coord_gt) ** 2, 1))

        # total pose error
        tot_err = np.mean(mpjpe)
        eval_summary = 'MPJPE (mm)    >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

        tot_err = np.mean(pampjpe)
        eval_summary = 'PA-MPJPE (mm) >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

    def evaluate(self, outs):
        print('Evaluation start...')
        annots = self.datalist
        assert len(annots) == len(outs)
        sample_num = len(outs)

        # eval H36M joints
        pose_error_h36m = np.zeros((sample_num, len(self.human36_eval_joint)))  # pose error
        pose_error_action_h36m = [[] for _ in range(len(self.action_name))]  # pose error for each sequence
        pose_pa_error_h36m = np.zeros((sample_num, len(self.human36_eval_joint)))  # pose error
        pose_pa_error_action_h36m = [[] for _ in range(len(self.action_name))]  # pose error for each sequence

        # eval SMPL joints and mesh vertices
        pose_error = np.zeros((sample_num, self.smpl_joint_num))  # pose error
        pose_error_action = [[] for _ in range(len(self.action_name))]  # pose error for each sequence
        mesh_error = np.zeros((sample_num, self.smpl_vertex_num))  # mesh error
        mesh_error_action = [[] for _ in range(len(self.action_name))]  # mesh error for each sequence
        for n in range(sample_num):
            annot = annots[n]
            out = outs[n]

            # render materials
            img_path = annot['img_path']
            obj_name = '_'.join(img_path.split('/')[-2:])[:-4]

            # root joint alignment
            mesh_coord_out, mesh_coord_gt = out['mesh_coord'], out['mesh_coord_target']
            joint_coord_out, joint_coord_gt = np.dot(self.joint_regressor_smpl, mesh_coord_out), np.dot(self.joint_regressor_smpl, mesh_coord_gt)
            mesh_coord_out = mesh_coord_out - joint_coord_out[self.smpl_root_joint_idx:self.smpl_root_joint_idx+1]
            mesh_coord_gt = mesh_coord_gt - joint_coord_gt[self.smpl_root_joint_idx:self.smpl_root_joint_idx+1]
            pose_coord_out = joint_coord_out - joint_coord_out[self.smpl_root_joint_idx:self.smpl_root_joint_idx+1]
            pose_coord_gt = joint_coord_gt - joint_coord_gt[self.smpl_root_joint_idx:self.smpl_root_joint_idx+1]

            # pose error calculate
            pose_error[n] = np.sqrt(np.sum((pose_coord_out - pose_coord_gt) ** 2, 1))
            img_name = annot['img_path']
            action_idx = int(img_name[img_name.find('act') + 4:img_name.find('act') + 6]) - 2
            pose_error_action[action_idx].append(pose_error[n].copy())

            # mesh error calculate
            mesh_error[n] = np.sqrt(np.sum((mesh_coord_out - mesh_coord_gt) ** 2, 1))
            img_name = annot['img_path']
            action_idx = int(img_name[img_name.find('act') + 4:img_name.find('act') + 6]) - 2
            mesh_error_action[action_idx].append(mesh_error[n].copy())

            # pose error of h36m calculate
            pose_coord_out_h36m = np.dot(self.joint_regressor_human36, mesh_coord_out)
            pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[self.human36_root_joint_idx]
            pose_coord_out_h36m = pose_coord_out_h36m[self.human36_eval_joint, :]
            pose_coord_gt_h36m = annot['joint_cam']
            pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[self.human36_root_joint_idx]
            pose_coord_gt_h36m = pose_coord_gt_h36m[self.human36_eval_joint, :]
            pose_error_h36m[n] = np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m) ** 2, 1))
            pose_coord_out_h36m = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m) # perform rigid alignment
            pose_pa_error_h36m[n] = np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m) ** 2, 1))
            img_name = annot['img_path']
            action_idx = int(img_name[img_name.find('act') + 4:img_name.find('act') + 6]) - 2
            pose_error_action_h36m[action_idx].append(pose_error_h36m[n].copy())
            pose_pa_error_action_h36m[action_idx].append(pose_pa_error_h36m[n].copy())

            vis = cfg.TEST.vis
            if vis and (n % 500 == 0):
                mesh_to_save = mesh_coord_out / 1000
                obj_path = osp.join(cfg.vis_dir, f'{obj_name}.obj')
                save_obj(mesh_to_save, self.mesh_model.face, obj_path)

        # total pose error (H36M joint set)
        tot_err = np.mean(pose_error_h36m)
        metric = 'MPJPE'
        eval_summary = 'Protocol ' + str(self.protocol) + ' H36M pose error (' + metric + ') >> tot: %.2f\n' % (tot_err)
        # pose error for each action
        for i in range(len(pose_error_action_h36m)):
            err = np.mean(np.array(pose_error_action_h36m[i]))
            eval_summary += (self.action_name[i] + ': %.2f ' % err)
        print(eval_summary)

        tot_err = np.mean(pose_pa_error_h36m)
        metric = 'PA-MPJPE'
        eval_summary = 'Protocol ' + str(self.protocol) + ' H36M pose error (' + metric + ') >> tot: %.2f\n' % (tot_err)
        # pose error for each action
        for i in range(len(pose_pa_error_action_h36m)):
            err = np.mean(np.array(pose_pa_error_action_h36m[i]))
            eval_summary += (self.action_name[i] + ': %.2f ' % err)
        print(eval_summary)

        # total pose error (SMPL joint set)
        tot_err = np.mean(pose_error)
        metric = 'MPJPE'
        eval_summary = 'Protocol ' + str(self.protocol) + ' SMPL pose error (' + metric + ') >> tot: %.2f\n' % (tot_err)
        # pose error for each action
        for i in range(len(pose_error_action)):
            err = np.mean(np.array(pose_error_action[i]))
            eval_summary += (self.action_name[i] + ': %.2f ' % err)
        print(eval_summary)

        # total mesh error
        tot_err = np.mean(mesh_error)
        metric = 'MPVPE'
        eval_summary = 'Protocol ' + str(self.protocol) + ' SMPL mesh error (' + metric + ') >> tot: %.2f\n' % (tot_err)
        # mesh error for each action
        for i in range(len(mesh_error_action)):
            err = np.mean(np.array(mesh_error_action[i]))
            eval_summary += (self.action_name[i] + ': %.2f ' % err)
        print(eval_summary)
