import os
import os.path as osp
import numpy as np
import copy
import json
import torch
import scipy.sparse
from pycocotools.coco import COCO

import __init_path
from core.config import cfg 
from graph_utils import build_coarse_graphs
from smpl import SMPL
from coord_utils import process_bbox, get_bbox
from aug_utils import j2d_processing
from vis import vis_3d_pose, vis_2d_pose


class COCOHuman(torch.utils.data.Dataset):
    def __init__(self, img_res):
        dataset_name = 'COCO'
        self.data_split = 'val'  # 'train' if data_split == 'train' else 'val'
        self.img_path = osp.join(cfg.data_dir, dataset_name, 'images')
        self.annot_path = osp.join(cfg.data_dir, dataset_name, 'annotations')
        self.img_res = img_res

        self.smpl = SMPL()

        # COCO joint set
        self.coco_joint_num = 19  # 17 + 2, manually added pelvis and neck
        self.coco_joints_name = (
            'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
            'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
        self.coco_flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        # self.coco_skeleton = (
        #     (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15),
        #     (5, 6), (11, 12))
        self.coco_skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9),  (11, 13), (13, 15),
            (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 0), (12, 14), (14, 16),)

        self.coco_root_joint_idx = self.coco_joints_name.index('Pelvis')
        self.joint_regressor_coco = self.smpl.joint_regressor_coco

        input_joint_name = 'coco'
        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(input_joint_name)

        self.datalist_cocoj_det = self.load_coco_joints_det(osp.join(cfg.data_dir, dataset_name, 'hrnet_output_on_valset.json'))
        self.datalist = self.load_data()[:6337]

        self.graph_Adj, self.graph_L, self.graph_perm, self.graph_perm_reverse = \
            build_coarse_graphs(self.smpl.face, self.joint_num, self.skeleton, self.flip_pairs, levels=9)

        print("# of COCO annotation and detection data: ", len(self.datalist), len(self.datalist_cocoj_det))

    def get_joint_setting(self, joint_category='human36'):
        joint_num = eval(f'self.{joint_category}_joint_num')
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')

        return joint_num, skeleton, flip_pairs

    def load_coco_joints_det(self, data_path):
        with open(data_path) as f:
            datalist = json.load(f)

        datalist = sorted(datalist, key=lambda x: x['aid'])

        return datalist

    def load_data(self):
        db = COCO(osp.join(self.annot_path, 'person_keypoints_' + self.data_split + '2017.json'))
        datalist = []

        for aid in db.anns.keys():
            ann = db.anns[aid]
            img = db.loadImgs(ann['image_id'])[0]
            imgname = osp.join(f'{self.data_split}2017', img['file_name'])
            img_path = osp.join(self.img_path, imgname)
            width, height = img['width'], img['height']

            if ann['iscrowd'] or (ann['num_keypoints'] == 0):
                continue

            # bbox
            bbox = process_bbox(ann['bbox'])
            if bbox is None: continue

            # joint coordinates
            joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            joint_img = self.add_pelvis_and_neck(joint_img)
            joint_valid = (joint_img[:, 2].copy().reshape(-1, 1) > 0).astype(np.float32)
            joint_img[:, 2] = 0

            datalist.append({
                'aid': aid,
                'img_path': img_path,
                'img_shape': (height, width),
                'bbox': bbox,
                'joint_img': joint_img,  # [org_img_x, org_img_y, 0]
                'joint_valid': joint_valid,
                'root_joint_depth': np.array([10], dtype=np.float32)[0]  # dummy
            })

        datalist = sorted(datalist, key=lambda x: x['aid'])

        return datalist

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

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        flip, rot = 0, 0

        aid, img_path, img_shape = data['aid'], data['img_path'], data['img_shape']
        gt_joint_img_coco = data['joint_img']

        det_data = self.datalist_cocoj_det[idx]
        det_aid = det_data['aid']
        assert det_aid == aid, f"detection aid: {det_aid}, dataset aid: {aid} / det_aid type: {type(det_aid)}, aid type: {type(aid)}"
        det_img_coco = np.array(det_data['keypoints']).reshape(-1, 3)
        joint_img = self.add_pelvis_and_neck(det_img_coco)
        # vis_2d_joints(gt_joint_img, img_path, self.coco_skeleton, prefix=f"{img_path.split('/')[-1]}")

        bbox = get_bbox(joint_img)
        bbox1 = process_bbox(bbox.copy(), aspect_ratio=1.0, scale=1.25)
        bbox2 = process_bbox(bbox.copy())
        proj_target_joint_img, trans = j2d_processing(joint_img.copy(), (self.img_res, self.img_res), bbox1, rot, flip, None)

        joint_img, _ = j2d_processing(joint_img.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]), bbox2, rot, flip, None)

        #  -> 0~1
        joint_img = joint_img[:, :2]
        joint_img /= np.array([[cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]]])

        # normalize loc&scale
        mean, std = np.mean(joint_img, axis=0), np.std(joint_img, axis=0)
        joint_img = (joint_img.copy() - mean) / std

        return joint_img[:, :2], proj_target_joint_img[:, :2], img_path, trans
