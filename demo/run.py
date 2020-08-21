import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np

import __init_path
import models
from core.config import cfg
from aug_utils import j2d_processing
from coord_utils import get_bbox, process_bbox
from funcs_utils import load_checkpoint, save_obj
from graph_utils import build_coarse_graphs
from renderer import Renderer
from vis import vis_2d_keypoints
from _mano import MANO
from smpl import SMPL


def render(result, out_path, focal, img_res, mesh_face, skeleton):
    renderer = Renderer(focal_length=focal, img_res=img_res, faces=mesh_face)

    pred_vertices, pred2d_pose, cam_param, trans = \
        result['mesh'], result['pred2d_pose'], result['cam_param'], result['trans']

    camera_translation = np.array([cam_param[1], cam_param[2], 2 * focal / (img_res * cam_param[0] + 1e-9)])

    # img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)[:, :, ::-1]
    # img = cv2.warpAffine(img, trans, (img_res, img_res), flags=cv2.INTER_LINEAR)
    img = np.zeros((img_res, img_res, 3))  # dummy

    img_shape = 255 * renderer(pred_vertices, camera_translation, img)
    tmpimg = img.copy().astype(np.uint8)
    tmpkps = np.zeros((3, len(pred2d_pose)))
    tmpkps[0, :], tmpkps[1, :], tmpkps[2, :] = pred2d_pose[:, 0], pred2d_pose[:, 1], 1
    img_2dpose = vis_2d_keypoints(tmpimg, tmpkps, skeleton)

    # Save reconstructions
    cv2.imwrite(out_path + 'demo_mesh.png', img_shape[:, :, ::-1])
    cv2.imwrite(out_path + 'demo_2Dpose.png', img_2dpose[:, :, ::-1])


def get_joint_setting(mesh_model, joint_category='human36'):
    joint_regressor, joint_num, skeleton, graph_L, graph_perm_reverse = None, None, None, None, None
    if joint_category == 'human36':
        joint_regressor = mesh_model.joint_regressor_h36m
        joint_num = 17
        skeleton = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
        (2, 3), (0, 4), (4, 5), (5, 6))
        flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        graph_Adj, graph_L, graph_perm,graph_perm_reverse = \
            build_coarse_graphs(mesh_model.face, joint_num, skeleton, flip_pairs, levels=9)
        model_chk_path = './experiment/exp_07-07_23:02:27.03/checkpoint'

    elif joint_category == 'coco':
        joint_regressor = mesh_model.joint_regressor_coco
        joint_num = 19  # add pelvis and neck
        skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15),  # (5, 6), #(11, 12),
            (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 0))
        flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        graph_Adj, graph_L, graph_perm, graph_perm_reverse = \
            build_coarse_graphs(mesh_model.face, joint_num, skeleton, flip_pairs, levels=9)
        model_chk_path = './experiment/exp_07-15_14:50/checkpoint'

    elif joint_category == 'smpl':
        joint_regressor = mesh_model.layer['neutral'].th_J_regressor.numpy().astype(np.float32)
        joint_num = 24
        skeleton = (
            (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
            (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15))
        flip_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
        graph_Adj, graph_L, graph_perm, graph_perm_reverse = \
            build_coarse_graphs(mesh_model.face, joint_num, skeleton, flip_pairs, levels=9)
        model_chk_path = './experiment/exp_08-18_21:09/checkpoint'

    elif joint_category == 'mano':
        joint_regressor = mesh_model.joint_regressor
        joint_num = 21
        skeleton = ( (0,1), (0,5), (0,9), (0,13), (0,17), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16), (17,18), (18,19), (19,20) )
        hori_conn = (
        (1, 5), (5, 9), (9, 13), (13, 17), (2, 6), (6, 10), (10, 14), (14, 18), (3, 7), (7, 11), (11, 15), (15, 19),
        (4, 8), (8, 12), (12, 16), (16, 20))
        graph_Adj, graph_L, graph_perm, graph_perm_reverse = \
            build_coarse_graphs(mesh_model.face, joint_num, skeleton, hori_conn, levels=6)
        model_chk_path = './experiment//checkpoint'

    else:
        raise NotImplementedError(f"{joint_category}: unknown joint set category")

    model = models.pose2mesh_net.get_model(joint_num, graph_L)
    checkpoint = load_checkpoint(load_dir=model_chk_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, joint_regressor, joint_num, skeleton, graph_L, graph_perm_reverse


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Render Pose2Mesh output')
    parser.add_argument('--gpu', type=str, default='0', help='assign gpu number')
    parser.add_argument('--res', type=int, default=500, help='assign image resolution for visualization of mesh projection')
    parser.add_argument('--input', type=str, default='input.npy', help='path of input 2D pose')
    parser.add_argument('--joint_set', type=str, default='human36', help='select the topology of input 2D pose from [human36, coco, smpl, mano]')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    FOCAL_LENGTH = 1500
    IMG_RES = args.res

    joint_set = args.joint_set
    input_path = args.input
    output_path = './demo/result/'

    # fix config
    cfg.DATASET.target_joint_set = joint_set
    cfg.MODEL.posenet_pretrained = False

    # get model and input
    if joint_set == 'mano':
        mesh_model = MANO()
    else:
        mesh_model = SMPL()
    model, joint_regressor, joint_num, skeleton, graph_L, graph_perm_reverse = get_joint_setting(mesh_model, joint_category=joint_set)
    model = model.cuda()
    project_net = models.project_net.get_model().cuda()
    joint_regressor = torch.Tensor(joint_regressor).cuda()
    joint_input = np.load(input_path)

    # pre-process input
    bbox = get_bbox(joint_input)
    bbox1 = process_bbox(bbox.copy(), aspect_ratio=1.0, scale=1.25)
    bbox2 = process_bbox(bbox.copy())
    proj_target_joint_img, trans = j2d_processing(joint_input.copy(), (IMG_RES, IMG_RES), bbox1, 0, 0, None)
    joint_img, _ = j2d_processing(joint_input.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]), bbox2, 0, 0, None)

    joint_img = joint_img[:, :2]
    joint_img /= np.array([[cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]]])
    mean, std = np.mean(joint_img, axis=0), np.std(joint_img, axis=0)
    joint_img = (joint_img.copy() - mean) / std
    joint_img = torch.Tensor(joint_img[None, :, :]).cuda()
    target_joint = torch.Tensor(proj_target_joint_img[None, :, :2]).cuda()

    # get optimization settings for projection
    criterion = nn.L1Loss()
    optimizer = optim.Adam(project_net.parameters(), lr=0.1)

    model.eval()
    project_net.train()
    # estimate mesh, pose
    pred_mesh, _ = model(joint_img)
    pred_mesh = pred_mesh[:, graph_perm_reverse[:mesh_model.face.max() + 1], :]
    pred_3d_joint = torch.matmul(joint_regressor, pred_mesh)

    out = {}
    # assume batch=1
    for j in range(0, 1500):
        # projection
        pred_2d_joint = project_net(pred_3d_joint.detach())

        loss = criterion(pred_2d_joint, target_joint[:, :17, :])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if j == 500:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.05
        if j == 1000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001

    out['mesh'] = pred_mesh[0].detach().cpu().numpy()
    out['cam_param'] = project_net.cam_param[0].detach().cpu().numpy()
    out['trans'] = trans
    out['pred2d_pose'] = proj_target_joint_img

    render(out, output_path, FOCAL_LENGTH, IMG_RES, mesh_model.face, skeleton)
    save_obj(out['mesh'], mesh_model.face, osp.join(output_path,'demo_output.obj'))

