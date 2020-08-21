import numpy as np
import cv2
import torch
import random

from coord_utils import get_center_scale
from core.config import cfg


def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def flip_2d_joint(kp, width, flip_pairs):
    kp[:, 0] = width - kp[:, 0] - 1
    """Flip keypoints."""
    for lr in flip_pairs:
        kp[lr[0]], kp[lr[1]] = kp[lr[1]].copy(), kp[lr[0]].copy()

    return kp


def flip_3d_joint(kp, flip_pairs):
    """Flip keypoints."""
    for lr in flip_pairs:
        kp[lr[0]], kp[lr[1]] = kp[lr[1]].copy(), kp[lr[0]].copy()

    kp[:,0] = - kp[:,0]
    return kp


def j2d_processing(kp, res, bbox, rot, f, flip_pairs):
    """Process gt 2D keypoints and apply all augmentation transforms."""
    # flip the x coordinates
    center, scale = get_center_scale(bbox)
    trans = get_affine_transform(center, scale, rot, res)

    nparts = kp.shape[0]
    for i in range(nparts):
        kp[i, :2] = affine_transform(kp[i, :2].copy(), trans)

    if f:
        kp = flip_2d_joint(kp, res[0], flip_pairs)
    kp = kp.astype('float32')
    return kp, trans


def j3d_processing(S, r, f, flip_pairs):
    """Process gt 3D keypoints and apply all augmentation transforms."""
    # in-plane rotation
    rot_mat = np.eye(3)
    if not r == 0:
        rot_rad = -r * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
    S = np.einsum('ij,kj->ki', rot_mat, S)

    # flip the x coordinates
    if f:
        S = flip_3d_joint(S, flip_pairs)
    S = S.astype('float32')

    return S


def my3d_processing(S, r):
    rot_mat = np.eye(3)
    if not r == 0:
        rot_rad = -r * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, [0,2]] = [cs, sn]
        rot_mat[2, [0,2]] = [-sn, cs]
    S = np.einsum('ij,kj->ki', rot_mat, S)

    return S


def augm_params(is_train):
    if not is_train:
        return 0, 0

    """Get augmentation parameters."""
    flip = 0  # flipping
    rot = 0  # rotation
    # We flip with probability 1/2
    if cfg.AUG.flip and random.uniform(0, 1) <= 0.5:
        flip = 1

    rot_factor = cfg.AUG.rotate_factor
    # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
    rot = min(2 * rot_factor,
              max(-2 * rot_factor, np.random.randn() * rot_factor))

    if random.uniform(0, 1) <= 0.5:
        rot = 0

    return flip, rot



def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def flip_back(output_flipped, matched_pairs):

    output_flipped = output_flipped[:, :, :, ::-1]
    for pair in matched_pairs:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

        return output_flipped


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


# rotate src_point by rot_Rad
def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def flip_joints(joints, joints_vis, width, matched_parts):
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints, joints_vis


def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)


def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
                            dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    assert flipped.device == tensor.device
    assert flipped.requires_grad == tensor.requires_grad
    return flipped
