import numpy as np
import torch
from torch.nn import functional as F
from core.config import cfg


def get_center_scale(box_info):
        x, y, w, h = box_info

        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        scale = np.array([
            w * 1.0, h * 1.0
        ], dtype=np.float32)

        return center, scale


def get_bbox(joint_img):
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    xmin = min(x_img);
    ymin = min(y_img);
    xmax = max(x_img);
    ymax = max(y_img);

    x_center = (xmin + xmax) / 2.;
    width = xmax - xmin;
    xmin = x_center - 0.5 * width #* 1.2
    xmax = x_center + 0.5 * width #* 1.2

    y_center = (ymin + ymax) / 2.;
    height = ymax - ymin;
    ymin = y_center - 0.5 * height #* 1.2
    ymax = y_center + 0.5 * height #* 1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def process_bbox(bbox, aspect_ratio=None, scale=1.0):
    # sanitize bboxes
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x+(w-1), y+(h-1)
    if w*h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    if aspect_ratio is None:
        aspect_ratio = cfg.MODEL.input_shape[1] / cfg.MODEL.input_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * scale #*1.25
    bbox[3] = h * scale #*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    return bbox


def get_local_pose_trans(joints, kps_line):
    local_pose_trans = np.zeros((len(joints), 3))

    for l in range(len(kps_line)):
        parent = kps_line[l][0]
        child = kps_line[l][1]

        local_pose_trans[child] = joints[child] - joints[parent]

    return local_pose_trans


def make_skeleton_dict(kps_line, num_joints):  
    num_skeleton = len(kps_line)
    skeleton_dict = [{'child_id': []} for _ in range(num_joints)]

    for l in range(num_skeleton):
        parent = kps_line[l][0]
        child = kps_line[l][1]

        skeleton_dict[parent]['child_id'].append(child)

    return skeleton_dict


def forward_kinematics(skeleton, cur_joint_idx, local_pose, global_pose):
    child_id = skeleton[cur_joint_idx]['child_id']
    if len(child_id) == 0:
        return

    for joint_id in child_id:
        global_pose[joint_id] = torch.matmul(global_pose[cur_joint_idx], local_pose[joint_id])
        forward_kinematics(skeleton, joint_id, local_pose, global_pose)


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2]) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2]) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord


def pixel2cam(coords, c, f):
    cam_coord = np.zeros((len(coords), 3))
    z = coords[..., 2].reshape(-1, 1)

    cam_coord[..., :2] = (coords[..., :2] - c) * z / f
    cam_coord[..., 2] = coords[..., 2]

    return cam_coord


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s)

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2


def euler2mat(theta, to_4x4=False):
    assert theta.shape[-1] == 3

    original_shape = list(theta.shape)
    original_shape.append(3)

    theta = theta.view(-1, 3)
    theta_x = theta[:, 0:1]
    theta_y = theta[:, 1:2]
    theta_z = theta[:, 2:3]

    R_x = torch.cat([ \
        torch.stack([torch.ones_like(theta_x), torch.zeros_like(theta_x), torch.zeros_like(theta_x)], 2), \
        torch.stack([torch.zeros_like(theta_x), torch.cos(theta_x), -torch.sin(theta_x)], 2), \
        torch.stack([torch.zeros_like(theta_x), torch.sin(theta_x), torch.cos(theta_x)], 2) \
        ], 1)

    R_y = torch.cat([ \
        torch.stack([torch.cos(theta_y), torch.zeros_like(theta_y), torch.sin(theta_y)], 2), \
        torch.stack([torch.zeros_like(theta_y), torch.ones_like(theta_y), torch.zeros_like(theta_y)], 2), \
        torch.stack([-torch.sin(theta_y), torch.zeros_like(theta_y), torch.cos(theta_y)], 2), \
        ], 1)

    R_z = torch.cat([ \
        torch.stack([torch.cos(theta_z), -torch.sin(theta_z), torch.zeros_like(theta_z)], 2), \
        torch.stack([torch.sin(theta_z), torch.cos(theta_z), torch.zeros_like(theta_z)], 2), \
        torch.stack([torch.zeros_like(theta_z), torch.zeros_like(theta_z), torch.ones_like(theta_z)], 2), \
        ], 1)

    R = torch.bmm(R_z, torch.bmm(R_y, R_x))

    if to_4x4:
        batch_size = R.shape[0]
        R = torch.cat([R, torch.zeros((batch_size, 3, 1)).cuda().float()], 2)
        R = torch.cat([R, torch.cuda.FloatTensor([0, 0, 0, 1])[None, None, :].repeat(batch_size, 1, 1)], 1)  # 0001
        original_shape[-2] = 4;
        original_shape[-1] = 4

    R = R.view(original_shape)
    return R


def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)






