import os.path as osp
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from core.config import cfg


def vis_coco_skeleton(img, kps, kps_lines, given_color, alpha=1):
    colors = [
            # face
            (255/255, 153/255, 51/255),
            (255/255, 153/255, 51/255),
            (255/255, 153/255, 51/255),
            (255/255, 153/255, 51/255),

            # left arm
            (102/255, 255/255, 102/255),
            (51/255, 255/255, 51/255),

            # right leg
            (255 / 255, 102 / 255, 255 / 255),
            (255 / 255, 51 / 255, 255 / 255),


            # left leg

            (255 / 255, 102 / 255, 102 / 255),
            (255 / 255, 51 / 255, 51 / 255),

            # shoulder-thorax, hip-pevlis,
            (153/255, 255/255, 153/255), # l shoulder - thorax
            (153/255, 204/255, 255/255), # r shoulder - thorax
            (255/255, 153/255, 153/255), # l hip - pelvis
            (255/255, 153/255, 255/255), # r hip -pelvis

            # center body line
            (255/255, 204/255, 153/255),
            (255/255, 178/255, 102/255),

            # right arm
            (102 / 255, 178 / 255, 255 / 255),
            (51 / 255, 153 / 255, 255 / 255),
            ]

    colors = [[c[2]*255,c[1]*255,c[0]*255] for c in colors]
    given_color = [given_color[0]*255, given_color[1]*255, given_color[2]*255]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    line_thick = 2 #13
    circle_rad = 2 #10
    circle_thick = 3 #7

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        cv2.line(
            kp_mask, p1, p2,
            color=given_color, thickness=line_thick, lineType=cv2.LINE_AA)
        cv2.circle(
            kp_mask, p1,
            radius=circle_rad, color=given_color, thickness=circle_thick, lineType=cv2.LINE_AA)
        cv2.circle(
            kp_mask, p2,
            radius=circle_rad, color=given_color, thickness=circle_thick, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_2d_keypoints(img, kps, kps_line, bbox=None, kp_thre=0.4, alpha=1):
    # Convert form plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_line))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perfrom the drawing on a copy of the image, to allow for blending
    kp_mask = np.copy(img)

    # Draw bounding box
    if bbox is not None:
        b1 = bbox[0, 0].astype(np.int32), bbox[0, 1].astype(np.int32)
        b2 = bbox[1, 0].astype(np.int32), bbox[1, 1].astype(np.int32)
        b3 = bbox[2, 0].astype(np.int32), bbox[2, 1].astype(np.int32)
        b4 = bbox[3, 0].astype(np.int32), bbox[3, 1].astype(np.int32)

        cv2.line(kp_mask, b1, b2, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(kp_mask, b2, b3, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(kp_mask, b3, b4, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(kp_mask, b4, b1, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    # Draw the keypoints
    for l in range(len(kps_line)):
        i1 = kps_line[l][0]
        i2 = kps_line[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thre and kps[2, i2] > kp_thre:
            cv2.line(
                kp_mask, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thre:
            cv2.circle(
                kp_mask, p1, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thre:
            cv2.circle(kp_mask, p2, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_2d_pose(pred, img, kps_line, prefix='vis2dpose', bbox=None):
    if type(img) == str:
        img = cv2.imread(img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    tmpimg = img.copy().astype(np.uint8)
    tmpkps = np.zeros((3, len(pred)))
    tmpkps[0, :], tmpkps[1, :] = pred[:, 0], pred[:, 1]
    tmpkps[2, :] = 1
    tmpimg = vis_2d_keypoints(tmpimg, tmpkps, kps_line, bbox)

    now = datetime.now()
    file_name = f'{prefix}_{now.isoformat()[:-7]}_2d_joint.jpg'
    cv2.imwrite(osp.join(cfg.vis_dir, file_name), tmpimg)
    cv2.imshow(prefix, tmpimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def vis_3d_pose(kps_3d, kps_line, joint_set_name='', prefix='vis3dpose', gt=False, ax_in=None):
    if joint_set_name == 'human36':
        r_joints = [1, 2, 3, 14, 15, 16]
    elif joint_set_name == 'coco':
        r_joints = [2, 4, 6, 8, 10, 12, 14, 16]
    elif joint_set_name == 'smpl':
        r_joints = [2, 5, 8, 11, 14, 17, 19, 21, 23]
    else:
        r_joints = []

    kps_3d_vis = np.ones((len(kps_3d), 1))
    if not ax_in:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', projection='3d')
    else:
        ax = ax_in

    for l in range(len(kps_line)):
        i1 = kps_line[l][0]
        i2 = kps_line[l][1]
        x = np.array([kps_3d[i1, 0], kps_3d[i2, 0]])
        y = np.array([kps_3d[i1, 1], kps_3d[i2, 1]])
        z = np.array([kps_3d[i1, 2], kps_3d[i2, 2]])

        if kps_3d_vis[i1, 0] > 0 and kps_3d_vis[i2, 0] > 0:
            ax.plot(x, z, -y, c='r', linewidth=1)
        if kps_3d_vis[i1, 0] > 0:
            c = 'g' if i1 in r_joints else 'b'
            ax.scatter(kps_3d[i1, 0], kps_3d[i1, 2], -kps_3d[i1, 1], c=c, marker='o')
        if kps_3d_vis[i2, 0] > 0:
            c = 'g' if i2 in r_joints else 'b'
            ax.scatter(kps_3d[i2, 0], kps_3d[i2, 2], -kps_3d[i2, 1], c=c, marker='o')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    ax.set_zlabel('Y axis')

    title = f'3D Ground Truth' if gt else f' 3D Prediction'
    ax.set_title(title)
    ax.legend()
    axisEqual3D(ax)

    if not ax_in:
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        now = datetime.now()
        file_name = f'{prefix}_{now.isoformat()[:-7]}_{"3d_gt" if gt else "3d_pred"}.jpg'
        fig.savefig(osp.join(cfg.vis_dir, file_name))
        plt.close(fig=fig)
    else:
        return ax

