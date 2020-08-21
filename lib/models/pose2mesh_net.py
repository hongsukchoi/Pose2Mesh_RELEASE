import torch
import torch.nn as nn

from core.config import cfg as cfg
from models import meshnet, posenet


class FlatPose2Mesh(nn.Module):
    def __init__(self, num_joint, graph_L):
        super(FlatPose2Mesh, self).__init__()

        self.num_joint = num_joint
        self.pose_lifter = posenet.get_model(num_joint, hid_dim=4096, num_layer=2, p_dropout=0.5, pretrained=cfg.MODEL.posenet_pretrained)
        self.pose2mesh = meshnet.get_model(num_joint_input_chan=2 + 3, num_mesh_output_chan=3, graph_L=graph_L)

    def forward(self, pose2d):
        pose3d = self.pose_lifter(pose2d.view(len(pose2d), -1))
        pose3d = pose3d.reshape(-1, self.num_joint, 3)
        pose_combine = torch.cat((pose2d, pose3d.detach() / 1000), dim=2)
        cam_mesh = self.pose2mesh(pose_combine)

        return cam_mesh, pose3d


def get_model(num_joint, graph_L):
    model = FlatPose2Mesh(num_joint, graph_L)

    return model


