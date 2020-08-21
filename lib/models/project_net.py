import torch
import torch.nn as nn

from core.config import cfg


class OptimzeCamLayer(nn.Module):
    def __init__(self):
        super(OptimzeCamLayer, self).__init__()

        self.img_res = 500 / 2
        self.cam_param = nn.Parameter(torch.rand((1,3)))

    def forward(self, pose3d):
        output = pose3d[:, :, :2] + self.cam_param[None, :, 1:]
        output = output * self.cam_param[None, :, :1] * self.img_res + self.img_res
        return output


def get_model():
    model = OptimzeCamLayer()

    return model


