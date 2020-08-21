import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
import graph_utils
from models.backbones.cheby_graph_conv import graph_conv_cheby


class Pose2Mesh(nn.Module):
    def __init__(self, num_joint_input_chan, num_mesh_output_chan, graph_L):
        super(Pose2Mesh, self).__init__()

        self.num_joint_input_chan = num_joint_input_chan
        self.num_mesh_output_chan = num_mesh_output_chan
        self.graph_L = graph_L

        # parameters
        # each block outputs same feature channels
        if cfg.DATASET.target_joint_set == 'mano':
            # each block outputs same feature channels
            self.CL_K = [3, 3, 3, 3, 3, 3, 3]  # kernel size between features
            self.CL_F = [(num_joint_input_chan, 32, 64, 64),
                         (64, 128, 256), (256, 256, 256), (256, 256, 256),
                         (256, 256, 256), (256, 128, 128),
                         (128, 64, num_mesh_output_chan)]  # 21 -> 68 -> 136 -> 272 -> 544 -> 1088 (778) -> 1088 (778)
        else:
            self.CL_K = [3, 3,3,3,  3,3,3, 3,3,3]  # kernel size between features
            self.CL_F = [(num_joint_input_chan, 32, 64, 64),
                        (64, 128, 256), (256, 256, 256), (256, 256, 256),
                        (256, 256, 256), (256, 256, 256), (256, 128, 128),
                        (128, 128, 128), (128, 128, 128), (128, 64, num_mesh_output_chan)]

        del self.graph_L[-2]  # remove 48*48 Lap matrix
        self.fc = nn.Linear(self.graph_L[-1].shape[0] * (self.CL_F[0][-1]),
                            self.graph_L[-2].shape[0] * self.CL_F[1][0])  # upsample layer: 24 -> 96

        _cl = []
        _bn = []
        for i in range(len(self.CL_F)):
            for layer_i in range(len(self.CL_F[i]) - 1):
                Fin = self.CL_K[i] * self.CL_F[i][layer_i]
                Fout = self.CL_F[i][layer_i + 1]

                _cl.append(nn.Linear(Fin, Fout))

                scale = np.sqrt(2.0 / (Fin + Fout))
                _cl[-1].weight.data.uniform_(-scale, scale)
                _cl[-1].bias.data.fill_(0.0)

                if i == len(self.CL_F) - 1 and layer_i == len(self.CL_F[i]) - 2:  # 16 -> num_mesh_output_chan
                    _bn.append(None)
                else:
                    _bn.append(nn.BatchNorm1d(Fout))

        self.cl = nn.ModuleList(_cl)
        self.bn = nn.ModuleList(_bn)

        # convert scipy sparse matric L to pytorch
        for graph_i in range(len(graph_L)):
            self.graph_L[graph_i] = graph_utils.sparse_python_to_torch(self.graph_L[graph_i])

    def init_weights(self, W, Fin, Fout):
        scale = np.sqrt(2.0 / (Fin + Fout))
        W.uniform_(-scale, scale)

        return W

    # Upsampling of size p.
    def graph_upsample(self, x, p):
        if p > 1:
            x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
            x = nn.Upsample(scale_factor=p)(x)  # B x F x (V*p)
            x = x.permute(0, 2, 1).contiguous()  # x = B x (V*p) x F
            return x
        else:
            return x

    def forward(self, x):
        self.graph_L = [self.graph_L[graph_i].cuda() for graph_i in range(len(self.graph_L))]
        x = x.view(-1, self.graph_L[-1].shape[0], self.num_joint_input_chan)
        # x: B x num_joints x 3

        cl_i = 0
        for i in range(len(self.CL_F)):
            input_x = x

            for layer_i in range(len(self.CL_F[i]) - 1):
                Fout = self.CL_F[i][layer_i + 1]

                ldx = -(i + 1)
                if i == len(self.CL_F) -1:
                    ldx += 1

                x = graph_conv_cheby(x, self.cl[cl_i], self.bn[cl_i], self.graph_L[ldx],
                                          Fout, self.CL_K[i])

                if i != len(self.CL_F) - 1 or layer_i != len(self.CL_F[i]) - 2:  # 16 -> num_mesh_output_chan
                    x = F.relu(x)

                cl_i += 1

            if i == 0:  # upsample layer: num_joints -> 96
                x = self.fc(x.view(-1, self.graph_L[-1].shape[0] * self.CL_F[0][-1]))
                x = x.view(-1, self.graph_L[-2].shape[0], self.CL_F[1][0])

            elif i < len(self.CL_F) - 2:
                input_x = nn.functional.interpolate(input_x, size=x.shape[2], mode='linear')
                x = input_x + x
                x = self.graph_upsample(x, 2)

            elif i == len(self.CL_F) - 2:
                input_x = nn.functional.interpolate(input_x, size=x.shape[2], mode='linear')
                x = input_x + x

        return x  # x: B x 12288 x 3


def get_model(num_joint_input_chan, num_mesh_output_chan, graph_L):
    model = Pose2Mesh(num_joint_input_chan, num_mesh_output_chan, graph_L)

    return model
