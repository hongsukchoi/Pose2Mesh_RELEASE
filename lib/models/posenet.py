import torch.nn as nn
from core.config import cfg as cfg
from funcs_utils import load_checkpoint


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.batch_norm1(x)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w1(y)

        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w2(y)

        out = x + y

        return out


class LinearModel(nn.Module):
    def __init__(self,
                 num_joint,
                 linear_size=4096,
                 num_stage=2,
                 p_dropout=0.5,
                 pretrained=False):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size =  num_joint * 2
        # 3d joints
        self.output_size = num_joint * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        # pre-processing
        y = self.w1(x)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        return y

    def _load_pretrained_model(self):
        print("Loading pretrained posenet...")
        checkpoint = load_checkpoint(load_dir=cfg.MODEL.posenet_path, pick_best=True)
        self.load_state_dict(checkpoint['model_state_dict'])


def get_model(num_joint, hid_dim, num_layer, p_dropout, pretrained=False):
    model = LinearModel(num_joint, hid_dim, num_layer, p_dropout, pretrained)

    return model
