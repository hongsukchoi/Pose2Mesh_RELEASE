import os
import argparse
import torch
import shutil
import __init_path

from core.config import update_config, config


parser = argparse.ArgumentParser(description='Test Pose2Mesh')

parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--cpu', action='store_true', help='use cpu')
parser.add_argument('--cfg', type=str, help='experiment configure file name')
parser.add_argument('--debug', action='store_true', help='reduce dataset items')
parser.add_argument('--gpu', type=str, default='0,', help='assign multi-gpus by comma concat')

args = parser.parse_args()
if args.cfg:
    update_config(args.cfg)
torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device(f"cpu" if args.cpu else "cuda")
print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

from core.base import Tester


print("===> Start testing...")
tester = Tester(args, load_dir=config.TEST.weight_path)
tester.test(0)

