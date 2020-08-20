import os
import argparse
import torch
import __init_path
import shutil

from funcs_utils import save_checkpoint, save_plot, check_data_pararell, count_parameters
from core.config import config, update_config

parser = argparse.ArgumentParser(description='Train Pose2Mesh')

parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--cpu', action='store_true', help='use cpu')
parser.add_argument('--resume_training', action='store_true', help='Resume Training')
parser.add_argument('--debug', action='store_true', help='reduce dataset items')
parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat')
parser.add_argument('--cfg', type=str, help='experiment configure file name')


args = parser.parse_args()
if args.cfg:
    update_config(args.cfg)
torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device(f"cpu" if args.cpu else "cuda")
print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

from core.base import Trainer, Tester


trainer = Trainer(args, load_dir='./experiment/exp_08-17_12:45/checkpoint')
tester = Tester(args)  # if not args.debug else None

print("===> Start training...")
for epoch in range(config.TRAIN.begin_epoch, config.TRAIN.end_epoch+1):
    trainer.train(epoch)
    trainer.lr_scheduler.step()

    tester.test(epoch, current_model=trainer.model)

    if epoch > 1:
        is_best = tester.joint_error < min(trainer.error_history['joint'])
    else:
        is_best = None

    trainer.error_history['surface'].append(tester.surface_error)
    trainer.error_history['joint'].append(tester.joint_error)

    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': check_data_pararell(trainer.model.state_dict()),
        'optim_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
        'train_log': trainer.loss_history,
        'test_log': trainer.error_history
    }, epoch, is_best)

    save_plot(trainer.loss_history, epoch)
    save_plot(trainer.error_history['surface'], epoch, title='Surface Error')
    save_plot(trainer.error_history['joint'], epoch, title='Joint Error')






