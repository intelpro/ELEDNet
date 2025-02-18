import torch
import os
import datetime
import argparse
from collections import OrderedDict
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from math import ceil
from utils.utils import *
from utils.dataloader import get_train_dataset, get_test_dataset
from models.model_manager import ModelManager


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_batch_size', type = int, default=1)
    parser.add_argument('--batch_size', type = int, default=1)
    parser.add_argument('--total_epochs', type = int, default=201)
    # training params
    parser.add_argument('--num_train_video_frames', type = int, default=3)
    parser.add_argument('--num_test_video_frames', type = int, default=3)
    parser.add_argument('--voxel_num_bins', type = int, default=16)
    parser.add_argument('--num_video_frames', type = int, default=1)
    parser.add_argument('--crop_size', type = int, default=256)
    parser.add_argument('--learning_rate', type = float, default=1e-4)
    parser.add_argument('--mode', type = str, default='train')
    # model discription
    parser.add_argument('--model_folder', type=str, default='model_factory')
    parser.add_argument('--model_name', type=str, default='models_final')
    # data loading params
    parser.add_argument('--num_threads', type = int, default=12)
    parser.add_argument('--test_epoch_every', type=int, default=40)
    parser.add_argument('--experiment_name', type = str, default='train_networks')
    parser.add_argument('--loss_type', type = str, default='multi_scale')
    parser.add_argument('--loss_name', type = str, default='multi_scale_single')
    parser.add_argument('--tb_update_thresh', type = int, default=200)
    parser.add_argument('--data_dir', type = str, default = '/media/mnt2/dataset/RELED/')
    parser.add_argument('--use_multigpu', type=str2bool, default='True')
    parser.add_argument('--resume_ckpt', type=str2bool, default=False)
    parser.add_argument('--ckpt_dir', type = str, default='./experiments/240302-train_networks_v27_ms_hpf_dp_SA_v4_reduced_ffn_mimo_dffn/0056/saved_model/')
    args = parser.parse_args()
    return args


class Trainer:
    def __init__(self, args):
        self.args = args
        self.tb_iter_cnt = 0
        self.tb_iter_cnt_val = 0
        self.tb_iter_thresh = args.tb_update_thresh
        tb_path = f'./experiments/{datetime.datetime.now().strftime("%y%m%d-" + args.experiment_name + "/%H%M")} '
        self.tb = SummaryWriter(tb_path, flush_secs=1)

        self.train_loader = DataLoader(get_train_dataset(args, mode='train'),
                                       batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_threads, pin_memory=True, drop_last=True)
        
        self.test_loader = DataLoader(get_test_dataset(args, mode='test'),
                                      batch_size=args.val_batch_size, shuffle=False,
                                      num_workers=args.num_threads, pin_memory=False)
        
        self.model = ModelManager(args)
        self.model.initilalize_deblur_model(args, model_folder=args.model_folder, model_name=args.model_name, tb_path=tb_path)
        self._setup_device()
        self.optimizer = Adam(self.model.get_deblurnet_optimizer_params(), lr=args.learning_rate)
        self.start_epoch = 0
        self.end_epoch = args.total_epochs
        self.PSNR_calculator = PSNR()
        self.SSIM_calculator = SSIM()
        self.logger = get_logger(tb_path, 'log.txt', 'append')
        self._log_arguments()

    def _setup_device(self):
        if torch.cuda.is_available():
            self.model.cuda_deblur()
        if self.args.use_multigpu:
            self.model.use_multi_gpu_deblur()

    def _log_arguments(self):
        self.logger.info(f'Overall parameter count: {self.model.count_total_parameters() * 1e-6:.4f} MB')
        for arg, val in vars(self.args).items():
            self.logger.info(f'{arg}: {val}')
        self.tb.add_text('logs', f'model name: {self.args.model_name}')

    def train(self):
        for self.epoch in trange(self.start_epoch, self.end_epoch, desc='Epoch Progress'):
            for sample in tqdm(self.train_loader, desc='Training Progress'):
                self.optimizer.zero_grad()
                sample = batch2device(sample)
                self.model.set_video_inputs(sample)
                self.model.forward_deblur_net()
                loss = self.model.get_single_loss()
                loss.backward()
                self.optimizer.step()
                self.model.update_loss_meters_deblur()
                self.tb_iter_cnt += 1
                if self.args.batch_size * self.tb_iter_cnt > self.tb_iter_thresh:
                    self.log_train_tb()
            if self.epoch % 20 == 0:
                self.test(self.epoch)
                self.save_model(self.epoch)

    def test(self, epoch):
        psnr_meter, ssim_meter = AverageMeter(), AverageMeter()
        self.model.del_batch()
        with torch.no_grad():
            for sample in tqdm(self.test_loader, desc='Testing Progress'):
                sample = batch2device(sample)
                self.model.set_video_inputs(sample)
                self.model.forward_deblur_net()
                psnr_meter.update(self.PSNR_calculator(self.model.batch['clean_middle'], self.model.batch['output_deblur'][0]).mean().item())
                ssim_meter.update(self.SSIM_calculator(self.model.batch['clean_middle'], self.model.batch['output_deblur'][0]).mean().item())
            self.tb.add_scalar('val_progress/avg_psnr/', psnr_meter.avg, epoch)
            self.tb.add_scalar('val_progress/avg_ssim/', ssim_meter.avg, epoch)
        self.model.del_batch()
        torch.cuda.empty_cache()
        return psnr_meter.avg

    def log_train_tb(self):
        self.tb.add_scalar('train_progress/loss_total', self.model.loss_total_meter.avg, self.tb_iter_cnt)
        self.tb.add_image('train_blur/input', self.model.batch['blur_input'][0], self.tb_iter_cnt)
        self.tb.add_image('train_output/clean_est', self.model.batch['output_deblur'][0], self.tb_iter_cnt)
        self.tb.add_image('train_output/clean_gt', self.model.batch['clean_middle'][0], self.tb_iter_cnt)
        self.tb_iter_cnt = 0
        self.model.reset_loss_meters_deblur()

    def save_model(self, epoch):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.deblur_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        save_path = os.path.join(self.model.save_path, f'model_{epoch}_ep.pth')
        torch.save(state, save_path)


if __name__ == '__main__':
    args = get_argument()
    trainer = Trainer(args)
    if args.mode == 'train':
        trainer.train()
    else:
        trainer.test(0)
