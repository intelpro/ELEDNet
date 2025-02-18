import torch.nn as nn
import torch
import os
from math import ceil
import importlib
from utils.utils import AverageMeter
import torch.nn.functional as F


class ModelManager(object):
    def __init__(self, args):
        self.batch = {}
        self.args = args
        self.voxel_num_bins = args.voxel_num_bins
        self.smoothness_weight = 1.0
        self.scale = 3 if getattr(args, 'loss_type', None) == 'multi_scale' else 1
        self.ms_lambda_dict = [1.0, 0.5, 0.25] if self.scale == 3 else []
        self.downsample = nn.AvgPool2d(2, stride=2) if self.scale == 3 else None
    
    def initilalize_deblur_model(self, args, model_folder, model_name, tb_path):
        mod = importlib.import_module('models.' + model_folder + '.' + model_name)
        self.deblur_net = mod.EventDeblurNet()
        self.save_path = os.path.join(tb_path, 'saved_model') if tb_path else None
        os.makedirs(self.save_path, exist_ok=True) if self.save_path else None
        self.loss_total_meter = AverageMeter()
        self.loss_deblur_meter = AverageMeter()
    
    def cuda_deblur(self):
        self.deblur_net.cuda()
    
    def use_multi_gpu_deblur(self): # data parallel
        self.deblur_net = nn.DataParallel(self.deblur_net)
    
    def count_total_parameters(self):
        return sum(p.numel() for p in self.deblur_net.parameters())
    
    def get_deblurnet_optimizer_params(self):
        return self.deblur_net.parameters()
    
    def get_l1_loss(self, x, y, reduction_):
        loss = F.l1_loss(x, y, reduction=reduction_)
        return loss

    def get_chainbor_loss(self, x, y):
        loss = ((((x - y) ** 2 + 1e-6) ** 0.5).mean())
        return loss

    def backward_warp(self, x, flo):
        '''
		x shape : [B,C,T,H,W]
		t_value shape : [B,1] ###############
		'''
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # [B,H,W,2]
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)
        return output * mask
    
    def set_video_inputs(self, sample):
        self.batch['event_vox_clip'] = sample['event_vox_clip']
        self.batch['blur_input_clip'] = sample['blur_input_clip']
        self.batch['clean_middle'] = sample['clean_middle']
    
    def set_test_video_inputs(self, sample):
        self.batch['event_vox_clip'] = sample['event_vox_clip']
        self.batch['blur_input_clip'] = sample['blur_input_clip']
        self.batch['clean_gt_clip'] = sample['clean_gt_clip']
        self.batch['clean_middle'] = sample['clean_middle']

    def forward_deblur_net(self):
        self.batch['output_deblur'] = self.deblur_net(self.batch)
    
    def get_single_loss(self):
        self.loss_total = self.get_chainbor_loss(self.batch['clean_middle'], self.batch['output_deblur'][0])
        return self.loss_total

    def get_single_dict_loss(self):
        loss_dict = []
        for i in range(len(self.batch['output_deblur'])):
            loss_dict.append(self.get_chainbor_loss(self.batch['clean_middle'], self.batch['output_deblur'][i]))
        self.loss_total = sum(loss_dict)
        return self.loss_total
    
    ## multi scale loss
    def get_multi_scale_single_loss(self):
        loss_dict = []
        for scale_idx in range(len(self.batch['output_deblur'])):
            loss_dict.append(self.ms_lambda_dict[scale_idx]*self.get_chainbor_loss(self.batch['clean_middle_ms'][scale_idx], self.batch['output_deblur'][scale_idx]))
        self.loss_total = sum(loss_dict)
        return self.loss_total

    def update_loss_meters_deblur(self):
        # total loss update
        self.loss_total_meter.update(self.loss_total.item(), 1)
    
    def update_loss_meters_all(self):
        # total loss update
        self.loss_total_meter.update(self.loss_total.item(), 1)
        self.loss_deblur_meter.update(self.loss_image.item(), 1)

    def reset_loss_meters_deblur(self):
        self.loss_total_meter.reset()

    def reset_loss_meters_all(self):
        self.loss_total_meter.reset()
        self.loss_deblur_meter.reset()
    
    def del_batch(self):
        del self.batch
        self.batch = dict()
    
    def load_model(self, state_dict):
        self.deblur_net.load_state_dict(state_dict)
        print('load model')