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
    # training params
    parser.add_argument('--num_train_video_frames', type = int, default=3)
    parser.add_argument('--num_test_video_frames', type = int, default=3)
    parser.add_argument('--voxel_num_bins', type = int, default=16)
    parser.add_argument('--learning_rate', type = float, default=1e-4)
    parser.add_argument('--mode', type = str, default='test')
    # model discription
    parser.add_argument('--model_folder', type=str, default='model_factory')
    parser.add_argument('--model_name', type=str, default='models_final')
    # data loading params
    parser.add_argument('--experiment_name', type = str, default='test_networks')
    parser.add_argument('--num_threads', type = int, default=12)
    parser.add_argument('--data_dir', type = str, default = '/media/mnt2/dataset/RELED/')
    parser.add_argument('--use_multigpu', type=str2bool, default='True')
    parser.add_argument('--resume_ckpt', type=str2bool, required=True)
    parser.add_argument('--ckpt_dir', type = str, required=True)
    args = parser.parse_args()
    return args


class Tester:
    def __init__(self, args):
        """
        Initializes the Tester class for evaluating the model.
        - Sets up the test data loader.
        - Initializes and loads the model.
        - Defines evaluation metrics (PSNR, SSIM).
        - Configures logging.
        """
        self.args = args
        # Define the logging and saving path for the experiment (includes date and experiment name).
        tb_path = f'./experiments/{datetime.datetime.now().strftime("%y%m%d-" + args.experiment_name + "/%H%M")} '
        # Create the test dataset loader.
        self.test_loader = DataLoader(get_test_dataset(args, mode='test'),
                                      batch_size=args.val_batch_size, shuffle=False,
                                      num_workers=args.num_threads, pin_memory=False)
        # Initialize the model.
        self.model = ModelManager(args)
        self.model.initilalize_deblur_model(args, model_folder=args.model_folder, model_name=args.model_name, tb_path=tb_path)
        # Define evaluation metrics.
        self.PSNR_calculator = PSNR()
        self.SSIM_calculator = SSIM()
        # Load the checkpoint if resuming from a saved model.
        if args.resume_ckpt:
            ckpt = torch.load(args.ckpt_dir)['model_state_dict']
            # Remove "module." prefix if it exists (for models trained with DataParallel).
            new_ckpt = {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in ckpt.items()}
            self.model.load_model(new_ckpt)
        # Set up the logger.
        self.logger = get_logger(tb_path, 'log.txt', 'append')
        # Configure device settings
        self._setup_device()
        # Log the provided arguments for tracking.
        self._log_arguments()

    def _setup_device(self):
        """
        Configures the computing device.
        - Moves the model to GPU if available.
        - Enables multi-GPU support if specified.
        """
        if torch.cuda.is_available():
            self.model.cuda_deblur()
        if self.args.use_multigpu:
            self.model.use_multi_gpu_deblur()

    def _log_arguments(self):
        """
        Logs all the provided arguments and the total parameter count of the model.
        """
        self.logger.info(f'Overall parameter count: {self.model.count_total_parameters() * 1e-6:.4f} MB')
        for arg, val in vars(self.args).items():
            self.logger.info(f'{arg}: {val}')

    def test(self):
        """
        Performs testing on the dataset.
        - Iterates through the test loader and evaluates the model.
        - Computes PSNR and SSIM for each sample.
        - Logs the final evaluation results.
        """
        psnr_meter, ssim_meter = AverageMeter(), AverageMeter()
        self.model.del_batch() 

        with torch.no_grad():  # Disable gradient calculations for testing.
            for sample in tqdm(self.test_loader, desc='Testing Progress'):
                sample = batch2device(sample)  
                self.model.set_video_inputs(sample) 
                self.model.forward_deblur_net() 
                # Compute PSNR and SSIM metrics.
                psnr_meter.update(self.PSNR_calculator(self.model.batch['clean_middle'], self.model.batch['output_deblur'][0]).mean().item())
                ssim_meter.update(self.SSIM_calculator(self.model.batch['clean_middle'], self.model.batch['output_deblur'][0]).mean().item())

        self.model.del_batch()
        self.logger.info(f'Total evaluation:  PSNR: {psnr_meter.avg}  SSIM: {ssim_meter.avg}')
        # Free up GPU memory.
        torch.cuda.empty_cache()


if __name__ == '__main__':
    args = get_argument()  # Parse arguments.
    tester = Tester(args)  # Initialize the Tester.
    tester.test()  # Run the test.