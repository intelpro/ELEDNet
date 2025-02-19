import torch
import os
import datetime
import argparse
from torch.utils.data import DataLoader
from utils.utils import *
from utils.dataloader import  get_test_dataset
from models.model_manager import ModelManager
from tqdm import tqdm


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_batch_size', type = int, default=4)
    # training params
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
    parser.add_argument('--resume_ckpt', type=str2bool, default='True')
    parser.add_argument('--ckpt_dir', type = str, default='./pretrained_model/Ours_RELED.pth')
    parser.add_argument('--saved_dir', type = str, default='./saved_img')
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
        # Load the checkpoint if resuming from a saved model.
        if args.resume_ckpt:
            ckpt = torch.load(args.ckpt_dir)['model_state_dict']
            # Remove "module." prefix if it exists (for models trained with DataParallel).
            new_ckpt = {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in ckpt.items()}
            # Save the modified checkpoint
            self.model.load_model(new_ckpt)
        # Configure device settings
        self._setup_device()
        # save 
        self.output_dir = os.path.join(args.saved_dir, 'output_img')
        os.makedirs(self.output_dir, exist_ok=True)
        self.gt_dir = os.path.join(args.saved_dir, 'gt_img')
        os.makedirs(self.gt_dir, exist_ok=True)

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

    def test(self):
        """
        Performs testing on the dataset.
        - Iterates through the test loader and evaluates the model.
        - Computes PSNR and SSIM for each sample.
        - Logs the final evaluation results.
        """
        self.model.del_batch() 
        # 
        global_cnt = 0
        with torch.no_grad():  # Disable gradient calculations for testing.
            for sample in tqdm(self.test_loader, desc='Testing Progress'):
                sample = batch2device(sample)  
                self.model.set_video_inputs(sample) 
                self.model.forward_deblur_net() 
                # Compute PSNR and SSIM metrics.
                for batch_idx in range(args.val_batch_size):
                    output_img = 255*self.model.batch['output_deblur'][0][batch_idx, ...].squeeze().detach().cpu().numpy().transpose(1,2,0)
                    clean_middle = 255*self.model.batch['clean_middle'][batch_idx, ...].squeeze().detach().cpu().numpy().transpose(1,2,0)
                    cv2.imwrite(os.path.join(self.output_dir, str(global_cnt).zfill(5) + '.png'),  cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(self.gt_dir, str(global_cnt).zfill(5) + '.png'), cv2.cvtColor(clean_middle, cv2.COLOR_RGB2BGR))
                    global_cnt += 1
        self.model.del_batch()
        # Free up GPU memory.
        torch.cuda.empty_cache()


if __name__ == '__main__':
    args = get_argument()  # Parse arguments.
    tester = Tester(args)  # Initialize the Tester.
    tester.test()  # Run the test.