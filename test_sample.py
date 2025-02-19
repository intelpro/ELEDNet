import torch
import argparse
from utils.utils import *
from models.model_manager import ModelManager
from torchvision import transforms
import torchvision.transforms.functional as TF  # Functional API 사용
from PIL import Image



def get_argument():
    parser = argparse.ArgumentParser()
    # params
    parser.add_argument('--num_test_video_frames', type = int, default=3)
    parser.add_argument('--mode', type = str, default='test')
    # model discription
    parser.add_argument('--model_folder', type=str, default='model_factory')
    parser.add_argument('--model_name', type=str, default='models_final')
    # data loading params
    parser.add_argument('--experiment_name', type = str, default='test_networks')
    parser.add_argument('--num_threads', type = int, default=12)
    parser.add_argument('--sample_folder_path', type = str, default='./sample_data')
    parser.add_argument('--resume_ckpt', type=str2bool, required=True)
    parser.add_argument('--ckpt_dir', type = str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Parse command-line arguments
    args = get_argument()
    # Initialize model manager
    model = ModelManager(args)
    # Initialize the deblurring model
    model.initilalize_deblur_model(args, model_folder=args.model_folder, model_name=args.model_name, tb_path=None)
    # Load the checkpoint if resuming from a saved model
    if args.resume_ckpt:
        ckpt = torch.load(args.ckpt_dir)['model_state_dict']
        # Remove "module." prefix if it exists (for models trained with DataParallel)
        new_ckpt = {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in ckpt.items()}
        # Load the modified checkpoint into the model
        model.load_model(new_ckpt)
    # Configure device settings
    if torch.cuda.is_available():
        model.cuda_deblur() 
    # Configure output directory for saving results
    output_dir = os.path.join(args.sample_folder_path, 'output_folder')
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    # Clear any previous batch data in the model
    model.del_batch()
    # Define transformation for image conversion
    transform = transforms.ToTensor()
    with torch.no_grad():
        sample = dict() 
        ## Load file paths for input data
        blur_image_path = os.path.join(args.sample_folder_path, 'blur_images')
        event_voxel_path = os.path.join(args.sample_folder_path, 'event_voxel') 
        # Get sorted filenames for blur images and event voxels
        blur_image_names = sorted(os.listdir(blur_image_path))
        event_voxel_names = sorted(os.listdir(event_voxel_path))
        # Lists to store tensor representations of images and event voxels
        event_vox_list, blur_list = [], []
        # Iterate through the given number of test video frames
        for i in range(args.num_test_video_frames):
            # Load blur image and convert it to tensor
            blur_image = Image.open(os.path.join(blur_image_path, blur_image_names[i]))
            blur_image_tensor = transform(blur_image) 
            # Load event voxel data (assuming .npz format with "data" key)
            event_voxel = np.load(os.path.join(event_voxel_path, event_voxel_names[i]))["data"]
            event_vox_tensor = torch.from_numpy(event_voxel)
            # Append tensors to the corresponding lists (adding batch dimension)
            event_vox_list.append(event_vox_tensor[None, ...])
            blur_list.append(blur_image_tensor[None, ...])
        # Concatenate the tensors along the batch dimension and add extra dimension
        event_vox_tensor = torch.cat(event_vox_list)[None, ...]  # Shape: (1, num_frames, ...)
        blur_input_clip = torch.cat(blur_list)[None, ...]  # Shape: (1, num_frames, C, H, W)
        # Assign processed inputs to the sample dictionary
        sample['event_vox_clip'] = event_vox_tensor
        sample['blur_input_clip'] = blur_input_clip
        # Move data to the appropriate device
        sample = batch2device(sample)
        # Set inputs for the model
        model.set_test_inputs(sample)
        # Run the deblurring model
        model.forward_deblur_net()
        # Extract the deblurred output from the model's batch dictionary
        output_deblur = model.batch['output_deblur'][0]  # Extract first frame of output
        # Convert tensor output to a PIL image
        output_deblur_cpu = TF.to_pil_image(output_deblur.cpu().squeeze())
        # Save the output image to the designated directory
        output_deblur_cpu.save(os.path.join(output_dir, blur_image_names[args.num_test_video_frames // 2]))
    # Clear batch data after processing
    model.del_batch()
