import os
import time
import torch
from torch.utils import data as data
from torch.utils.data import ConcatDataset
from torchvision import transforms
import numpy as np
import random
from utils.utils import randomCrop
from PIL import Image


class Train_Video_Dataset(data.Dataset):
    def __init__(self, args, data_path, crop_size=256):
        super(Train_Video_Dataset, self).__init__()
        ## 
        self.num_frames_seq = args.num_test_video_frames
        self.middle_frame_id = self.num_frames_seq//2
        ## image and event prefix
        self.event_vox_prefix = 'event_voxel_parsed'
        self.blur_image_prefix =  'blur_processed_parsed'
        self.sharp_image_prefix = 'gt_processed_parsed'
        # transform
        self.transform = transforms.ToTensor()
        # data aug params
        self.get_filetaxnomy(data_path)
        ## crop
        self.crop_height = crop_size
        self.crop_width = crop_size
    
    def get_filetaxnomy(self, data_dir):
        self.input_dict = {}
        self.input_dict['blur_images'] = {}
        self.input_dict['blur_images']['0'] = []
        self.input_dict['blur_images']['1'] = []
        self.input_dict['blur_images']['2'] = []
        self.input_dict['blur_images']['3'] = []
        self.input_dict['sharp_images'] = {}
        self.input_dict['sharp_images']['0'] = []
        self.input_dict['sharp_images']['1'] = []
        self.input_dict['sharp_images']['2'] = []
        self.input_dict['sharp_images']['3'] = []
        self.input_dict['event_voxel'] = {}
        self.input_dict['event_voxel']['0'] = []
        self.input_dict['event_voxel']['1'] = []
        self.input_dict['event_voxel']['2'] = []
        self.input_dict['event_voxel']['3'] = []
        event_voxel_dir = os.path.join(data_dir, self.event_vox_prefix)
        blur_image_dir = os.path.join(data_dir, self.blur_image_prefix)
        sharp_image_dir = os.path.join(data_dir, self.sharp_image_prefix)
        for patch_idx in range(4):
            event_vox_patch_dir = os.path.join(event_voxel_dir, str(patch_idx).zfill(5))
            blur_patch_dir = os.path.join(blur_image_dir, str(patch_idx).zfill(5))
            sharp_patch_dir = os.path.join(sharp_image_dir, str(patch_idx).zfill(5))
            num_blur_images = len(os.listdir(blur_patch_dir))
            for image_idx in range(num_blur_images):
                blur_name = os.path.join(blur_patch_dir, str(image_idx).zfill(5) + '.png')
                sharp_name = os.path.join(sharp_patch_dir, str(image_idx).zfill(5) + '.png')
                left_voxel_name = os.path.join(event_vox_patch_dir, str(image_idx).zfill(5) + '.npz')
                self.input_dict['blur_images'][str(patch_idx)].append(blur_name)
                self.input_dict['sharp_images'][str(patch_idx)].append(sharp_name)
                self.input_dict['event_voxel'][str(patch_idx)].append(left_voxel_name)

    def __getitem__(self, index):
        ## patch number
        rand_patch_idx = np.random.randint(0, 4)
        ## event vox read
        event_vox_list = []
        blur_list, gt_list = [], []
        # new_index = self.num_frames_seq
        for video_num_idx in range(index, index + self.num_frames_seq):
            ## event voxel
            left_event_vox = np.load(self.input_dict['event_voxel'][str(rand_patch_idx)][video_num_idx])["data"]
            left_event_vox_tensor = torch.from_numpy(left_event_vox)
            event_vox_list.append(left_event_vox_tensor[None, ...])
            blur_image = Image.open(self.input_dict['blur_images'][str(rand_patch_idx)][video_num_idx])
            gt_image = Image.open(self.input_dict['sharp_images'][str(rand_patch_idx)][video_num_idx])
            blur_image_tensor = self.transform(blur_image)
            gt_image_tensor = self.transform(gt_image)
            blur_list.append(blur_image_tensor[None, ...])
            gt_list.append(gt_image_tensor[None, ...])
        blur_input_clip = torch.cat(blur_list)
        gt_clip = torch.cat(gt_list)
        gt_clip_middle = gt_clip[self.middle_frame_id]
        event_vox_tensor = torch.cat(event_vox_list)
        _, _, height, width =  gt_clip.shape
        # random crop
        x = random.randint(0, width - self.crop_width)
        y = random.randint(0, height - self.crop_height)
        gt_image_tensor = randomCrop(gt_clip, x, y, self.crop_height, self.crop_width)
        gt_clip_middle = randomCrop(gt_clip_middle, x, y, self.crop_height, self.crop_width )
        blur_input_clip = randomCrop(blur_input_clip, x, y, self.crop_height, self.crop_width)
        event_vox_cropped = randomCrop(event_vox_tensor, x, y, self.crop_height, self.crop_width)
        ### sample
        sample = {}
        sample['clean_gt_clip'] = gt_image_tensor
        sample['clean_middle'] = gt_clip_middle
        sample['blur_input_clip'] = blur_input_clip
        sample['event_vox_clip'] = event_vox_cropped
        return sample
    
    def __len__(self):
        return len(self.input_dict['blur_images']['0'])-self.num_frames_seq//2-1


class Test_Video_Dataset(data.Dataset):
    def __init__(self, args, data_path):
        super(Test_Video_Dataset, self).__init__()
        self.num_frames_seq = args.num_test_video_frames
        self.middle_frame_id = self.num_frames_seq//2
        ## image and event prefix
        self.event_vox_prefix = 'event_voxel'
        self.blur_image_prefix =  'blur_processed'
        self.sharp_image_prefix = 'gt_processed'
        # transform
        self.transform = transforms.ToTensor()
        # data aug params
        self.get_filetaxnomy(data_path)
    
    def get_filetaxnomy(self, data_dir):
        self.input_dict = {}
        self.input_dict['blur_images'] = []
        self.input_dict['sharp_images'] = []
        self.input_dict['event_voxel'] = []
        event_voxel_dir = os.path.join(data_dir, self.event_vox_prefix)
        blur_image_dir = os.path.join(data_dir, self.blur_image_prefix)
        sharp_image_dir = os.path.join(data_dir, self.sharp_image_prefix)
        num_blur_images = len(os.listdir(blur_image_dir))
        for image_idx in range(num_blur_images):
            blur_name = os.path.join(blur_image_dir, str(image_idx).zfill(5) + '.png')
            sharp_name = os.path.join(sharp_image_dir, str(image_idx).zfill(5) + '.png')
            left_voxel_name = os.path.join(event_voxel_dir, str(image_idx).zfill(5) + '.npz')
            self.input_dict['blur_images'].append(blur_name)
            self.input_dict['sharp_images'].append(sharp_name)
            self.input_dict['event_voxel'].append(left_voxel_name)

    def __getitem__(self, index):
        ## event vox read
        event_vox_list = []
        blur_list, gt_list = [], []
        for video_num_idx in range(index, index + self.num_frames_seq):
            ## event voxel
            left_event_vox = np.load(self.input_dict['event_voxel'][video_num_idx])["data"]
            left_event_vox_tensor = torch.from_numpy(left_event_vox)
            ## images
            blur_image = Image.open(self.input_dict['blur_images'][video_num_idx])
            gt_image = Image.open(self.input_dict['sharp_images'][video_num_idx])
            blur_image_tensor = self.transform(blur_image)
            gt_image_tensor = self.transform(gt_image)
            ## append to the list
            event_vox_list.append(left_event_vox_tensor[None, ...])
            blur_list.append(blur_image_tensor[None, ...])
            gt_list.append(gt_image_tensor[None, ...])
        blur_input_clip = torch.cat(blur_list)
        gt_clip = torch.cat(gt_list)
        gt_clip_middle = gt_clip[self.middle_frame_id]
        event_vox_tensor = torch.cat(event_vox_list)
        ## prepare sample
        sample = {}
        sample['clean_gt_clip'] = gt_clip
        sample['clean_middle'] = gt_clip_middle
        sample['blur_input_clip'] = blur_input_clip
        sample['event_vox_clip'] = event_vox_tensor
        return sample
    
    def __len__(self):
        return len(self.input_dict['blur_images'])-self.num_frames_seq//2-1

def get_train_dataset(args, mode):
    data_with_mode = os.path.join(args.data_dir, mode)
    scene_list = os.listdir(data_with_mode)
    dataset_list = []
    for scene in scene_list:
        data_path = os.path.join(args.data_dir, mode, scene)
        dset = Train_Video_Dataset(args, data_path)
        dataset_list.append(dset)
    dataset_train_concat = ConcatDataset(dataset_list)
    return dataset_train_concat

def get_test_dataset(args, mode):
    data_with_mode = os.path.join(args.data_dir, mode)
    scene_list = os.listdir(data_with_mode)
    dataset_list = []
    for scene in scene_list:
        data_path = os.path.join(data_with_mode, scene)
        dsets = Test_Video_Dataset(args, data_path)
        dataset_list.append(dsets)
    dataset_test_concat = ConcatDataset(dataset_list)
    return dataset_test_concat