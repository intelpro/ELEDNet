import os
import glob
import cv2
import numpy as np
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Low-light deblurring dataset parser")
    parser.add_argument("--train_data_dir", type=str, default = '/media/mnt2/dataset/RELED/train',
                        help="Path to the training dataset directory")
    args = parser.parse_args()
    # Prefix setting..
    event_vox_prefix = 'event_voxel'
    blur_prefix = 'blur_processed'
    gt_prefix = 'gt_processed'
    event_vox_parsed_prefix = 'event_voxel_parsed'
    blur_parsed_prefix = 'blur_processed_parsed'
    gt_parsed_prefix = 'gt_processed_parsed'
    print(f"Train data directory: {args.train_data_dir}")
    print(f"Event voxel directory: {event_vox_prefix}")
    print(f"Blur directory: {blur_prefix}")
    print(f"GT directory: {gt_prefix}")
    ## scene list
    scene_list = os.listdir(args.train_data_dir)
    scene_list.sort()
    for scene in scene_list:
        ## event vox
        event_vox_dir = os.path.join(args.train_data_dir, scene, event_vox_prefix)
        event_vox_list = glob.glob(os.path.join(event_vox_dir, '*.npz'))
        ## blur
        blur_dir = os.path.join(args.train_data_dir, scene, blur_prefix)
        blur_list = glob.glob(os.path.join(blur_dir, '*.png'))
        ## gt
        gt_dir = os.path.join(args.train_data_dir, scene, gt_prefix)
        gt_list = glob.glob(os.path.join(gt_dir, '*.png'))
        event_vox_list.sort()
        blur_list.sort()
        gt_list.sort()
        num_data = len(event_vox_list)
        ## target dir
        event_vox_parsed_dir = os.path.join(args.train_data_dir, scene, event_vox_parsed_prefix)
        blur_parsed_dir = os.path.join(args.train_data_dir, scene, blur_parsed_prefix)
        gt_parsed_dir = os.path.join(args.train_data_dir, scene, gt_parsed_prefix)
        if not os.path.exists(event_vox_parsed_dir):
            os.makedirs(event_vox_parsed_dir)
        if not os.path.exists(blur_parsed_dir):
            os.makedirs(blur_parsed_dir)
        if not os.path.exists(gt_parsed_dir):
            os.makedirs(gt_parsed_dir)
        ## clean dir
        clean_image_dir_0 = os.path.join(gt_parsed_dir, str(0).zfill(5))
        clean_image_dir_1 = os.path.join(gt_parsed_dir, str(1).zfill(5))
        clean_image_dir_2 = os.path.join(gt_parsed_dir, str(2).zfill(5))
        clean_image_dir_3 = os.path.join(gt_parsed_dir, str(3).zfill(5))
        if not os.path.exists(clean_image_dir_0):
            os.makedirs(clean_image_dir_0)
        if not os.path.exists(clean_image_dir_1):
            os.makedirs(clean_image_dir_1)
        if not os.path.exists(clean_image_dir_2):
            os.makedirs(clean_image_dir_2)
        if not os.path.exists(clean_image_dir_3):
            os.makedirs(clean_image_dir_3)
        ## blur dir
        blur_dir_0 = os.path.join(blur_parsed_dir, str(0).zfill(5))
        blur_dir_1 = os.path.join(blur_parsed_dir, str(1).zfill(5))
        blur_dir_2 = os.path.join(blur_parsed_dir, str(2).zfill(5))
        blur_dir_3 = os.path.join(blur_parsed_dir, str(3).zfill(5))
        if not os.path.exists(blur_dir_0):
            os.makedirs(blur_dir_0)
        if not os.path.exists(blur_dir_1):
            os.makedirs(blur_dir_1)
        if not os.path.exists(blur_dir_2):
            os.makedirs(blur_dir_2)
        if not os.path.exists(blur_dir_3):
            os.makedirs(blur_dir_3)
        ## vox dir
        vox_dir_0 = os.path.join(event_vox_parsed_dir, str(0).zfill(5))
        vox_dir_1 = os.path.join(event_vox_parsed_dir, str(1).zfill(5))
        vox_dir_2 = os.path.join(event_vox_parsed_dir, str(2).zfill(5))
        vox_dir_3 = os.path.join(event_vox_parsed_dir, str(3).zfill(5))
        if not os.path.exists(vox_dir_0):
            os.makedirs(vox_dir_0)
        if not os.path.exists(vox_dir_1):
            os.makedirs(vox_dir_1)
        if not os.path.exists(vox_dir_2):
            os.makedirs(vox_dir_2)
        if not os.path.exists(vox_dir_3):
            os.makedirs(vox_dir_3)
        for data_idx in range(num_data):
            ### sharp
            image_name = gt_list[data_idx]
            cur_image = cv2.imread(image_name)

            h,w,c = cur_image.shape 
            h_u = int(h/2)
            w_u = int(w/2)

            saved_sharp_image_name1 = os.path.join(clean_image_dir_0, str(data_idx).zfill(5) + '.png')
            saved_sharp_image_name2 = os.path.join(clean_image_dir_1, str(data_idx).zfill(5) + '.png')
            saved_sharp_image_name3 = os.path.join(clean_image_dir_2, str(data_idx).zfill(5) + '.png')
            saved_sharp_image_name4 = os.path.join(clean_image_dir_3, str(data_idx).zfill(5) + '.png')

            c1=cur_image[:h_u, :w_u,:]
            c2=cur_image[h_u:, w_u:,:]
            c3=cur_image[:h_u, w_u:,:]
            c4=cur_image[h_u:, :w_u,:]

            cv2.imwrite(saved_sharp_image_name1, c1)   
            cv2.imwrite(saved_sharp_image_name2, c2)   
            cv2.imwrite(saved_sharp_image_name3, c3)   
            cv2.imwrite(saved_sharp_image_name4, c4)  

            ### blur
            blur_image_name = blur_list[data_idx]
            blur_image = cv2.imread(blur_image_name)

            saved_blur_image_name1 = os.path.join(blur_dir_0, str(data_idx).zfill(5) + '.png')
            saved_blur_image_name2 = os.path.join(blur_dir_1, str(data_idx).zfill(5) + '.png')
            saved_blur_image_name3 = os.path.join(blur_dir_2, str(data_idx).zfill(5) + '.png')
            saved_blur_image_name4 = os.path.join(blur_dir_3, str(data_idx).zfill(5) + '.png')

            b1=blur_image[:h_u, :w_u,:]
            b2=blur_image[h_u:, w_u:,:]
            b3=blur_image[:h_u, w_u:,:]
            b4=blur_image[h_u:, :w_u,:]

            cv2.imwrite(saved_blur_image_name1, b1)   
            cv2.imwrite(saved_blur_image_name2, b2)   
            cv2.imwrite(saved_blur_image_name3, b3)   
            cv2.imwrite(saved_blur_image_name4, b4)  

            ## event voxel
            voxel_grid = np.load(event_vox_list[data_idx])["data"]
            v1 = voxel_grid[:,:h_u, :w_u]
            v2 = voxel_grid[:,h_u:, w_u:]
            v3 = voxel_grid[:,:h_u, w_u:]
            v4 = voxel_grid[:,h_u:, :w_u]

            np.savez_compressed(os.path.join(vox_dir_0, str(data_idx).zfill(5) + '.npz'), data=v1)
            np.savez_compressed(os.path.join(vox_dir_1, str(data_idx).zfill(5) + '.npz'), data=v2)
            np.savez_compressed(os.path.join(vox_dir_2, str(data_idx).zfill(5) + '.npz'), data=v3)
            np.savez_compressed(os.path.join(vox_dir_3, str(data_idx).zfill(5) + '.npz'), data=v4)