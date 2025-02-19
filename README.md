# ELEDNet (ECCV 2024)

Official repository for the ECCV 2024 paper, **"Towards Real-world Event-guided Low-light Video Enhancement and Deblurring."** 

[[Paper](https://arxiv.org/abs/2408.14916)] 
[[Supp](https://drive.google.com/file/d/1xBy29Iy3ae7V0YTasPGBbE9Xf6fNUX3L/view?usp=sharing)] 


## Video Demos
![ELEDNet Demo_1](https://github.com/intelpro/ELEDNet/blob/main/Figure/ELEDNet_demo1_v2.gif)
![ELEDNet Demo_2](https://github.com/intelpro/ELEDNet/blob/main/Figure/ELEDNet_demo2_v2.gif)



## Downloading the RELED datasets 
Please download and unzip the RELED dataset.

* [[RELED-Train](https://drive.google.com/file/d/1SiUTEOm6ZrLgXnh2t1LeUqy0xDjiubH6/view?usp=drive_link)] / [[RELED-Test](https://drive.google.com/file/d/18XXfjZ59rQulFRH18UNHI9Gm0ZRGeJwN/view)]

The dataset follows the below directory format:
```
├── RELED/
    ├── train/
    │   ├── 0000/
    │   │   ├── blur_processed/
    │   │   │   ├── 00000.png
    │   │   │   ├── ...
    │   │   │   └── 00148.png
    │   │   ├── gt_processed/
    │   │   │   ├── 00000.png
    │   │   │   ├── ...
    │   │   │   └── 00148.png
    │   │   ├── events/
    │   │   │   ├── 00000.npz
    │   │   │   ├── ...
    │   │   │   └── 00148.npz
    │   │   └── event_voxel/
    │   │       ├── 00000.npz
    │   │       ├── ...
    │   │       └── 00148.npz
    │   ├── 0001/
    │   │   ├── ...
    ├── test/
    │   ├── 0000/
    │   │   ├── ...
    │   ├── 0001/
    │   │   ├── ...
```

Sub-directory Descriptions:
- **blur_processed**: Contains low-light blurred images (`*.png` files).
- **gt_processed**: Contains normal-light sharp images (`*.png` files).
- **events**: Contains raw event data in `.npz` format.
- **event_voxel**: Contains event voxel data in `.npz` format.

Reading Raw Event Data (`events`) and Event Voxel Data(`event_voxel`):

To read `event` and `event voxel` data from `.npz` files using Python and NumPy:

```python
import numpy as np

# Replace YOUR_EVENT_DIR with the path to the directory containing the .npz files for events
event_data = np.load('YOUR_EVENT_DIR/*.npz')['data']
```

## Requirements
* PyTorch 1.8.0
* CUDA 11.2
* python 3.8

## Quick Train model

Download repository:

``` bash
$ git clone https://github.com/intelpro/ELEDNet
```


If you want to start training our model, you need to preprocess the raw dataset first. 

Run the following command to preprocess the dataset:  

```bash
$ python utils/make_train_dataset --train_data_dir ${TRAIN_DATASET_DIR} 
```

- **`${TRAIN_DATASET_DIR}`**: Specifies the directory containing the **training dataset** of the **RELED dataset**.
- The process **divides the blur, event voxel, and ground truth (GT) data into four parts** to enhance training speed.  

Once preprocessing is complete, you can proceed to the model training step.

```bash
$ python train.py --data_dir ${DATSET_DIR}
```

- **`${DATSET_DIR}`**: Specifies the directory containing the complete RELED dataset, including both training and test sets.

## Quick Test model 

Download repository:

``` bash
$ git clone https://github.com/intelpro/ELEDNet
```

Download network weights(trained on RELED datasets) and place downloaded model in ./pretrained_model

* [[Ours](https://drive.google.com/file/d/1VixOqExExkulDx4RV7O15H4SCQN-iB9o/view?usp=sharing)]

Generate output images using our model and sample data provided in this repository.

``` bash
python test_sample.py --resume_ckpt True --ckpt_dir ./pretrained_model/Ours_RELED.pth
```


## Reference  
> Taewoo Kim, Jaeseok Jeong, Hoonhee Cho, Yuhwan Jeong, and Kuk-Jin Yoon, **"Towards Real-World Event-guided Low-Light Video Enhancement and Deblurring,"** In *ECCV*, 2024.  
```bibtex
@inproceedings{kim2024towards,
  title={Towards Real-world Event-guided Low-light Video Enhancement and Deblurring},
  author={Kim, Taewoo and Jeong, Jaeseok and Cho, Hoonhee and Jeong, Yuhwan and Yoon, Kuk-Jin},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={433--451},
  year={2024},
  publisher={Springer}
}
```

## Contact
If you have any question, please send an email to taewoo(an625148@gmail.com)

## License
The project codes and datasets can be used for research and education only. 