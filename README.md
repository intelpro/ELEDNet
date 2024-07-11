# ELEDNet(ECCV 2024)
**Official repository for the ECCV 2024 paper, "Towards Real-world Event-guided Low-light Video Enhancement and Deblurring"**

Currently, this pages only includes information about the dataset. I will soon be sharing more details on this pages.

### Downloading the RELED datasets 
Please download and unzip the RELED dataset.

* [[RELED-Train](https://drive.google.com/file/d/1Syf_hhmyzXvHlhoMHQU4TSaEtXPkIT3U/view?usp=drive_link)] / [[RELED-Test](https://drive.google.com/file/d/1y-8cjnTHOyOz6jgy0T-gMcmiAnwdRf5R/view?usp=drive_link)]

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

### Sub-directory Descriptions:
- **blur_processed/**: Contains low-light blurred images (`*.png` files).
- **gt_processed/**: Contains normal-light sharp images (`*.png` files).
- **events/**: Contains raw event data in `.npz` format.
- **event_voxel/**: Contains event voxel data in `.npz` format.
