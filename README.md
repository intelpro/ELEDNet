# ELEDNet(ECCV 2024)
**Official repository for the ECCV 2024 paper, "Towards Real-world Event-guided Low-light Video Enhancement and Deblurring"**

Currently, this pages only includes information about the dataset. I will soon be sharing more details on this pages.

### Downloading the RELED datasets 
Please download and unzip the RELED dataset.

* [[RELED-Train](https://drive.google.com/file/d/1Syf_hhmyzXvHlhoMHQU4TSaEtXPkIT3U/view?usp=drive_link)] / [[RELED-Test](https://drive.google.com/file/d/1y-8cjnTHOyOz6jgy0T-gMcmiAnwdRf5R/view?usp=drive_link)]

RELED dataset

It follows the below directory format:
```
├──── YOUR_DIR/
    ├──── test/
       ├──── 0001/
          ├──── blur_processed/
             ├──── 00000.png
             ├──── ...
             └──── 00148.png
          ├──── gt_processed/
             ├──── 00000.png
             ├──── ...
             └──── 000148.png
          ├──── parsed_event/
             ├──── 00000.npz
             ├──── ...
             └──── 00148.npz
          ├──── event_voxel/
             ├──── 00000.npz
             ├──── ...
             └──── 00148.npz
       ├──── .../
```

Each subdirectory corresponding to blur_processed, gt_processed, parsed_event, and event_voxel represents the following:
* blur_processed: low-light blurred image
* gt_processed: normal-light sharp image
* parsed_event: parsed event
* event_voxel: event voxel
