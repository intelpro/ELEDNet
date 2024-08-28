# ELEDNet (ECCV 2024)

Official repository for the ECCV 2024 paper, **"Towards Real-world Event-guided Low-light Video Enhancement and Deblurring."** 

[[Paper](https://arxiv.org/abs/2408.14916)] 
[[Supp](https://drive.google.com/file/d/1xBy29Iy3ae7V0YTasPGBbE9Xf6fNUX3L/view?usp=sharing)] 


Currently, this pages only includes information about the dataset and paper. I will soon be sharing more details on this pages.

## Video Demos
![ELEDNet Demo_2]([https://github.com//myProject/raw/main/assets/ELEDNet_demo2.gif](https://github.com/intelpro/ELEDNet/blob/main/ELEDNet_demo2.gif))



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

## Contact
If you have any question, please send an email to taewoo(intelpro@kaist.ac.kr)

## License
The project codes and datasets can be used for research and education only. 
