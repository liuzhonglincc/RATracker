# Representation Alignment Contrastive Regularization for Multi-Object Tracking

**Paper Link**: [arXiv](https://arxiv.org/abs/2404.02562)

<div align="center">
    <img src="figure/method.png" width="100%">
</div>

## Requirements

**Step 1.** Install python and create Conda environment.
```
conda create -n Ratracker python=3.8
conda activate Ratracker
```

**Step 2.** Install torch and matched torchvision from [pytorch.org](https://pytorch.org/get-started/locally/).<br>
The code was tested using torch==1.13.0 and torchvision==0.14.0.

**Step 3.** Install RATracker.
```
git clone https://github.com/liuzhonglincc/RATracker.git
cd RATracker
pip3 install -r requirements.txt
python3 setup.py develop
```

**Step 4.** Install [pycocotools](https://github.com/cocodataset/cocoapi).
```
pip3 install cython; 
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

**Step 5.** Others
```
pip3 install cython_bbox
pip3 install faiss-cpu
pip3 install faiss-gpu
```

We additionally used [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and [FastReID](https://github.com/JDAI-CV/fast-reid), please refer to their installation guides for additional setup options.

## Data Preparation

Download [MOT17](https://motchallenge.net/data/MOT17/) and [MOT20](https://motchallenge.net/data/MOT20/) from the [official website](https://motchallenge.net/). And put them in the following structure:

```
<dataets_dir>
      │
      ├── MOT17
      │      ├── train
      │      └── test    
      │
      └── MOT20
             ├── train
             └── test
```
Run the following code to obtain the validation set for MOT17:
```
cd tools/datasets
python convert_mot17_to_coco.py 
```

## Training
Run the following command to train TRAM, SRAM, and STRAM, and you can specify the training dataset through the --dataset and --half parameters.
```
bash configs/train_tram.sh
bash configs/train_sram.sh
bash configs/train_stram.sh
```
You can download the pretrained model [here](https://pan.baidu.com/s/1_67t832ofmumPGMwkpsXGw?pwd=7712).

## Tracking

We have used ByteTrack and FastReID, you need to download [bytetrack_ablation](https://github.com/ifzhang/ByteTrack), [bytetrack_x_mot17](https://github.com/ifzhang/ByteTrack), [bytetrack_x_mot20](https://github.com/ifzhang/ByteTrack), and [mot17_sbs_S50](https://drive.google.com/file/d/1QZFWpoa80rqo7O-HXmlss8J8CnS7IUsN/view).

* **Evaluation on MOT17 half val**

The following code is used to apply tram, sram, and stram to the original Baseline method.
```
python tools/track_baseline.py
python tools/track_baseline.py --tram --pretrained pretrained/mot17_half_tram.pth
python tools/track_baseline.py --sram --pretrained pretrained/mot17_half_sram.pth
python tools/track_baseline.py --stram --pretrained pretrained/mot17_half_stram.pth
```
| Model     | IDF1 | MOTA | IDS |
|------------|------|------|------|
|Baseline |  75.56 | 79.85 | 495 |
|Baseline+TRAM |  76.34 | 80.51 | 478 |
|Baseline+SRAM |  75.94 | 80.50 | 480 |
|Baseline+STRAM |  77.14 | 81.14 | 479 |

The following code is used to apply stram to the original IoU metric, Re-ID metric, or IoU+Re-ID metric. You can use the "metric" parameter to specify which metric to use, and the "stram" parameter to indicate whether stram should be used.
```
python3 tools/track.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse --metric iou
python3 tools/track.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse --metric iou --stram --pretrained pretrained/mot17_half_stram.pth
python3 tools/track.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse --metric reid
python3 tools/track.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse --metric reid --stram --pretrained pretrained/mot17_half_stram.pth
python3 tools/track.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse --metric iou_reid
python3 tools/track.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse --metric iou_reid --stram --pretrained pretrained/mot17_half_stram.pth
```

* **Test on MOT17 and MOT20**

Run the following code to verify on the MOT17 and MOT20 test sets:
```
python3 tools/track.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "test" --fp16 --fuse --metric iou --stram --pretrained pretrained/mot17_test_stram.pth
python3 tools/track.py datasets/MOT20 --default-parameters --benchmark "MOT20" --eval "test" --fp16 --fuse --metric iou --stram --pretrained pretrained/mot20_test_stram.pth
```

By submitting the results from the YOLOX_outputs folder to the MOTChallenge website, you can achieve the same performance as the paper. We adopted the ByteTrack approach by fine-tuning tracking parameters for each scene to achieve higher performance.

## Applying RAM to other trackers

See [tutorials](https://github.com/liuzhonglincc/RATracker/tree/main/tutorials).

## Visualization results on MOT17 and MOT20

<img src="figure/MOT17_09_FRCNN.gif" width="400"/>
<img src="figure/MOT20_02.gif" width="400"/>   

## Citation
If you find this repository useful, please consider citing our paper:
```
@article{chen2024representation,
  title={Representation Alignment Contrastive Regularization for Multi-Object Tracking},
  author={Chen, Shujie and Liu, Zhonglin and Dong, Jianfeng and Zhou, Di},
  journal={arXiv preprint arXiv:2404.02562},
  year={2024}
}
```
## Acknowledgement
A large part of the code is borrowed from [ByteTrack](https://github.com/ifzhang/ByteTrack), [FastReID](https://github.com/JDAI-CV/fast-reid), [MOTR](https://github.com/megvii-research/MOTR), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and [YOLOv7](https://github.com/wongkinyiu/yolov7). Many thanks for their wonderful works.

This research was funded by the National Natural Science Foundation of China grant number 62002323.