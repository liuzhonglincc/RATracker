# Representation Alignment Contrastive Regularization for Multi-Object Tracking

**Paper Link**: [arXiv](https://arxiv.org/abs/2404.02562)

<div align="center">
    <img src="figure/method.png" width="100%">
</div>

## Requirements

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
## Tracking

## Applying RAM to other trackers

## Demo

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