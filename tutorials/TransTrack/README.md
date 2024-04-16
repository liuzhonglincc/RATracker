# TransTrack

Place main_track_tram.py, main_track_sram.py and main_track_stram.py in the TransTrack root directory.

Place the tools folder in the TransTrack root directory.

Replace models/tracker.py file.

Replace engine_track.py file.

Place [pre-trained weights](https://pan.baidu.com/s/1KvSzIbMeYQ4LWKxAUFMylw?pwd=7712) in the pretrained folder.

Run the following code to use tram, sram, and stram:

```
python3 main_track_tram.py --dataset_file mot --coco_path mot --batch_size 1 --resume pretrained/671mot17_crowdhuman_mot17.pth --eval --with_box_refine
python3 main_track_sram.py --dataset_file mot --coco_path mot --batch_size 1 --resume pretrained/671mot17_crowdhuman_mot17.pth --eval --with_box_refine --num_queries 500
python3 main_track_stram.py --dataset_file mot --coco_path mot --batch_size 1 --resume pretrained/671mot17_crowdhuman_mot17.pth --eval --with_box_refine --num_queries 500
```
