# ByteTrack

Place track_tram.py, track_sram.py and track_stram.py in the tools directory.

Place the tools folder in the ByteTrack root directory.

Replace yolox/evaluators/mot_evaluator.py file.

Replace yolox/tracker/byte_tracker.py file.

Replace yolox/tracker/matching.py file.

Place [pre-trained weights](https://pan.baidu.com/s/1KvSzIbMeYQ4LWKxAUFMylw?pwd=7712) in the pretrained folder.

Run the following code to use tram, sram, and stram:

```
python3 tools/track_tram.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/track_sram.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/track_stram.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse
```