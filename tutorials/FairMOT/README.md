# FairMOT

Place track_half_tram.py, track_half_sram.py and track_half_stram.py in the src directory.

Place the tools folder in the FairMOT root directory.

Replace src/lib/opts.py file.

Replace src/lib/tracker/multitracker.py file.

Replace src/lib/tracker/matching.py file.

Place [pre-trained weights](https://pan.baidu.com/s/1KvSzIbMeYQ4LWKxAUFMylw?pwd=7712) in the pretrained folder.

Run the following code to use tram, sram, and stram:

```
python track_half_tram.py mot --load_model ../exp/mot/mix_mot17_half_dla34.pth --val_mot17 True
python track_half_sram.py mot --load_model ../exp/mot/mix_mot17_half_dla34.pth --val_mot17 True
python track_half_stram.py mot --load_model ../exp/mot/mix_mot17_half_dla34.pth --val_mot17 True
```
