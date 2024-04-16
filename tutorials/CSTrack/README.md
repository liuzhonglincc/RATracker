# CSTrack

Place test_cstrack_sram.py, test_cstrack_tram.py and test_cstrack_stram.py in the CSTrack/tracking directory.

Place the tools folder in the CSTrack/tracking root directory.

Replace CSTrack/lib/tracker/cstrack.py file.

Replace CSTrack/lib/mot_online/matching.py file.

Place [pre-trained weights](https://pan.baidu.com/s/1KvSzIbMeYQ4LWKxAUFMylw?pwd=7712) in the pretrained folder.

Run the following code to use tram, sram, and stram:

```
python test_cstrack_tram.py
python test_cstrack_sram.py
python test_cstrack_stram.py
```
