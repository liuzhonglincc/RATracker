# OC_SORT

Place run_ocsort_tram.py, run_ocsort_sram.py and run_ocsort_stram.py in the tools directory.

Place the tools folder in the OC_SORT root directory.

Replace yolox/evaluators/mot_evaluator.py file.

Replace trackers/ocsort_tracker/ocsort.py file.

Replace trackers/ocsort_tracker/association.py file.

Replace utils/args.py file.

Add trackers/ocsort_tracker/matching.py.

Place [pre-trained weights](https://pan.baidu.com/s/1KvSzIbMeYQ4LWKxAUFMylw?pwd=7712) in the pretrained folder.

Run the following code to use tram, sram, and stram:

```
python3 tools/run_ocsort_tram.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse --expn aaa --weight_1 0.8
python3 tools/run_ocsort_sram.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse --expn aaa --weight_1 0.8
python3 tools/run_ocsort_stram.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse --expn aaa --weight_st 1.2
```
