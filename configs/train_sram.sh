EXP_DIR=output_dir
CUDA_VISIBLE_DEVICES=0
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 1111 \
    --use_env train_sram.py \
    --epoch 30 \
    --lr 2e-3 \
    --batch_size 15 \
    --weight_decay 0.01 \
    --lr_drop 10 \
    --output_dir ${EXP_DIR} \
    --merger_dropout 0 \
    --mark_ration 0.2 \
    --dataset 'MOT17' \
    --half