#!/bin/bash

ngpu=1
dataset="coco"
batch_size=64
print_freq=10
lr=0.01
epochs=3
period=300
img_size1=640
img_size2=640
iters=-1
data_dir="../kitti"

#nvprof -f -o ~/test_%p.nvprof --profile-child-processes \
nsys profile --trace=cuda,cudnn,nvtx --output=$1 --force-overwrite=true \
python3 -m torch.distributed.launch --nproc_per_node=${ngpu} --use_env train.py \
--use-cuda --epochs ${epochs} --period ${period} --batch-size ${batch_size} --lr ${lr} --img-sizes ${img_size1} ${img_size2} \
--dataset ${dataset} --data-dir ${data_dir} --iters ${iters} --mosaic --dali --print-freq ${print_freq}
