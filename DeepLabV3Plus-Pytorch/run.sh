#!/bin/bash

#nvprof -f -o ~/test.nvprof \
nsys profile --trace=cuda,cudnn,nvtx --output=$1 --force-overwrite=true \
python3 main.py --model deeplabv3plus_mobilenet --dataset cityscapes --gpu_id 0  --lr 0.1  --crop_size 512 --batch_size 22 --output_stride 16 --data_root ./datasets/data/cityscapes

