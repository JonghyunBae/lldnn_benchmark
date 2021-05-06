#!/bin/bash

#nvprof -f -o ~/test.nvprof \
nsys profile --trace=cuda,cudnn,nvtx --output=$1 --force-overwrite=true \
python3 train_net.py --config-file configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml --num-gpus 1
