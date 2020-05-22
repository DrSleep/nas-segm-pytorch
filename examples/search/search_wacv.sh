#!/bin/sh
PYTHONPATH=$(pwd):$PYTHONPATH/src python src/main_search.py \
    --n-task0 1000 \
    --ctrl-version 'wacv' \
    --num-ops 6 \
    --num-agg-ops 2 \
    --do-kd '' \
    --dec-aux-weight 0 \
    --val-crop-size 512 \
    --val-batch-size 10 \
    --val-resize-side 1024 \
    --batch-size 32 32 \
    --crop-size 321 321 \
    --resize-side 1024 1024 \
    --resize-longer-side \
    --enc-lr 1e-3 5e-4 \
    --dec-lr 7e-3 2e-3 \
    --ctrl-lr 1e-4 \
    --num-classes 19 19 \
    --enc-optim 'adam' \
    --dec-optim 'adam' \
    --num-segm-epochs 5 5 \
    --val-every 5 1 \
    --cell-num-layers 7 \
    --dec-num-cells 3 \
    --train-dir './data/datasets/cs' \
    --val-dir './data/datasets/cs' \
    --train-list './data/lists/train.cs' \
    --val-list './data/lists/train.cs'



