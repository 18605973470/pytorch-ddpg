#!/bin/bash

save_dir="torcs-084"
load_dir="torcs-084"

python3 main.py --mode train \
            --seed 123456789 --experiment-name ${save_dir} --load True --load-dir ${load_dir} --gpu True --save-interval-step 5000 \
            --max-epsilon 1.0 --min-epsilon 0.01 \
            --value-lr 0.001 --policy-lr 0.0001 --gamma 0.99 --soft-tau 0.001 \
            --batch-size 32 --replay-buffer-size 20000 --num-heatup 100 \
            --max-episode-step 1000 --max-total-step 30000 --optimize-step 1 \
            --eval-interval-step 5000 --eval-episode 0
