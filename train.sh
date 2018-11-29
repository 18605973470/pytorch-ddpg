#!/bin/bash

dir="ddpg_gym"

python3 main.py --seed 123456789 --experiment-name ${dir} --load False --gpu True --save-interval-step 50000 \
            --value-lr 0.001 --policy-lr 0.0001 --gamma 0.99 --soft-tau 0.001 \
            --batch_size 128 --replay-buffer-size 1000000 --num-heatup 2000 \
            --max-episode-step 2000 --max-total-step 2000000 --optimize-step 1 \
            --eval-interval-step 10000 --eval-episode 1