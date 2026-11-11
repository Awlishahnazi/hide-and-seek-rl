#!/bin/bash

# E1: Boxes Ablation
for boxes in 0 1 2 3
do
    for seed in 42 123 456
    do
        echo "Running E1: n_boxes=$boxes, seed=$seed"
        python src/hideseek/train.py --updates 20 --n_boxes $boxes --seed $seed
    done
done