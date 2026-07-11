#!/bin/bash
### OOD masking augmentation 학습: 일단은 딱 2개만 Freezer, Epilepsy 전체 5-fold

for data in freezer epilepsy
do
  for cv in 0 1 2 3 4
  do
    python ood_experiments/main_td.py \
      --model_type state \
      --train True \
      --data $data \
      --explainers empty \
      --fold $cv \
      --seed 42 \
      --device cuda:0 \
      --mask_prob 0.5 \
      --mask_num_segments 5
  done
done