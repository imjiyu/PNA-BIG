#!/bin/bash
mkdir -p logs
mkdir -p results_pna

bs_for() {
  case $1 in
    epilepsy) echo 5;;
    boiler) echo 30;;
    PAM) echo 3;;
    *) echo 5;;
  esac
}

i=0
NGPU=8

for data in boiler epilepsy; do
  for l0 in 0.1 1 10; do
    for lf in 0.1 1 10; do
      gpu=$(( i % NGPU ))

      CUDA_VISIBLE_DEVICES=$gpu python -u real/main_td.py \
        --data $data \
        --fold 0 \
        --seed 42 \
        --explainers our_td \
        --num_segments 0 \
        --min_seg_len 1 \
        --max_seg_len 48 \
        --baseline pna \
        --pna_feature hidden \
        --pna_ka 5 \
        --pna_lam0 $l0 \
        --pna_lamf $lf \
        --eval_split val \
        --model_type state \
        --device cuda:0 \
        --testbs $(bs_for $data) \
        > logs/${data}_orderavg_plainig_lam${l0}x${lf}.log 2>&1 &

      i=$(( i + 1 ))

      # 8개 채우면 대기 후 다음 웨이브
      (( i % NGPU == 0 )) && wait
    done
  done
done

wait
echo "sweep done"