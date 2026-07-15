#!/usr/bin/env bash
# run_pna_all.sh
# PNA-BIG attribution 생성: 4 dataset × 5 fold, lambda 10x10 고정
set -u
mkdir -p logs/pna_lam10 results_pna

idx=0; pids=(); names=(); failed=0
wait_batch() {
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "[OK] ${names[$i]}"
    else echo "[FAIL] ${names[$i]} — check log"; failed=$((failed+1)); fi
  done
  pids=(); names=()
}

for data in epilepsy wafer PAM boiler; do
  for fold in 0 1 2 3 4; do
    gpu=$((idx % 4))                       # GPU 개수에 맞게 조정
    name="${data}_f${fold}"
    echo "[START] gpu=${gpu} ${name}"
    CUDA_VISIBLE_DEVICES=${gpu} \
      python real/main_td.py \
        --explainers our_td \
        --data "${data}" --fold "${fold}" --seed 42 \
        --baseline pna --model_type state --eval_split test \
        --pna_lam0 10.0 --pna_lamf 10.0 --pna_ka 5 \
        --num_segments 50 --min_seg_len 1 --max_seg_len 48 \
        --device cuda:0 --testbs 200 \
        > "logs/pna_lam10/${name}.log" 2>&1 &
    pids+=("$!"); names+=("${name}"); idx=$((idx+1))
    if (( ${#pids[@]} == 4 )); then wait_batch; fi   # GPU 개수만큼 배치
  done
done
(( ${#pids[@]} > 0 )) && wait_batch
echo "[DONE] attribution 생성 완료, 실패=${failed}"
