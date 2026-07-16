#!/bin/bash
set -u

mkdir -p logs_td

FAILED_DATASETS=()

run_dataset () {
  data=$1
  num_segments=$2
  min_seg_len=$3
  max_seg_len=$4
  testbs=$5
  log_prefix=$6

  echo "===== ${data} 시작 ====="

  pids=()
  logs=()

  for f in 0 1 2 3 4; do
    gpu=$((f + 3))
    log_file="logs_td/${log_prefix}_fold${f}_gpu${gpu}.log"

    CUDA_VISIBLE_DEVICES=${gpu} python real/main_td.py \
      --explainers our_td \
      --data ${data} \
      --device cuda:0 \
      --fold ${f} \
      --seed 42 \
      --num_segments ${num_segments} \
      --min_seg_len ${min_seg_len} \
      --max_seg_len ${max_seg_len} \
      --testbs ${testbs} \
      > "${log_file}" 2>&1 &

    pids+=($!)
    logs+=("${log_file}")

    echo "${data} fold ${f} -> GPU ${gpu}, log: ${log_file}"
  done

  status=0

  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[ERROR] ${data} fold ${i} 실패. log 확인: ${logs[$i]}"
      status=1
    fi
  done

  if [ ${status} -ne 0 ]; then
    echo "===== ${data} 일부 실패. 그래도 다음 데이터셋으로 넘어감 ====="
    FAILED_DATASETS+=("${data}")
  else
    echo "===== ${data} 완료 ====="
  fi

  echo ""
}

run_dataset epilepsy 10 10 10 5 epilepsy
run_dataset freezer  5  10 100 5  freezer
run_dataset boiler   50 1  36  30 boiler
run_dataset wafer    5  10 152 10 wafer
run_dataset PAM      10 10 600 3  PAM

echo "전체 실행 종료"

if [ ${#FAILED_DATASETS[@]} -ne 0 ]; then
  echo "실패 또는 일부 실패 데이터셋:"
  printf ' - %s\n' "${FAILED_DATASETS[@]}"
else
  echo "모든 데이터셋 완료"
fi
