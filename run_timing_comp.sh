#!/bin/bash
# TIMING-global (completeness form) 추출 — 데이터셋별 seg, 5-fold 병렬, zero baseline, 100 step.
# run_td_all.sh 구조에서 --explainers our_td → timing_comp, --baseline zero 로만 바꾼 버전.
set -u
mkdir -p logs_timing_comp
FAILED=()

run_dataset () {
  data=$1; num_segments=$2; min_seg_len=$3; max_seg_len=$4; testbs=$5; log_prefix=$6
  echo "===== ${data} 시작 ====="
  pids=(); logs=()
  for f in 0 1 2 3 4; do
    gpus=(0 1 3 4 6); gpu=${gpus[$f]}
    log_file="logs_timing_comp/${log_prefix}_fold${f}_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES=${gpu} python real/main_td.py \
      --explainers our timing_comp \
      --baseline zero \
      --data ${data} \
      --device cuda:0 \
      --fold ${f} \
      --seed 42 \
      --num_segments ${num_segments} \
      --min_seg_len ${min_seg_len} \
      --max_seg_len ${max_seg_len} \
      --testbs ${testbs} \
      > "${log_file}" 2>&1 &
    pids+=($!); logs+=("${log_file}")
    echo "${data} fold ${f} -> GPU ${gpu}, log: ${log_file}"
  done
  status=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[ERROR] ${data} fold ${i} 실패. log: ${logs[$i]}"; status=1
    fi
  done
  [ ${status} -ne 0 ] && FAILED+=("${data}") && echo "===== ${data} 일부 실패 =====" || echo "===== ${data} 완료 ====="
  echo ""
}

# 데이터셋별 seg 인자는 run_td_all.sh / check_completeness.py 와 동일
run_dataset epilepsy 10 10 10  5  epilepsy
run_dataset boiler   50 1  36  30 boiler
run_dataset wafer    5  10 152 10 wafer
run_dataset PAM      10 10 600 3  PAM

echo "전체 실행 종료"
if [ ${#FAILED[@]} -ne 0 ]; then
  echo "실패 데이터셋:"; printf ' - %s\n' "${FAILED[@]}"
else
  echo "모든 데이터셋 완료"
fi
