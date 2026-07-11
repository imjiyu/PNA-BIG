#!/bin/bash

set -u

mkdir -p logs
mkdir -p results_pna/eval_tmp

DATA="PAM"
FOLD=1
SEED=42
LAM0="10.0"
LAMFS=("10.0" "1.0" "0.1")

# 기본값 30. 실행할 때 TESTBS=10 등으로 변경 가능
TESTBS="${TESTBS:-30}"

FINAL_CSV="results_pna/PAM_val_cpd_combined_kalman.csv"

# 중간 평가 결과만 삭제합니다.
# 기존 attribution NPY는 절대 삭제하지 않습니다.
rm -f results_pna/eval_tmp/PAM_val_cpd_combined_kalman_lam10.0x*.csv
rm -f "${FINAL_CSV}"

pids=()
names=()

for i in "${!LAMFS[@]}"; do
  lf="${LAMFS[$i]}"
  gpu="${i}"

  method="timing_td_combined_kalman_seg50_min1_max48_val_lam${LAM0}x${lf}"
  out_csv="results_pna/eval_tmp/PAM_val_cpd_combined_kalman_lam${LAM0}x${lf}.csv"
  log_file="logs/eval_PAM_lam${LAM0}x${lf}.log"

  npy_file="results_pna/PAM_state_${method}_result_${FOLD}_${SEED}.npy"

  if [[ ! -f "${npy_file}" ]]; then
    echo "[ERROR] attribution NPY가 없습니다."
    echo "${npy_file}"
    exit 1
  fi

  echo "[RUN] lam=${LAM0}x${lf}, gpu=${gpu}, batch=${TESTBS}"

  CUDA_VISIBLE_DEVICES="${gpu}" python -u eval_cpd_cpp.py \
    --data "${DATA}" \
    --fold "${FOLD}" \
    --seed "${SEED}" \
    --model_type state \
    --device cuda:0 \
    --testbs "${TESTBS}" \
    --npy_dir results_pna \
    --output_file "${out_csv}" \
    --eval_split val \
    --topk 0.1 \
    --top 0 \
    --methods "${method}" \
    > "${log_file}" 2>&1 &

  pids+=("$!")
  names+=("${LAM0}x${lf}")
done

failed=0

for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${names[$i]}"

  if wait "${pid}"; then
    echo "[DONE] lam=${name}"
  else
    echo "[FAILED] lam=${name}"
    echo "로그: logs/eval_PAM_lam${name}.log"
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "[ERROR] 일부 평가가 실패했습니다."
  exit 1
fi

echo "[MERGE] CSV 병합 중"

first=1

for lf in "${LAMFS[@]}"; do
  csv_file="results_pna/eval_tmp/PAM_val_cpd_combined_kalman_lam${LAM0}x${lf}.csv"

  if [[ ! -s "${csv_file}" ]]; then
    echo "[ERROR] CSV가 없거나 비어 있습니다: ${csv_file}"
    exit 1
  fi

  if (( first == 1 )); then
    cat "${csv_file}" > "${FINAL_CSV}"
    first=0
  else
    tail -n +2 "${csv_file}" >> "${FINAL_CSV}"
  fi
done

echo "============================================"
echo "PAM CPD evaluation finished"
echo "Result: ${FINAL_CSV}"
echo "============================================"