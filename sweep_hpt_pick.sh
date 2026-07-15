#!/usr/bin/env bash
# Attribution은 기존 결과 재사용
# average-fill CPD 평가 후 데이터셋별 lambda 선택
set -u

DATASETS="epilepsy wafer boiler PAM"
FOLDS="0 1 2 3 4"
GPU_IDS=(0 1 3 4 5)
NGPU=${#GPU_IDS[@]}

SEG="kalman_seg0_min1_max48"
NPY_DIR="results_pna"
MASK_REFS_VAL="average"
SEED=42

COMBOS=(
  "0.5 0.5"  "1 0.5"  "3 0.5"  "5 0.5"  "10 0.5"
  "0.5 1"    "1 1"    "3 1"    "5 1"    "10 1"
  "0.5 3"    "1 3"    "3 3"    "5 3"    "10 3"
  "0.5 5"    "1 5"    "3 5"    "5 5"    "10 5"
  "0.5 10"   "1 10"   "3 10"   "5 10"   "10 10"
  "0.5 15"   "1 15"   "3 15"   "5 15"   "10 15"
  "0.5 20"   "1 20"   "3 20"   "5 20"   "10 20"
)

float_tag() {
  python -c 'import sys; print(str(float(sys.argv[1])))' "$1"
}

# Wafer validation split을 attribution 생성 때와 동일하게 재현
export PNA_TUNE_VAL=1

mkdir -p logs "${NPY_DIR}/sweep5_eval"

# eval 파일 위치 자동 확인
if [[ -f real/eval_cpd_cpp.py ]]; then
  EVAL_SCRIPT="real/eval_cpd_cpp.py"
elif [[ -f eval_cpd_cpp.py ]]; then
  EVAL_SCRIPT="eval_cpd_cpp.py"
else
  echo "[ERROR] eval_cpd_cpp.py를 찾을 수 없습니다."
  exit 1
fi

if [[ ! -f pick_lambda.py ]]; then
  echo "[ERROR] pick_lambda.py가 없습니다."
  exit 1
fi

# ============================================================================
# 사전 검사: 필요한 attribution 700개가 전부 있는지 확인
# ============================================================================
echo "================ Attribution 파일 검사 ================"

missing=0
: > logs/sweep5_missing_attr.txt

for combo in "${COMBOS[@]}"; do
  read -r l0 lf <<< "$combo"
  l0t="$(float_tag "$l0")"
  lft="$(float_tag "$lf")"

  for data in $DATASETS; do
    for fold in $FOLDS; do
      npy="${NPY_DIR}/${data}_state_timing_td_combined_${SEG}_val_lam${l0t}x${lft}_result_${fold}_${SEED}.npy"

      if [[ ! -f "$npy" ]]; then
        echo "$npy" | tee -a logs/sweep5_missing_attr.txt
        missing=$((missing+1))
      fi
    done
  done
done

if ((missing > 0)); then
  echo "[STOP] Attribution 파일 ${missing}개가 없습니다."
  echo "       logs/sweep5_missing_attr.txt를 확인하세요."
  exit 1
fi

echo "[OK] Attribution 700개 확인 완료"

# ============================================================================
# 공통 wave 스케줄러
# ============================================================================
pids=()
names=()
eval_fail=0
: > logs/sweep5_eval_failed.txt

wait_wave() {
  local idx

  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      echo "[OK][EVAL] ${names[$idx]}"
    else
      echo "[FAIL][EVAL] ${names[$idx]}"
      echo "${names[$idx]}" >> logs/sweep5_eval_failed.txt
      eval_fail=$((eval_fail+1))
    fi
  done

  pids=()
  names=()
}

# ============================================================================
# [2/3] CPD 평가: dataset별로 모든 lambda 조합 수행
# ============================================================================
echo
echo "================ [2/3] CPD 평가 (val, average only) ================"

for data in $DATASETS; do
  echo
  echo "================ DATASET: ${data} ================"

  i=0

  for combo in "${COMBOS[@]}"; do
    read -r l0 lf <<< "$combo"

    l0t="$(float_tag "$l0")"
    lft="$(float_tag "$lf")"
    method="timing_td_combined_${SEG}_val_lam${l0t}x${lft}"

    for fold in $FOLDS; do
      gpu="${GPU_IDS[$((i % NGPU))]}"
      name="${data}_f${fold}_lam${l0}x${lf}"
      out="${NPY_DIR}/sweep5_eval/cmb_${name}.csv"

      rm -f "$out"

      CUDA_VISIBLE_DEVICES="$gpu" python -u "$EVAL_SCRIPT" \
        --data "$data" --fold "$fold" --seed "$SEED" \
        --device cuda:0 \
        --eval_split val \
        --npy_dir "$NPY_DIR" \
        --mask_refs "$MASK_REFS_VAL" \
        --cpd_only \
        --output_file "$out" \
        --methods "$method" \
        > "logs/sweep5_eval_${name}.log" 2>&1 &

      pids+=("$!")
      names+=("$name")
      i=$((i+1))

      ((${#pids[@]} == NGPU)) && wait_wave
    done
  done

  ((${#pids[@]} > 0)) && wait_wave
  echo "[DATASET DONE] ${data}"
done

echo "[EVAL DONE] 실패=${eval_fail}"

# ============================================================================
# [3/3] 집계 + lambda 선택
# ============================================================================
echo
echo "================ [3/3] 5-fold val 집계 & lambda 선택 ================"

python pick_lambda.py \
  --eval_dir "${NPY_DIR}/sweep5_eval" \
  --select_ref average

echo
echo "============================== DONE =============================="
echo "Evaluation 실패 : ${eval_fail}"
echo "per-fold CSV    : ${NPY_DIR}/sweep5_eval/"
echo "선택 결과       : chosen_lambdas.csv"
echo "전체 평균 표    : sweep5_table_average.csv"
echo "fold 표준편차   : sweep5_std_average.csv"
echo "fold 개수       : sweep5_count_average.csv"
echo "=================================================================="