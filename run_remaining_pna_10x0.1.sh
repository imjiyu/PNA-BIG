#!/usr/bin/env bash

set -Eeuo pipefail
shopt -s nullglob

export PYTHONUNBUFFERED=1

ROOT="results_pna_10x0.1"
SEG="kalman_seg50_min1_max48"
LAM="_lam10.0x0.1"
SEED=42

DATASETS=(PAM boiler epilepsy wafer)
FOLDS=(0 1 2 3 4)
GPUS=(1 2 3 4 5 6 7)

DOMINANCE_METHODS=(
  "timing_td_trend_${SEG}${LAM}"
  "timing_td_residual_${SEG}${LAM}"
)

mkdir -p \
  "${ROOT}/eval_combined" \
  "${ROOT}/eval_dominance/logs"

rm -f \
  "${ROOT}/PIPELINE_DONE" \
  "${ROOT}/PIPELINE_FAILED"

timestamp()
{
  date '+%Y-%m-%d %H:%M:%S'
}

on_error()
{
  local rc=$?

  {
    echo "[$(timestamp)] PIPELINE FAILED"
    echo "exit_code=${rc}"
    echo "line=${BASH_LINENO[0]}"
  } | tee "${ROOT}/PIPELINE_FAILED"

  exit "${rc}"
}

trap on_error ERR


###############################################################################
# 0. 기존 combined 결과 확인
###############################################################################

combined_files=(
  "${ROOT}"/eval_combined/combined_eval_*_f*.csv
)

if ((${#combined_files[@]} != 20)); then
  echo "[ERROR] combined 개별 CSV가 20개여야 합니다."
  echo "[ERROR] 현재 발견된 파일 수: ${#combined_files[@]}"
  printf '  - %s\n' "${combined_files[@]}"
  exit 1
fi

echo "[$(timestamp)] combined 결과 확인 완료: 20/20"


###############################################################################
# 1. dominance attribution 파일 확인
###############################################################################

missing=0

for data in "${DATASETS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    output="${ROOT}/eval_dominance/full_eval_${data}_f${fold}.csv"

    # 이미 생성된 결과가 있으면 attribution 재검증 생략
    if [[ -s "${output}" ]]; then
      continue
    fi

    for method in "${DOMINANCE_METHODS[@]}"; do
      attribution="${ROOT}/${data}_state_${method}_result_${fold}_${SEED}.npy"

      if [[ ! -s "${attribution}" ]]; then
        echo "[MISSING] ${attribution}"
        missing=1
      fi
    done
  done
done

if ((missing != 0)); then
  echo "[ERROR] dominance 평가에 필요한 attribution 파일이 누락되었습니다."
  exit 1
fi

echo "[$(timestamp)] dominance attribution 파일 확인 완료"


###############################################################################
# 2. dominance 평가
###############################################################################

pids=()
labels=()
launched=0

wait_batch()
{
  local failed=0
  local i

  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "[$(timestamp)] [OK] ${labels[$i]}"
    else
      echo "[$(timestamp)] [FAIL] ${labels[$i]}"
      failed=1
    fi
  done

  pids=()
  labels=()

  if ((failed != 0)); then
    return 1
  fi
}

echo
echo "[$(timestamp)] dominance 평가를 시작합니다."

for data in "${DATASETS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    output="${ROOT}/eval_dominance/full_eval_${data}_f${fold}.csv"
    log_file="${ROOT}/eval_dominance/logs/evalD_${data}_f${fold}.log"
    label="${data}:fold${fold}"

    if [[ -s "${output}" ]]; then
      echo "[$(timestamp)] [SKIP] ${label}: 기존 CSV 사용"
      continue
    fi

    gpu="${GPUS[$((launched % ${#GPUS[@]}))]}"

    echo "[$(timestamp)] [START] ${label}, physical GPU=${gpu}"

    CUDA_VISIBLE_DEVICES="${gpu}" \
      python eval_cpd_cpp.py \
        --data "${data}" \
        --fold "${fold}" \
        --device cuda:0 \
        --npy_dir "${ROOT}" \
        --output_file "${output}" \
        --methods "${DOMINANCE_METHODS[@]}" \
        > "${log_file}" 2>&1 &

    pids+=("$!")
    labels+=("${label}")
    launched=$((launched + 1))

    # GPU 1~7에 7개씩 실행
    if ((${#pids[@]} == ${#GPUS[@]})); then
      wait_batch
    fi
  done
done

if ((${#pids[@]} > 0)); then
  wait_batch
fi

echo "[$(timestamp)] dominance 평가 작업 완료"


###############################################################################
# 3. dominance 결과 검증
###############################################################################

dominance_files=(
  "${ROOT}"/eval_dominance/full_eval_*_f*.csv
)

if ((${#dominance_files[@]} != 20)); then
  echo "[ERROR] dominance 개별 CSV가 20개여야 합니다."
  echo "[ERROR] 현재 발견된 파일 수: ${#dominance_files[@]}"
  exit 1
fi

for file in "${dominance_files[@]}"; do
  if [[ ! -s "${file}" ]]; then
    echo "[ERROR] 비어 있는 dominance CSV: ${file}"
    exit 1
  fi
done

echo "[$(timestamp)] dominance 결과 확인 완료: 20/20"


###############################################################################
# 4. combined 및 dominance CSV 병합
###############################################################################

echo
echo "[$(timestamp)] CSV 병합을 시작합니다."

python - "${ROOT}" <<'PY'
import glob
import os
import sys

import pandas as pd


root = sys.argv[1]

targets = [
    {
        "name": "combined",
        "pattern": os.path.join(
            root,
            "eval_combined",
            "combined_eval_*_f*.csv",
        ),
        "output": os.path.join(
            root,
            "eval_combined",
            "combined_eval.csv",
        ),
    },
    {
        "name": "dominance",
        "pattern": os.path.join(
            root,
            "eval_dominance",
            "full_eval_*_f*.csv",
        ),
        "output": os.path.join(
            root,
            "eval_dominance",
            "full_eval.csv",
        ),
    },
]

for target in targets:
    files = sorted(glob.glob(target["pattern"]))

    print(
        f"[INFO] {target['name']} 개별 CSV: "
        f"{len(files)}개"
    )

    if len(files) != 20:
        raise RuntimeError(
            f"{target['name']} CSV가 20개여야 하지만 "
            f"{len(files)}개가 발견되었습니다."
        )

    frames = []

    for file in files:
        frame = pd.read_csv(file)

        if frame.empty:
            raise RuntimeError(f"빈 CSV입니다: {file}")

        frames.append(frame)
        print(f"  - {file}: {len(frame)} rows")

    merged = pd.concat(
        frames,
        ignore_index=True,
    )

    merged.to_csv(
        target["output"],
        index=False,
    )

    print(
        f"[DONE] {target['output']}: "
        f"{len(merged)} rows"
    )
PY


###############################################################################
# 5. TR 표 생성
###############################################################################

echo
echo "[$(timestamp)] TR_table.py를 실행합니다."

python TR_table.py \
  --results_dir "${ROOT}" \
  --eval_csv "${ROOT}/eval_dominance/full_eval.csv" \
  --out_dir "${ROOT}/eval_dominance" \
  --folds 0 1 2 3 4


###############################################################################
# 완료
###############################################################################

{
  echo "[$(timestamp)] PIPELINE DONE"
  echo "root=${ROOT}"
} | tee "${ROOT}/PIPELINE_DONE"

echo
echo "============================================================"
echo "[$(timestamp)] 남은 3단계가 모두 완료되었습니다."
echo "============================================================"
