#!/usr/bin/env bash
# =============================================================================
# eval_wafer_lf01_only.sh
#
# 목적:
#   Wafer에 대해서만 아래 추가 lambda 조합을 검증한다.
#
#     lam0 ∈ {0.1, 0.5, 1, 3, 5, 10}
#     lamf = 0.1
#     Ka   = 5
#
# 기존 lambda sweep 결과와 chosen_lambdas.csv는 절대 수정하지 않는다.
#
# 새 결과:
#   attribution : results_pna/wafer_lf01_check/npy/
#   CPD CSV     : results_pna/wafer_lf01_check/eval/
#   비교 결과   : results_pna/wafer_lf01_check/wafer_lf01_comparison.csv
# =============================================================================
set -uo pipefail

DATA="wafer"
FOLDS=(0 1 2 3 4)
GPU_IDS=(0 1 3 4 5)
NGPU=${#GPU_IDS[@]}

LAMF="0.1"
LAM0_VALUES=(0.1 0.5 1 3 5 10)

KA=5
SEED=42
SEG="kalman_seg0_min1_max48"

ATTR_TESTBS=200
EVAL_TESTBS=100

NPY_DIR="results_pna"

# 기존 CPD 결과: 읽기만 함
OLD_EVAL_DIR="${NPY_DIR}/sweep5_eval"

# 새 실험 전용 경로
OUT_ROOT="${NPY_DIR}/wafer_lf01_check"
NEW_NPY_DIR="${OUT_ROOT}/npy"
NEW_EVAL_DIR="${OUT_ROOT}/eval"
LOG_DIR="logs/wafer_lf01_check"

COMPARE_CSV="${OUT_ROOT}/wafer_lf01_comparison.csv"

float_tag() {
  python -c 'import sys; print(str(float(sys.argv[1])))' "$1"
}

export PNA_TUNE_VAL=1

mkdir -p \
  "$NEW_NPY_DIR" \
  "$NEW_EVAL_DIR" \
  "$LOG_DIR"

# =============================================================================
# 실행 파일 확인
# =============================================================================
if [[ -f real/main_td.py ]]; then
  MAIN_SCRIPT="real/main_td.py"
elif [[ -f main_td.py ]]; then
  MAIN_SCRIPT="main_td.py"
else
  echo "[ERROR] main_td.py를 찾을 수 없습니다."
  exit 1
fi

if [[ -f eval_cpd_cpp.py ]]; then
  EVAL_SCRIPT="eval_cpd_cpp.py"
elif [[ -f eval_cpd_cpp.py ]]; then
  EVAL_SCRIPT="eval_cpd_cpp.py"
else
  echo "[ERROR] eval_cpd_cpp.py를 찾을 수 없습니다."
  exit 1
fi

if [[ ! -d "$OLD_EVAL_DIR" ]]; then
  echo "[ERROR] 기존 평가 경로가 없습니다: ${OLD_EVAL_DIR}"
  exit 1
fi

# =============================================================================
# 경로 함수
# =============================================================================
expected_root_attr() {
  local fold="$1"
  local l0="$2"
  local l0t
  local lft

  l0t="$(float_tag "$l0")"
  lft="$(float_tag "$LAMF")"

  echo "${NPY_DIR}/${DATA}_state_timing_td_combined_${SEG}"\
"_val_lam${l0t}x${lft}_result_${fold}_${SEED}.npy"
}

expected_saved_attr() {
  local fold="$1"
  local l0="$2"
  local root_path

  root_path="$(expected_root_attr "$fold" "$l0")"
  echo "${NEW_NPY_DIR}/$(basename "$root_path")"
}

# =============================================================================
# wave 스케줄러
# =============================================================================
pids=()
names=()
attr_fail=0
eval_fail=0

: > "${LOG_DIR}/attr_failed.txt"
: > "${LOG_DIR}/eval_failed.txt"

wait_wave() {
  local kind="$1"
  local idx

  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      echo "[OK][${kind}] ${names[$idx]}"
    else
      echo "[FAIL][${kind}] ${names[$idx]}"

      if [[ "$kind" == "ATTR" ]]; then
        echo "${names[$idx]}" >> "${LOG_DIR}/attr_failed.txt"
        attr_fail=$((attr_fail + 1))
      else
        echo "${names[$idx]}" >> "${LOG_DIR}/eval_failed.txt"
        eval_fail=$((eval_fail + 1))
      fi
    fi
  done

  pids=()
  names=()
}

# 중단되더라도 main_td.py가 완성한 새 파일은 전용 폴더에 보존
collect_generated_attributions() {
  local l0 fold src dst

  for l0 in "${LAM0_VALUES[@]}"; do
    for fold in "${FOLDS[@]}"; do
      src="$(expected_root_attr "$fold" "$l0")"
      dst="$(expected_saved_attr "$fold" "$l0")"

      if [[ -s "$src" && ! -s "$dst" ]]; then
        mv -f "$src" "$dst"
        echo "[SAVE][ATTR] $(basename "$dst")"
      fi
    done
  done
}

cleanup() {
  local status="${1:-$?}"
  local pid

  trap - EXIT INT TERM

  if ((${#pids[@]} > 0)); then
    kill "${pids[@]}" 2>/dev/null || true

    for pid in "${pids[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
  fi

  collect_generated_attributions
  exit "$status"
}

trap 'cleanup $?' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

# =============================================================================
# [1/3] 새 lambda 조합 attribution 생성
# =============================================================================
echo
echo "================ [1/3] Wafer 추가 attribution 생성 ================"
echo "lam0 = ${LAM0_VALUES[*]}"
echo "lamf = ${LAMF}"
echo "Ka   = ${KA}"

for l0 in "${LAM0_VALUES[@]}"; do
  echo
  echo "---------------- lam0=${l0}, lamf=${LAMF} ----------------"

  pids=()
  names=()
  i=0

  for fold in "${FOLDS[@]}"; do
    src="$(expected_root_attr "$fold" "$l0")"
    dst="$(expected_saved_attr "$fold" "$l0")"
    name="${DATA}_f${fold}_ka${KA}_lam${l0}x${LAMF}"

    # 전용 폴더에 이미 있으면 재사용
    if [[ -s "$dst" ]]; then
      echo "[SKIP][ATTR] ${name}"
      continue
    fi

    # 이전 실행에서 results_pna 최상위에 남은 완성 파일 회수
    if [[ -s "$src" ]]; then
      mv -f "$src" "$dst"
      echo "[RECOVER][ATTR] ${name}"
      continue
    fi

    gpu="${GPU_IDS[$((i % NGPU))]}"

    CUDA_VISIBLE_DEVICES="$gpu" python -u "$MAIN_SCRIPT" \
      --data "$DATA" \
      --fold "$fold" \
      --seed "$SEED" \
      --explainers our_td \
      --num_segments 0 \
      --min_seg_len 1 \
      --max_seg_len 48 \
      --baseline pna \
      --pna_feature hidden \
      --pna_ka "$KA" \
      --pna_lam0 "$l0" \
      --pna_lamf "$LAMF" \
      --eval_split val \
      --model_type state \
      --device cuda:0 \
      --testbs "$ATTR_TESTBS" \
      > "${LOG_DIR}/attr_${name}.log" 2>&1 &

    pids+=("$!")
    names+=("$name")
    i=$((i + 1))
  done

  ((${#pids[@]} > 0)) && wait_wave ATTR

  # 현재 lambda 조합의 결과를 즉시 별도 폴더로 이동
  for fold in "${FOLDS[@]}"; do
    src="$(expected_root_attr "$fold" "$l0")"
    dst="$(expected_saved_attr "$fold" "$l0")"
    name="${DATA}_f${fold}_ka${KA}_lam${l0}x${LAMF}"

    if [[ -s "$dst" ]]; then
      continue
    fi

    if [[ -s "$src" ]]; then
      mv -f "$src" "$dst"
    else
      echo "[MISSING][ATTR] ${name}"
      echo "$name" >> "${LOG_DIR}/attr_failed.txt"
      attr_fail=$((attr_fail + 1))
    fi
  done
done

if ((attr_fail > 0)); then
  echo "[STOP] Attribution 실패 ${attr_fail}건이 있습니다."
  echo "       ${LOG_DIR}/attr_failed.txt 확인"
  exit 1
fi

echo "[ATTR DONE] 새 attribution 30개 확인 완료"

# =============================================================================
# CSV 완료 여부
# =============================================================================
eval_complete() {
  local path="$1"

  [[ -s "$path" ]] || return 1

  python - "$path" <<'PY'
import csv
import math
import sys

path = sys.argv[1]

try:
    with open(path, newline="") as fp:
        for row in csv.DictReader(fp):
            if row.get("metric") != "CPD":
                continue
            if row.get("mask_ref") != "average":
                continue

            try:
                value = float(row["cum_diff"])
            except (KeyError, TypeError, ValueError):
                continue

            if math.isfinite(value):
                raise SystemExit(0)
except (OSError, csv.Error):
    pass

raise SystemExit(1)
PY
}

# =============================================================================
# [2/3] 새 조합 average-fill CPD 평가
# =============================================================================
echo
echo "================ [2/3] Wafer 추가 조합 CPD 평가 ================"

for l0 in "${LAM0_VALUES[@]}"; do
  l0t="$(float_tag "$l0")"
  lft="$(float_tag "$LAMF")"

  method="timing_td_combined_${SEG}_val_lam${l0t}x${lft}"

  pids=()
  names=()
  i=0

  for fold in "${FOLDS[@]}"; do
    gpu="${GPU_IDS[$((i % NGPU))]}"
    name="${DATA}_f${fold}_lam${l0}x${LAMF}"
    out="${NEW_EVAL_DIR}/cmb_${name}.csv"

    if eval_complete "$out"; then
      echo "[SKIP][EVAL] ${name}"
      continue
    fi

    rm -f "$out"

    CUDA_VISIBLE_DEVICES="$gpu" python -u "$EVAL_SCRIPT" \
      --data "$DATA" \
      --fold "$fold" \
      --seed "$SEED" \
      --device cuda:0 \
      --eval_split val \
      --testbs "$EVAL_TESTBS" \
      --npy_dir "$NEW_NPY_DIR" \
      --mask_refs average \
      --cpd_only \
      --output_file "$out" \
      --methods "$method" \
      > "${LOG_DIR}/eval_${name}.log" 2>&1 &

    pids+=("$!")
    names+=("$name")
    i=$((i + 1))
  done

  ((${#pids[@]} > 0)) && wait_wave EVAL
done

# 프로세스 성공 여부와 별개로 결과 파일 최종 검사
for l0 in "${LAM0_VALUES[@]}"; do
  for fold in "${FOLDS[@]}"; do
    name="${DATA}_f${fold}_lam${l0}x${LAMF}"
    out="${NEW_EVAL_DIR}/cmb_${name}.csv"

    if ! eval_complete "$out"; then
      echo "[MISSING][EVAL] ${name}"
      echo "$name" >> "${LOG_DIR}/eval_failed.txt"
      eval_fail=$((eval_fail + 1))
    fi
  done
done

if ((eval_fail > 0)); then
  echo "[STOP] CPD 평가 실패 ${eval_fail}건이 있습니다."
  echo "       ${LOG_DIR}/eval_failed.txt 확인"
  exit 1
fi

echo "[EVAL DONE] 새 CPD 결과 30개 확인 완료"

# =============================================================================
# [3/3] 기존 Wafer sweep과 새 조합 비교
#
# chosen_lambdas.csv 및 기존 결과 파일은 수정하지 않는다.
# =============================================================================
echo
echo "================ [3/3] 기존 최적과 추가 조합 비교 ================"

python - \
  "$OLD_EVAL_DIR" \
  "$NEW_EVAL_DIR" \
  "$COMPARE_CSV" <<'PY'
import csv
import glob
import math
import os
import re
import statistics
import sys
from collections import defaultdict

old_dir = sys.argv[1]
new_dir = sys.argv[2]
out_csv = sys.argv[3]

pattern = re.compile(
    r"cmb_wafer_f(\d+)_lam"
    r"([0-9]+(?:\.[0-9]+)?)x"
    r"([0-9]+(?:\.[0-9]+)?)\.csv"
)

expected_folds = {0, 1, 2, 3, 4}


def read_cpd(path):
    values = []

    with open(path, newline="") as fp:
        for row in csv.DictReader(fp):
            if row.get("metric") != "CPD":
                continue
            if row.get("mask_ref") != "average":
                continue

            try:
                value = float(row["cum_diff"])
            except (KeyError, TypeError, ValueError):
                continue

            if math.isfinite(value):
                values.append(value)

    if not values:
        return None

    # 재실행 등으로 중복 행이 있으면 해당 파일 내 평균
    return sum(values) / len(values)


def collect(directory, source):
    fold_values = defaultdict(list)

    for path in sorted(glob.glob(os.path.join(
        directory, "cmb_wafer_f*_lam*x*.csv"
    ))):
        match = pattern.fullmatch(os.path.basename(path))
        if not match:
            continue

        fold = int(match.group(1))
        lam0 = float(match.group(2))
        lamf = float(match.group(3))

        value = read_cpd(path)
        if value is None:
            continue

        fold_values[(source, lam0, lamf, fold)].append(value)

    # 동일 fold 결과가 중복되면 평균
    dedup = {}
    for key, values in fold_values.items():
        dedup[key] = sum(values) / len(values)

    grouped = defaultdict(dict)
    for (src, lam0, lamf, fold), value in dedup.items():
        grouped[(src, lam0, lamf)][fold] = value

    rows = []

    for (src, lam0, lamf), folds in grouped.items():
        fold_set = set(folds)

        if fold_set != expected_folds:
            print(
                f"[warn] 불완전 결과 제외: source={src}, "
                f"lam0={lam0:g}, lamf={lamf:g}, "
                f"folds={sorted(fold_set)}"
            )
            continue

        values = [folds[f] for f in sorted(expected_folds)]

        rows.append({
            "source": src,
            "lam0": lam0,
            "lamf": lamf,
            "CPD_mean": statistics.mean(values),
            "CPD_std": statistics.stdev(values),
            "n_folds": len(values),
        })

    return rows


old_rows = collect(old_dir, "existing_sweep")
new_rows = collect(new_dir, "new_lamf_0.1")

if not old_rows:
    raise SystemExit(
        f"기존 Wafer 5-fold 결과를 찾지 못했습니다: {old_dir}"
    )

if len(new_rows) != 6:
    raise SystemExit(
        f"새 Wafer 완성 조합은 6개여야 하지만 {len(new_rows)}개입니다."
    )


def best(rows):
    # CPD 높을수록 우수
    # 동률이면 std가 작은 조합, 그다음 작은 lambda
    return sorted(
        rows,
        key=lambda r: (
            -r["CPD_mean"],
            r["CPD_std"],
            r["lam0"],
            r["lamf"],
        ),
    )[0]


old_best = best(old_rows)
new_best = best(new_rows)
combined_best = best(old_rows + new_rows)

for row in old_rows + new_rows:
    row["delta_vs_existing_best"] = (
        row["CPD_mean"] - old_best["CPD_mean"]
    )
    row["better_than_existing_best"] = (
        row["CPD_mean"] > old_best["CPD_mean"]
    )

all_rows = sorted(
    old_rows + new_rows,
    key=lambda r: (
        r["source"],
        r["lamf"],
        r["lam0"],
    ),
)

os.makedirs(os.path.dirname(out_csv), exist_ok=True)

with open(out_csv, "w", newline="") as fp:
    writer = csv.DictWriter(
        fp,
        fieldnames=[
            "source",
            "lam0",
            "lamf",
            "CPD_mean",
            "CPD_std",
            "n_folds",
            "delta_vs_existing_best",
            "better_than_existing_best",
        ],
    )
    writer.writeheader()
    writer.writerows(all_rows)

delta = new_best["CPD_mean"] - old_best["CPD_mean"]

print()
print("==== 기존 lambda sweep의 Wafer 최적 ====")
print(
    f"lam0={old_best['lam0']:g}, "
    f"lamf={old_best['lamf']:g}, "
    f"CPD={old_best['CPD_mean']:.6f}"
    f"±{old_best['CPD_std']:.6f}"
)

print()
print("==== 새 lamf=0.1 후보 중 최적 ====")
print(
    f"lam0={new_best['lam0']:g}, "
    f"lamf={new_best['lamf']:g}, "
    f"CPD={new_best['CPD_mean']:.6f}"
    f"±{new_best['CPD_std']:.6f}"
)
print(f"기존 최적 대비 차이: {delta:+.6f}")

print()
print("==== 기존 + 새 후보를 합친 최종 최적 ====")
print(
    f"source={combined_best['source']}, "
    f"lam0={combined_best['lam0']:g}, "
    f"lamf={combined_best['lamf']:g}, "
    f"CPD={combined_best['CPD_mean']:.6f}"
    f"±{combined_best['CPD_std']:.6f}"
)

print()
if delta > 0:
    print("[결론] 새 lamf=0.1 조합이 기존 Wafer 최적보다 좋습니다.")
elif delta < 0:
    print("[결론] 새 lamf=0.1 조합은 기존 Wafer 최적보다 좋지 않습니다.")
else:
    print("[결론] 새 조합과 기존 최적의 평균 CPD가 같습니다.")

print(f"[saved] {out_csv}")
PY

compare_status=$?

if ((compare_status != 0)); then
  echo "[ERROR] 기존 결과와의 비교에 실패했습니다."
  exit "$compare_status"
fi

# 정상 종료
trap - EXIT INT TERM

echo
echo "============================= DONE ============================="
echo "기존 결과는 수정하지 않았습니다."
echo "새 attribution : ${NEW_NPY_DIR}/"
echo "새 CPD         : ${NEW_EVAL_DIR}/"
echo "비교 결과      : ${COMPARE_CSV}"
echo "================================================================"