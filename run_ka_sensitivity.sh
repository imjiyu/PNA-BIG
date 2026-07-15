#!/usr/bin/env bash
# =============================================================================
# run_ka_sensitivity.sh
#
# chosen_lambdas.csv에서 데이터셋별 최적 lambda를 읽은 뒤,
# lambda를 고정하고 Ka={5,1,3,10} validation 민감도를 평가한다.
#
# Ka=5를 먼저 복사하는 이유:
# main_td.py의 출력 파일명에는 Ka가 포함되지 않으므로,
# Ka=1 등을 생성하기 전에 기존 Ka=5 결과를 보존해야 한다.
# =============================================================================
set -uo pipefail

# ----------------------------- 설정 -----------------------------------------
DATASETS=(epilepsy wafer boiler PAM)
FOLDS=(0 1 2 3 4)

# 반드시 Ka=5가 먼저 와야 기존 lambda sweep 결과를 보존할 수 있음
KA_VALUES=(5 1 3 10)

# 현재 사용 가능한 물리 GPU
GPU_IDS=(0 1 3 4 5)
NGPU=${#GPU_IDS[@]}

SEED=42
SEG="kalman_seg0_min1_max48"

# Ka 선택은 average 기준이지만 zero도 참고용으로 함께 계산
MASK_REFS_VAL=(zero average)

NPY_DIR="results_pna"
CHOSEN_CSV="chosen_lambdas.csv"
OUT_ROOT="${NPY_DIR}/ka_sensitivity"
LOG_DIR="logs/ka_sensitivity"
AGG_SCRIPT="agg_ka_sensitivity.py"

# 1: 기존 lambda sweep의 Ka=5 attribution 재사용
# 0: Ka=5도 새로 생성
REUSE_KA5=${REUSE_KA5:-1}

# Attribution 생성 배치
# PAM은 100, 나머지는 200
attr_bs_for() {
  case "$1" in
    PAM) echo 100 ;;
    *)   echo 200 ;;
  esac
}

# CPD 평가 배치
# PAM도 현재 100에서 정상 동작하므로 모두 100
eval_bs_for() {
  echo 100
}

float_tag() {
  python -c 'import sys; print(str(float(sys.argv[1])))' "$1"
}

# Wafer validation split을 기존 attribution 생성 때와 동일하게 재현
export PNA_TUNE_VAL=1

mkdir -p "$OUT_ROOT" "$LOG_DIR"

# ----------------------------- 파일 검사 ------------------------------------
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

if [[ ! -f "$CHOSEN_CSV" ]]; then
  echo "[ERROR] ${CHOSEN_CSV}가 없습니다."
  echo "        먼저 pick_lambda.py를 실행하세요."
  exit 1
fi

if [[ ! -f "$AGG_SCRIPT" ]]; then
  echo "[ERROR] ${AGG_SCRIPT}가 없습니다."
  exit 1
fi

# ----------------------- chosen lambda 읽기 ---------------------------------
declare -A LAM0
declare -A LAMF

mapfile -t chosen_lines < <(
  python - "$CHOSEN_CSV" <<'PY'
import csv
import sys

path = sys.argv[1]

with open(path, newline="") as fp:
    rows = list(csv.DictReader(fp))

if not rows:
    raise SystemExit(f"{path}: 내용이 없습니다.")

required = {"data", "lam0", "lamf"}
columns = set(rows[0].keys())

if not required.issubset(columns):
    raise SystemExit(
        f"{path}: 필요한 열 {sorted(required)}이 없습니다. "
        f"현재 열={sorted(columns)}"
    )

for row in rows:
    print(row["data"], row["lam0"], row["lamf"])
PY
)

for line in "${chosen_lines[@]}"; do
  read -r data l0 lf <<< "$line"
  LAM0["$data"]="$l0"
  LAMF["$data"]="$lf"
done

for data in "${DATASETS[@]}"; do
  if [[ -z "${LAM0[$data]+x}" || -z "${LAMF[$data]+x}" ]]; then
    echo "[ERROR] ${CHOSEN_CSV}에 ${data}의 lam0/lamf가 없습니다."
    exit 1
  fi

  echo "[LAMBDA] ${data}: lam0=${LAM0[$data]}, lamf=${LAMF[$data]}"
done

# ----------------------- 공통 상태 ------------------------------------------
pids=()
names=()

attr_fail=0
eval_fail=0

: > "${LOG_DIR}/attr_failed.txt"
: > "${LOG_DIR}/eval_failed.txt"

expected_attr_path() {
  local data="$1"
  local fold="$2"
  local l0="$3"
  local lf="$4"

  local l0t
  local lft

  l0t="$(float_tag "$l0")"
  lft="$(float_tag "$lf")"

  echo "${NPY_DIR}/${data}_state_timing_td_combined_${SEG}"\
"_val_lam${l0t}x${lft}_result_${fold}_${SEED}.npy"
}

record_failure() {
  local kind="$1"
  local name="$2"
  local file

  if [[ "$kind" == "ATTR" ]]; then
    file="${LOG_DIR}/attr_failed.txt"

    if ! grep -Fxq -- "$name" "$file"; then
      echo "$name" >> "$file"
      attr_fail=$((attr_fail + 1))
    fi
  else
    file="${LOG_DIR}/eval_failed.txt"

    if ! grep -Fxq -- "$name" "$file"; then
      echo "$name" >> "$file"
      eval_fail=$((eval_fail + 1))
    fi
  fi
}

wait_wave() {
  local kind="$1"
  local idx

  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      echo "[OK][${kind}] ${names[$idx]}"
    else
      echo "[FAIL][${kind}] ${names[$idx]}"
      record_failure "$kind" "${names[$idx]}"
    fi
  done

  pids=()
  names=()
}

# ---------------- 기존 Ka=5 파일 원위치 복원 -------------------------------
# Ka=1,3,10 생성 시 동일한 파일명을 사용하므로 results_pna 최상위의
# 기존 Ka=5 파일이 삭제된다. 스크립트 종료 시 Ka=5 복사본으로 복원한다.
restore_original_ka5() {
  local ka5_dir="${OUT_ROOT}/ka5/npy"
  local data fold l0 lf dst src

  [[ -d "$ka5_dir" ]] || return 0

  for data in "${DATASETS[@]}"; do
    [[ -n "${LAM0[$data]+x}" ]] || continue
    [[ -n "${LAMF[$data]+x}" ]] || continue

    l0="${LAM0[$data]}"
    lf="${LAMF[$data]}"

    for fold in "${FOLDS[@]}"; do
      dst="$(expected_attr_path "$data" "$fold" "$l0" "$lf")"
      src="${ka5_dir}/$(basename "$dst")"

      if [[ -s "$src" ]]; then
        cp -f "$src" "$dst"
      fi
    done
  done
}

cleanup() {
  local status="${1:-$?}"
  local pid

  trap - EXIT INT TERM

  # 실행 중인 자식 프로세스 종료
  if ((${#pids[@]} > 0)); then
    kill "${pids[@]}" 2>/dev/null || true

    for pid in "${pids[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
  fi

  restore_original_ka5
  exit "$status"
}

trap 'cleanup $?' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

# ----------------------- Ka=5 기존 결과 복사 --------------------------------
copy_existing_ka5() {
  local ka_dir="$1"
  local data fold l0 lf src dst name
  local missing=0

  mkdir -p "$ka_dir"

  for data in "${DATASETS[@]}"; do
    l0="${LAM0[$data]}"
    lf="${LAMF[$data]}"

    for fold in "${FOLDS[@]}"; do
      src="$(expected_attr_path "$data" "$fold" "$l0" "$lf")"
      dst="${ka_dir}/$(basename "$src")"
      name="${data}_f${fold}_ka5_lam${l0}x${lf}"

      # 이전 실행에서 이미 복사됐다면 재사용
      if [[ -s "$dst" ]]; then
        echo "[SKIP][COPY][Ka=5] ${name}"
        continue
      fi

      if [[ ! -s "$src" ]]; then
        echo "[MISSING][Ka=5] ${src}"
        record_failure ATTR "$name"
        missing=$((missing + 1))
        continue
      fi

      if cp -f "$src" "$dst"; then
        echo "[COPY][Ka=5] $(basename "$src")"
      else
        echo "[FAIL][COPY][Ka=5] ${src}"
        record_failure ATTR "$name"
        missing=$((missing + 1))
      fi
    done
  done

  ((missing == 0))
}

# ----------------------- Ka별 attribution 생성 -------------------------------
generate_attribution_for_ka() {
  local ka="$1"
  local ka_dir="$2"

  local data fold l0 lf gpu name src dst
  local i=0

  mkdir -p "$ka_dir"

  pids=()
  names=()

  for data in "${DATASETS[@]}"; do
    l0="${LAM0[$data]}"
    lf="${LAMF[$data]}"

    for fold in "${FOLDS[@]}"; do
      src="$(expected_attr_path "$data" "$fold" "$l0" "$lf")"
      dst="${ka_dir}/$(basename "$src")"
      name="${data}_f${fold}_ka${ka}_lam${l0}x${lf}"

      # 이전 실행에서 정상 생성된 파일은 재사용
      if [[ -s "$dst" ]]; then
        echo "[SKIP][ATTR] ${name}"
        continue
      fi

      # 이전 Ka 결과 또는 중간 실패 파일 방지
      rm -f "$src"

      gpu="${GPU_IDS[$((i % NGPU))]}"

      CUDA_VISIBLE_DEVICES="$gpu" python -u "$MAIN_SCRIPT" \
        --data "$data" \
        --fold "$fold" \
        --seed "$SEED" \
        --explainers our_td \
        --num_segments 0 \
        --min_seg_len 1 \
        --max_seg_len 48 \
        --baseline pna \
        --pna_feature hidden \
        --pna_ka "$ka" \
        --pna_lam0 "$l0" \
        --pna_lamf "$lf" \
        --eval_split val \
        --model_type state \
        --device cuda:0 \
        --testbs "$(attr_bs_for "$data")" \
        > "${LOG_DIR}/attr_${name}.log" 2>&1 &

      pids+=("$!")
      names+=("$name")
      i=$((i + 1))

      ((${#pids[@]} == NGPU)) && wait_wave ATTR
    done
  done

  ((${#pids[@]} > 0)) && wait_wave ATTR

  # main_td.py가 results_pna 최상위에 저장한 파일을 Ka별 폴더로 이동
  for data in "${DATASETS[@]}"; do
    l0="${LAM0[$data]}"
    lf="${LAMF[$data]}"

    for fold in "${FOLDS[@]}"; do
      src="$(expected_attr_path "$data" "$fold" "$l0" "$lf")"
      dst="${ka_dir}/$(basename "$src")"
      name="${data}_f${fold}_ka${ka}_lam${l0}x${lf}"

      # 이미 존재하는 정상 결과
      if [[ -s "$dst" ]]; then
        continue
      fi

      if [[ -s "$src" ]]; then
        mv -f "$src" "$dst"
      else
        echo "[MISSING][ATTR] ${name}: ${src}"
        record_failure ATTR "$name"
      fi
    done
  done
}

# ----------------------- CSV 완료 여부 검사 ---------------------------------
eval_complete() {
  local path="$1"

  [[ -s "$path" ]] || return 1

  python - "$path" "${MASK_REFS_VAL[@]}" <<'PY'
import csv
import math
import sys

path = sys.argv[1]
expected = set(sys.argv[2:])
seen = set()

try:
    with open(path, newline="") as fp:
        for row in csv.DictReader(fp):
            if row.get("metric") != "CPD":
                continue

            try:
                value = float(row["cum_diff"])
            except (KeyError, TypeError, ValueError):
                continue

            if not math.isfinite(value):
                continue

            seen.add(row.get("mask_ref", ""))
except (OSError, csv.Error):
    raise SystemExit(1)

raise SystemExit(0 if expected.issubset(seen) else 1)
PY
}

# ----------------------- Ka별 CPD 평가 --------------------------------------
evaluate_ka() {
  local ka="$1"
  local ka_dir="$2"
  local eval_dir="$3"

  local data fold l0 lf l0t lft
  local method gpu name out
  local i=0

  mkdir -p "$eval_dir"

  pids=()
  names=()

  for data in "${DATASETS[@]}"; do
    l0="${LAM0[$data]}"
    lf="${LAMF[$data]}"

    l0t="$(float_tag "$l0")"
    lft="$(float_tag "$lf")"

    method="timing_td_combined_${SEG}"\
"_val_lam${l0t}x${lft}"

    for fold in "${FOLDS[@]}"; do
      name="${data}_f${fold}_ka${ka}_lam${l0}x${lf}"
      out="${eval_dir}/cmb_${name}.csv"

      # zero와 average CPD가 모두 있으면 재사용
      if eval_complete "$out"; then
        echo "[SKIP][EVAL] ${name}"
        continue
      fi

      rm -f "$out"

      gpu="${GPU_IDS[$((i % NGPU))]}"

      CUDA_VISIBLE_DEVICES="$gpu" python -u "$EVAL_SCRIPT" \
        --data "$data" \
        --fold "$fold" \
        --seed "$SEED" \
        --device cuda:0 \
        --eval_split val \
        --testbs "$(eval_bs_for "$data")" \
        --npy_dir "$ka_dir" \
        --mask_refs "${MASK_REFS_VAL[@]}" \
        --cpd_only \
        --output_file "$out" \
        --methods "$method" \
        > "${LOG_DIR}/eval_${name}.log" 2>&1 &

      pids+=("$!")
      names+=("$name")
      i=$((i + 1))

      ((${#pids[@]} == NGPU)) && wait_wave EVAL
    done
  done

  ((${#pids[@]} > 0)) && wait_wave EVAL

  # 프로세스가 성공했더라도 CSV가 불완전한 경우 검출
  for data in "${DATASETS[@]}"; do
    l0="${LAM0[$data]}"
    lf="${LAMF[$data]}"

    for fold in "${FOLDS[@]}"; do
      name="${data}_f${fold}_ka${ka}_lam${l0}x${lf}"
      out="${eval_dir}/cmb_${name}.csv"

      if ! eval_complete "$out"; then
        echo "[MISSING][EVAL] ${name}: ${out}"
        record_failure EVAL "$name"
      fi
    done
  done
}

# =============================================================================
# 실행
# =============================================================================
for ka in "${KA_VALUES[@]}"; do
  echo
  echo "======================= Ka=${ka} ======================="

  ka_root="${OUT_ROOT}/ka${ka}"
  ka_npy="${ka_root}/npy"
  ka_eval="${ka_root}/eval"

  mkdir -p "$ka_npy" "$ka_eval"

  # eval_cpd_cpp.py가 npy_dir에서 validation index도 찾으므로 복사
  for data in "${DATASETS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      idx_src="${NPY_DIR}/${data}_state_val_idx_${fold}_${SEED}.npy"
      idx_dst="${ka_npy}/$(basename "$idx_src")"

      if [[ ! -s "$idx_src" ]]; then
        echo "[ERROR] validation index가 없습니다: ${idx_src}"
        exit 1
      fi

      cp -f "$idx_src" "$idx_dst"
    done
  done

  attr_fail_before=$attr_fail

  if [[ "$ka" -eq 5 && "$REUSE_KA5" -eq 1 ]]; then
    echo "[Ka=5] 기존 lambda sweep attribution을 재사용합니다."

    if ! copy_existing_ka5 "$ka_npy"; then
      echo "[STOP] Ka=5 attribution 재사용에 실패했습니다."
      exit 1
    fi
  else
    echo "[Ka=${ka}] validation attribution을 생성합니다."
    generate_attribution_for_ka "$ka" "$ka_npy"
  fi

  if ((attr_fail > attr_fail_before)); then
    echo "[STOP] Ka=${ka} attribution 실패가 있습니다."
    exit 1
  fi

  eval_fail_before=$eval_fail

  echo "[Ka=${ka}] CPD를 평가합니다."
  evaluate_ka "$ka" "$ka_npy" "$ka_eval"

  if ((eval_fail > eval_fail_before)); then
    echo "[STOP] Ka=${ka} CPD 평가 실패가 있습니다."
    exit 1
  fi

  echo "[Ka=${ka}] 완료"
done

# 기존 lambda sweep의 Ka=5 파일을 results_pna 최상위로 복원
restore_original_ka5

# ----------------------- 집계 ------------------------------------------------
echo
echo "==================== Ka sensitivity 집계 ===================="

if ! python "$AGG_SCRIPT" \
  --root "$OUT_ROOT" \
  --select_ref average \
  --out_dir "$OUT_ROOT"; then
  echo "[ERROR] Ka sensitivity 집계에 실패했습니다."
  exit 1
fi

# 정상 종료이므로 cleanup trap 해제
trap - EXIT INT TERM

echo
echo "============================= DONE ============================="
echo "Attribution 실패 : ${attr_fail}  (${LOG_DIR}/attr_failed.txt)"
echo "Evaluation  실패 : ${eval_fail}  (${LOG_DIR}/eval_failed.txt)"
echo "Ka별 npy         : ${OUT_ROOT}/ka*/npy/"
echo "Ka별 CSV         : ${OUT_ROOT}/ka*/eval/"
echo "집계 결과        : ${OUT_ROOT}/ka_sensitivity_*.csv"
echo "================================================================"