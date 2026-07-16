#!/usr/bin/env bash
# =============================================================================
# sweep_hpo.sh
#   PNA-BIG lambda 하이퍼파라미터 튜닝 (5-fold validation).
#
#   [1] 모든 (dataset x combo x fold) 에 대해 attribution 을 새로 생성 (val split)
#   [2] 생성된 attribution 을 CPD 로 평가 (mask_ref = zero, average)
#   [3] pick_lambda.py 로 5-fold val 평균을 집계 -> 데이터셋별 최적 (lam0,lamf)
#
#   * 선택 기준(SELECTION) = "average" fill CPD 의 5-fold 평균.
#     - average / zero 는 lambda 에 무관한 fill -> 측정자가 안 움직임 (공정).
#     - pna/na 는 fill 자체가 lambda 에 의존 -> 선택에 쓰면 순환. (최종 test 표에서만 봄)
#   * Wafer 는 PNA_TUNE_VAL=1 일 때만 TRAIN 에서 val 을 잘라냄 (아래 export).
#     다른 실행/최종 test 에는 영향 없음.
#
#   실패한 작업이 있어도 나머지는 계속 진행.
# =============================================================================
set -u

# ----------------------------- 설정 -----------------------------------------
DATASETS="epilepsy wafer PAM boiler"     # 4종 전부 서치 (고정 없음)
FOLDS="0 1 2 3 4"                        # 5-fold 전체
GPU_IDS=(0 1 2 3 4)
NGPU=${#GPU_IDS[@]}
PNA_KA=5
SEG="kalman_seg0_min1_max48"            # num_segments=0
MASK_REFS_VAL="average"                 # 선택은 average, zero 는 나중에 따로 측정!
NPY_DIR="results_pna"                   # ★ main_td 저장 폴더와 반드시 동일해야 함

# 서치 grid  (lam0 lamf)
COMBOS=(
  "0.5 0.5"  "1 0.5"  "3 0.5"  "5 0.5"  "10 0.5"
  "0.5 1"    "1 1"    "3 1"    "5 1"    "10 1"
  "0.5 3"    "1 3"    "3 3"    "5 3"    "10 3"
  "0.5 5"    "1 5"    "3 5"    "5 5"    "10 5"
  "0.5 10"   "1 10"   "3 10"   "5 10"   "10 10"
  "0.5 15"   "1 15"   "3 15"   "5 15"   "10 15"
  "0.5 20"   "1 20"   "3 20"   "5 20"   "10 20"
)

bs_for() { echo 200; }   # 데이터셋별 배치 (전부 200)

# float 태그: 10 -> 10.0 , 0.5 -> 0.5  (npy 파일명 규칙과 일치시키기 위함)
float_tag() { python -c 'import sys; print(str(float(sys.argv[1])))' "$1"; }

# ★★ Wafer val 을 만들기 위한 스위치. 이 스크립트 안에서만 켜짐. ★★
export PNA_TUNE_VAL=1

mkdir -p logs "${NPY_DIR}/sweep5_eval"

# ----------------------- 공통 wave 스케줄러 ---------------------------------
pids=(); names=()
attr_fail=0; eval_fail=0
: > logs/sweep5_attr_failed.txt
: > logs/sweep5_eval_failed.txt

wait_wave() {   # $1 = ATTR|EVAL
  local kind="$1" idx
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      echo "[OK][$kind] ${names[$idx]}"
    else
      echo "[FAIL][$kind] ${names[$idx]}"
      if [ "$kind" = ATTR ]; then
        attr_fail=$((attr_fail+1)); echo "${names[$idx]}" >> logs/sweep5_attr_failed.txt
      else
        eval_fail=$((eval_fail+1)); echo "${names[$idx]}" >> logs/sweep5_eval_failed.txt
      fi
    fi
  done
  pids=(); names=()
}

# ============================================================================
# [1/3]  Attribution 생성 (val split, 5-fold)
# ============================================================================
echo; echo "================ [1/3] Attribution 생성 (5-fold val) ================"
i=0
for combo in "${COMBOS[@]}"; do
  read -r l0 lf <<< "$combo"
  for data in $DATASETS; do
    for fold in $FOLDS; do
      gpu="${GPU_IDS[$((i % NGPU))]}"
      name="${data}_f${fold}_lam${l0}x${lf}"
      CUDA_VISIBLE_DEVICES="$gpu" python -u real/main_td.py \
        --data "$data" --fold "$fold" --seed 42 \
        --explainers our_td \
        --num_segments 0 --min_seg_len 1 --max_seg_len 48 \
        --baseline pna --pna_feature hidden --pna_ka "$PNA_KA" \
        --pna_lam0 "$l0" --pna_lamf "$lf" \
        --eval_split val --model_type state \
        --device cuda:0 --testbs "$(bs_for "$data")" \
        > "logs/sweep5_attr_${name}.log" 2>&1 &
      pids+=("$!"); names+=("$name"); i=$((i+1))
      ((${#pids[@]} == NGPU)) && wait_wave ATTR
    done
  done
done
((${#pids[@]} > 0)) && wait_wave ATTR
echo "[ATTR DONE] 실패=${attr_fail}"

# --- 가드: attribution npy 가 NPY_DIR 에 실제로 저장됐는지 확인 -------------
#   (main_td 가 다른 폴더에 저장하면 여기서 잡힘)
sample_probe="${NPY_DIR}/epilepsy_state_timing_td_combined_${SEG}_val_lam$(float_tag 10)x$(float_tag 10)_result_0_42.npy"
if ! ls "${NPY_DIR}"/*_timing_td_combined_${SEG}_val_lam*_result_*_42.npy >/dev/null 2>&1; then
  echo "!!! [가드] ${NPY_DIR} 에 combined attribution npy 가 없습니다."
  echo "    main_td.py 의 저장 폴더(라인 ~1183/1201)가 '${NPY_DIR}' 인지 확인하세요."
  echo "    (현재 저장소 기본값은 ./results_our/ 입니다. 두 곳을 일치시켜야 함)"
  exit 1
fi

# ============================================================================
# [2/3]  CPD 평가 (val, mask_ref = average)
# ============================================================================
echo; echo "================ [2/3] CPD 평가 (val, average) ================"
i=0
for combo in "${COMBOS[@]}"; do
  read -r l0 lf <<< "$combo"
  l0t="$(float_tag "$l0")"; lft="$(float_tag "$lf")"
  method="timing_td_combined_${SEG}_val_lam${l0t}x${lft}"
  for data in $DATASETS; do
    for fold in $FOLDS; do
      gpu="${GPU_IDS[$((i % NGPU))]}"
      name="${data}_f${fold}_lam${l0}x${lf}"
      out="${NPY_DIR}/sweep5_eval/cmb_${name}.csv"
      rm -f "$out"                       # append 모드 -> 중복 방지 위해 삭제 후 재생성
      CUDA_VISIBLE_DEVICES="$gpu" python -u eval_cpd_cpp.py \
        --data "$data" --fold "$fold" --device cuda:0 \
        --eval_split val --npy_dir "$NPY_DIR" \
        --mask_refs $MASK_REFS_VAL \
        --cpd_only \
        --output_file "$out" \
        --methods "$method" \
        > "logs/sweep5_eval_${name}.log" 2>&1 &
      pids+=("$!"); names+=("$name"); i=$((i+1))
      ((${#pids[@]} == NGPU)) && wait_wave EVAL
    done
  done
done
((${#pids[@]} > 0)) && wait_wave EVAL
echo "[EVAL DONE] 실패=${eval_fail}"

# ============================================================================
# [3/3]  집계 + lambda 선택
# ============================================================================
echo; echo "================ [3/3] 5-fold val 집계 & lambda 선택 ================"
python pick_lambda.py --eval_dir "${NPY_DIR}/sweep5_eval" --select_ref average

echo
echo "================================ [DONE] ================================"
echo " Attribution 실패 = ${attr_fail}   (logs/sweep5_attr_failed.txt)"
echo " Evaluation  실패 = ${eval_fail}   (logs/sweep5_eval_failed.txt)"
echo " per-fold CSV     : ${NPY_DIR}/sweep5_eval/"
echo " 선택 결과        : chosen_lambdas.csv"
echo " 전체 표          : sweep5_table_average.csv"
echo "========================================================================"
