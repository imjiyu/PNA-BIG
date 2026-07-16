#!/usr/bin/env bash
# =============================================================================
# run_all.sh — PNA-BIG 최종(test) 파이프라인 원스톱 실행
#
#   bash run_all.sh all         # Step 1 → 4 전부
#   bash run_all.sh attr        # Step 1  : PNA-BIG attribution (+ anchor idx 저장)
#   bash run_all.sh comp        # Step 2  : completeness (PNA-BIG vs TIMING)
#   bash run_all.sh baseattr    # Step 3b-1 : baseline 6종 attribution + zero/avg CPD
#   bash run_all.sh cpd         # Step 3a : PNA-BIG anchor-masking CPD (zero/average/pna)
#   bash run_all.sh baselines   # Step 3b-2 : baseline 7종 CPD (pna fill)
#   bash run_all.sh dominance   # Step 3c : Trend vs Residual (zero fill)
#   bash run_all.sh tables      # Step 4  : 표 생성
#
# 하이퍼파라미터(λ0, λf, Ka, TIMING npy 이름)는 hp_pna.sh 하나에서만 읽는다.
#
# 환경변수:
#   GPUS="1 2 3 4 5"   fold 0~4 를 매핑할 물리 GPU (기본: 1 2 3 4 5)
#   DATASETS / FOLDS
#   ATTR_DIR           PNA-BIG attribution 저장 폴더 (기본: results_pna_hpt)
#   BASE_DIR           baseline / TIMING attribution 폴더 (기본: results_our)
#
# 예)  GPUS="0 1 2 3 4" bash run_all.sh all 2>&1 | tee logs/run_all.log
# =============================================================================
set -uo pipefail

[[ -f hp_pna.sh ]] || { echo "[ERROR] hp_pna.sh 가 없습니다. repo 루트에서 실행하세요."; exit 1; }
source ./hp_pna.sh          # KA / L0 / LF / TIMING_NAME

DATASETS=${DATASETS:-"wafer boiler epilepsy PAM"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
read -r -a GPU_IDS <<< "${GPUS:-1 2 3 4 5}"

SEED=42
SEG="kalman_seg50_min1_max48"       # PNA-BIG 파일명 태그 (결과에 영향 없는 호환용 값)
ATTR_DIR=${ATTR_DIR:-results_pna_hpt}
BASE_DIR=${BASE_DIR:-results_our}
ANCHOR_DIR=results_pna              # main_td.py 가 pool/anchoridx 를 항상 여기에 저장(하드코딩)

EVAL_DIR="${ATTR_DIR}/eval_anchor"
DOM_DIR="${ATTR_DIR}/eval_dominance"

mkdir -p logs/pna_hpt logs/eval_anchor logs/eval_dom \
         "$ATTR_DIR" "$EVAL_DIR" "$DOM_DIR" "$ANCHOR_DIR"

gpu_for() { echo "${GPU_IDS[$(( $1 % ${#GPU_IDS[@]} ))]}"; }
hdr()     { echo; echo "================ $* ================"; }

# -----------------------------------------------------------------------------
# Step 1 — PNA-BIG attribution (test split) + anchor index 저장
# -----------------------------------------------------------------------------
stage_attr() {
  hdr "Step 1: PNA-BIG attribution → ${ATTR_DIR}/"
  for d in $DATASETS; do
    for f in $FOLDS; do
      gpu=$(gpu_for "$f")
      echo "[START] ${d} fold${f} → GPU ${gpu}  (lam=${L0[$d]}x${LF[$d]}, Ka=${KA[$d]})"
      SAVE_ANCHOR_IDX=1 SAVE_DIR="./${ATTR_DIR}" CUDA_VISIBLE_DEVICES="$gpu" \
        nohup python real/main_td.py --explainers our_td \
          --data "$d" --fold "$f" --seed $SEED \
          --baseline pna --model_type state --eval_split test \
          --pna_lam0 "${L0[$d]}" --pna_lamf "${LF[$d]}" --pna_ka "${KA[$d]}" \
          --num_segments 50 --min_seg_len 1 --max_seg_len 48 \
          --device cuda:0 --testbs 200 \
          > "logs/pna_hpt/${d}_f${f}.log" 2>&1 &
    done
    wait
  done

  hdr "Step 1 검증"
  echo "combined npy 개수 : $(ls ${ATTR_DIR}/*combined* 2>/dev/null | wc -l)   (기대: 20)"
  echo "anchor 미저장 로그: $(grep -L 'SAVE_ANCHOR_IDX. saved' logs/pna_hpt/*.log | tr '\n' ' ')   (기대: 없음)"
  echo "에러 로그         : $(grep -il 'error\|traceback' logs/pna_hpt/*.log | tr '\n' ' ')   (기대: 없음)"
}

# -----------------------------------------------------------------------------
# Step 2 — Completeness 진단 (PNA-BIG vs TIMING)
#   run_timing_comp.sh 는 --explainers our timing_comp --baseline zero 라서
#   TIMING attribution(timing_sample100_seg*) 과 completeness 버전을 한 번에 뽑는다.
#   → Step 3b 에서 쓰는 TIMING npy 도 여기서 생성된다.
# -----------------------------------------------------------------------------
stage_comp() {
  hdr "Step 2a: PNA-BIG completeness"
  python check_completeness.py --data boiler epilepsy wafer PAM \
    --results-dir "./${ATTR_DIR}" | tee completeness_pna_hpt.txt

  hdr "Step 2b: TIMING attribution + completeness 버전 생성 (→ ${BASE_DIR}/)"
  bash run_timing_comp.sh

  hdr "Step 2c: TIMING completeness"
  python check_completeness_timing.py --results-dir "./${BASE_DIR}" \
    > completeness_timing.txt 2>&1
  tail -n 12 completeness_timing.txt

  hdr "Step 2d: 표 생성 → completeness_tables/"
  python make_comp_table.py \
    --pna-file completeness_pna_hpt.txt \
    --timing-file completeness_timing.txt
}

# -----------------------------------------------------------------------------
# Step 3b-1 — baseline 6종 attribution + zero/average CPD
#   → results_our/*.npy  및  루트의 state_{data}_{fold}_0_results_baseline.csv
# -----------------------------------------------------------------------------
stage_baseattr() {
  hdr "Step 3b-1: baseline 6종 attribution + zero/avg CPD (100 step)"
  bash scripts/real/run_10perc_masking_6xai.sh
  echo
  echo "CSV 헤더 순서:"
  echo "  seed,fold,baseline,area,explainer,lambda_1,lambda_2,lambda_3,cum_50_diff,cum_diff,AUCC,accuracy,comprehensiveness,cross_entropy,log_odds,sufficiency"
}

# -----------------------------------------------------------------------------
# Step 3a — PNA-BIG anchor-masking CPD (mask_ref = zero / average / pna)
# -----------------------------------------------------------------------------
stage_cpd() {
  hdr "Step 3a: PNA-BIG CPD (zero / average / pna) → ${EVAL_DIR}/"
  for d in $DATASETS; do
    rm -f "${EVAL_DIR}/${d}_f"*.csv "${EVAL_DIR}/${d}.csv"   # append 모드 → 사전 삭제 필수
    for f in $FOLDS; do
      gpu=$(gpu_for "$f")
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
        nohup python eval_cpd_cpp.py \
          --data "$d" --fold "$f" --device cuda:0 \
          --mask_refs zero average pna \
          --pna_lam0 "${L0[$d]}" --pna_lamf "${LF[$d]}" --pna_ka "${KA[$d]}" \
          --anchor_idx_dir "$ANCHOR_DIR" --verify_anchors --anchor_chunk 200 \
          --npy_dir "$ATTR_DIR" \
          --methods "timing_td_combined_${SEG}_lam${L0[$d]}x${LF[$d]}" \
          --output_file "${EVAL_DIR}/${d}_f${f}.csv" \
          > "logs/eval_anchor/${d}_f${f}.log" 2>&1 &
    done
    wait
    awk 'FNR==1 && NR!=1 {next} {print}' "${EVAL_DIR}/${d}_f"*.csv > "${EVAL_DIR}/${d}.csv"
    echo "[MERGED] ${EVAL_DIR}/${d}.csv"
  done

  hdr "anchor 재현 검증 (전부 0.0 이어야 정상)"
  grep -h "max|loaded" logs/eval_anchor/*.log | sort -u
}

# -----------------------------------------------------------------------------
# Step 3b-2 — baseline 7종 CPD (pna fill 만; zero/average 는 Step 3b-1 산출물 사용)
# -----------------------------------------------------------------------------
stage_baselines() {
  hdr "Step 3b-2: baseline 7종 CPD (pna fill) → ${EVAL_DIR}/*_baselines_pna.csv"
  for d in $DATASETS; do
    rm -f "${EVAL_DIR}/${d}_baselines_pna_f"*.csv "${EVAL_DIR}/${d}_baselines_pna.csv"
    for f in $FOLDS; do
      gpu=$(gpu_for "$f")
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
        nohup python eval_cpd_cpp.py \
          --data "$d" --fold "$f" --device cuda:0 \
          --mask_refs pna \
          --testbs 200 \
          --pna_lam0 "${L0[$d]}" --pna_lamf "${LF[$d]}" --pna_ka "${KA[$d]}" \
          --anchor_idx_dir "$ANCHOR_DIR" --verify_anchors --anchor_chunk 200 \
          --npy_dir "$BASE_DIR" \
          --methods augmented_occlusion gate_mask gradientshap_abs timex timex++ \
                    integrated_gradients_base_abs "${TIMING_NAME[$d]}" \
          --output_file "${EVAL_DIR}/${d}_baselines_pna_f${f}.csv" \
          > "logs/eval_anchor/${d}_base_f${f}.log" 2>&1 &
    done
    wait
    # ★ aggregate_xai_results.py 가 찾는 이름: {data}_baselines_pna.csv
    awk 'FNR==1 && NR!=1 {next} {print}' "${EVAL_DIR}/${d}_baselines_pna_f"*.csv \
      > "${EVAL_DIR}/${d}_baselines_pna.csv"
    echo "[MERGED] ${EVAL_DIR}/${d}_baselines_pna.csv"
  done
}

# -----------------------------------------------------------------------------
# Step 3c — Trend vs Residual dominance
# -----------------------------------------------------------------------------
stage_dominance() {
  hdr "Step 3c: Trend vs Residual → ${DOM_DIR}/"
  rm -f "${DOM_DIR}"/*.csv
  for d in $DATASETS; do
    for f in $FOLDS; do
      gpu=$(gpu_for "$f")
      S="${SEG}_lam${L0[$d]}x${LF[$d]}"
      CUDA_VISIBLE_DEVICES="$gpu" \
        nohup python eval_cpd_cpp.py \
          --mask_refs zero average pna \
          --data "$d" --fold "$f" --device cuda:0 \
          --testbs 200 \ 
          --pna_lam0 "${L0[$d]}" --pna_lamf "${LF[$d]}" --pna_ka "${KA[$d]}" \
          --anchor_idx_dir "$ANCHOR_DIR" --verify_anchors --anchor_chunk 200 \
          --npy_dir "$ATTR_DIR" \
          --output_file "${DOM_DIR}/full_${d}_f${f}.csv" \
          --methods "timing_td_trend_${S}" "timing_td_residual_${S}" \
          > "logs/eval_dom/${d}_f${f}.log" 2>&1 &
    done
    wait
  done
  awk 'FNR==1 && NR!=1 {next} {print}' "${DOM_DIR}"/full_*.csv > "${DOM_DIR}/full_eval.csv"
  echo "[MERGED] ${DOM_DIR}/full_eval.csv"
}

# -----------------------------------------------------------------------------
# Step 4 — 표 생성
# -----------------------------------------------------------------------------
stage_tables() {
  hdr "Step 4a: Trend vs Residual 표"
  python TR_table.py --results_dir "$ATTR_DIR" \
    --eval_csv "${DOM_DIR}/full_eval.csv" \
    --out_dir "$DOM_DIR" --folds 0 1 2 3 4

  hdr "Step 4b: 8-method CPD 통합 (raw)"
  awk 'FNR==1 && NR!=1 {next} {print}' \
    "${EVAL_DIR}"/{wafer,boiler,epilepsy,PAM}{,_baselines_pna}.csv \
    > "${EVAL_DIR}/all_8methods.csv"
  echo "[MERGED] ${EVAL_DIR}/all_8methods.csv"

  hdr "Step 4c: 최종 CPD 표 → aggregated_results/"
  python aggregate_xai_results.py --root .
  echo
  echo "warnings.txt 를 반드시 확인하세요: aggregated_results/warnings.txt"
}

# -----------------------------------------------------------------------------
case "${1:-all}" in
  attr)       stage_attr ;;
  comp)       stage_comp ;;
  baseattr)   stage_baseattr ;;
  cpd)        stage_cpd ;;
  baselines)  stage_baselines ;;
  dominance)  stage_dominance ;;
  tables)     stage_tables ;;
  all)        stage_attr; stage_comp; stage_baseattr; stage_cpd
              stage_baselines; stage_dominance; stage_tables ;;
  *)          echo "usage: bash run_all.sh {all|attr|comp|baseattr|cpd|baselines|dominance|tables}"; exit 1 ;;
esac

hdr "DONE: $1"
