#!/usr/bin/env bash
# run_pna_eval.sh
# PNA-BIG 평가: (1) |T+R| vs |T|+|R|  (2) Trend vs Residual dominance
set -u
SEG="kalman_seg50_min1_max48"
LAM="_lam10.0x10.0"
ROOT="results_pna"
mkdir -p ${ROOT}/eval_combined ${ROOT}/eval_dominance logs/pna_eval

# --- 재실행 대비: 이전 결과 삭제 (append 모드 주의) ---
rm -f ${ROOT}/eval_combined/*.csv ${ROOT}/eval_dominance/*.csv

idx=0
run_eval() {   # $1=subdir $2=prefix  $3,$4=methods
  local sub="$1" pref="$2" m1="$3" m2="$4"
  for data in epilepsy wafer PAM boiler; do
    for fold in 0 1 2 3 4; do
      gpu=$((idx % 4))
      CUDA_VISIBLE_DEVICES=$gpu nohup python eval_cpd_cpp.py \
        --data "$data" --fold "$fold" --device cuda:0 \
        --npy_dir ${ROOT} \
        --output_file "${ROOT}/${sub}/${pref}_${data}_f${fold}.csv" \
        --methods "${m1}" "${m2}" \
        > "logs/pna_eval/${pref}_${data}_f${fold}.log" 2>&1 &
      idx=$((idx+1)); (( idx % 4 == 0 )) && wait
    done
  done
  wait
}

# (1) Aggregation: |T+R| vs |T|+|R|
run_eval eval_combined combined \
  "timing_td_combined_${SEG}${LAM}" \
  "timing_td_T_plus_R_${SEG}${LAM}"

# (2) Dominance: Trend vs Residual
run_eval eval_dominance full \
  "timing_td_trend_${SEG}${LAM}" \
  "timing_td_residual_${SEG}${LAM}"

# --- fold별 CSV 병합 ---
awk 'FNR==1 && NR!=1 {next} {print}' ${ROOT}/eval_combined/combined_*.csv \
  > ${ROOT}/eval_combined/combined_eval.csv
awk 'FNR==1 && NR!=1 {next} {print}' ${ROOT}/eval_dominance/full_*.csv \
  > ${ROOT}/eval_dominance/full_eval.csv
echo "[DONE] 평가 CSV 병합 완료"
echo "  - ${ROOT}/eval_combined/combined_eval.csv  (|T+R| vs |T|+|R|)"
echo "  - ${ROOT}/eval_dominance/full_eval.csv      (Trend vs Residual)"
echo ""
echo "다음: 표 조립"
echo "  ① python (README의 cpd_mean_std 스니펫)"
echo "  ② python TR_table.py --results_dir results_pna \\"
echo "       --eval_csv results_pna/eval_dominance/full_eval.csv \\"
echo "       --out_dir results_pna/eval_dominance --folds 0 1 2 3 4"
