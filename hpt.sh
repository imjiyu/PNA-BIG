#!/bin/bash

mkdir -p logs
mkdir -p results_pna
mkdir -p results_pna/eval_tmp

# CSV만 정리. npy는 건드리지 않음.
rm -f results_pna/eval_tmp/*_val_cpd_combined_kalman_*.csv
rm -f results_pna/boiler_val_cpd_combined_kalman.csv
rm -f results_pna/epilepsy_val_cpd_combined_kalman.csv

bs_for() {
  case $1 in
    epilepsy) echo 5;;
    boiler) echo 30;;
    PAM) echo 3;;
    *) echo 5;;
  esac
}

i=0
NGPU=8

for data in boiler epilepsy; do
  bs=$(bs_for $data)

  for l0 in 0.1 1.0 10.0; do
    for lf in 0.1 1.0 10.0; do
      gpu=$(( i % NGPU ))

      method="timing_td_combined_kalman_seg0_min1_max48_val_lam${l0}x${lf}"
      out_csv="results_pna/eval_tmp/${data}_val_cpd_combined_kalman_lam${l0}x${lf}.csv"
      log_file="logs/eval_${data}_lam${l0}x${lf}.log"

      echo "[RUN] data=${data}, lam=${l0}x${lf}, gpu=${gpu}"

      CUDA_VISIBLE_DEVICES=$gpu python -u eval_cpd_cpp.py \
        --data $data \
        --fold 0 \
        --seed 42 \
        --model_type state \
        --device cuda:0 \
        --testbs $bs \
        --npy_dir results_pna \
        --output_file $out_csv \
        --eval_split val \
        --topk 0.1 \
        --top 0 \
        --methods $method \
        > $log_file 2>&1 &

      i=$(( i + 1 ))
      (( i % NGPU == 0 )) && wait
    done
  done
done

wait

echo "[MERGE] merging csv files"

first=1
for f in results_pna/eval_tmp/boiler_val_cpd_combined_kalman_lam*.csv; do
  if [ $first -eq 1 ]; then
    cat "$f" > results_pna/boiler_val_cpd_combined_kalman.csv
    first=0
  else
    tail -n +2 "$f" >> results_pna/boiler_val_cpd_combined_kalman.csv
  fi
done

first=1
for f in results_pna/eval_tmp/epilepsy_val_cpd_combined_kalman_lam*.csv; do
  if [ $first -eq 1 ]; then
    cat "$f" > results_pna/epilepsy_val_cpd_combined_kalman.csv
    first=0
  else
    tail -n +2 "$f" >> results_pna/epilepsy_val_cpd_combined_kalman.csv
  fi
done

echo "8-GPU CPD evaluation done"
echo "boiler result:   results_pna/boiler_val_cpd_combined_kalman.csv"
echo "epilepsy result: results_pna/epilepsy_val_cpd_combined_kalman.csv"