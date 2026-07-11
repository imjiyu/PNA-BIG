mkdir -p logs

python ood_experiments/main_td.py \
  --model_type state --data epilepsy --explainers our_td \
  --fold 4 --seed 42 --device cuda:4 \
  --num_segments 10 --min_seg_len 10 --max_seg_len 10 \
  --testbs 150 \
  2>&1 | tee logs/epilepsy_fold4_gpu4_testbs200.log | tail -5