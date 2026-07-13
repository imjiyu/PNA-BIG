mkdir -p results_pna/eval_anchor
mkdir -p logs/eval_anchor

# 기존 CSV에 중복 append되는 것을 방지
for data in wafer boiler epilepsy PAM; do
    rm -f "results_pna/eval_anchor/${data}.csv"
done

for data in wafer boiler epilepsy PAM; do
    for fold in 0 1 2 3 4; do
        echo "===== ${data} fold ${fold} ====="

        CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 python eval_cpd_cpp.py \
            --data "${data}" \
            --fold "${fold}" \
            --device cuda:0 \
            --mask_refs zero average pna \
            --pna_lam0 10.0 \
            --pna_lamf 10.0 \
            --pna_ka 5 \
            --anchor_idx_dir results_pna \
            --verify_anchors \
            --npy_dir results_pna_10x10 \
            --output_file "results_pna/eval_anchor/${data}.csv" \
            --methods timing_td_combined_kalman_seg50_min1_max48_lam10.0x10.0 \
            2>&1 | tee "logs/eval_anchor/${data}_f${fold}.log"
    done
done