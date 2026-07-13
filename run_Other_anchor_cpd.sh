mkdir -p results_pna/eval_anchor logs/eval_anchor

declare -A TIMING_NAME=(
    [wafer]="timing_sample100_seg5_min10_max152"
    [boiler]="timing_sample100_seg50_min1_max36"
    [epilepsy]="timing_sample100_seg10_min10_max10"
    [PAM]="timing_sample100_seg10_min10_max600"
)
BASE=(augmented_occlusion gate_mask gradientshap_abs timex timex++ integrated_gradients_base_abs)

for data in wafer boiler epilepsy PAM; do
    METHODS=("${BASE[@]}" "${TIMING_NAME[$data]}")
    rm -f results_pna/eval_anchor/${data}_baselines_f*.csv

    for fold in 0 1 2 3 4; do
        GPU=$((fold + 1))          # fold 0→gpu1 ... fold 4→gpu5
        echo "===== ${data} fold ${fold} on GPU ${GPU} ====="
        CUDA_VISIBLE_DEVICES=${GPU} PYTHONUNBUFFERED=1 python eval_cpd_cpp.py \
            --data "${data}" \
            --fold "${fold}" \
            --device cuda:0 \
            --mask_refs zero average pna \
            --pna_lam0 10.0 \
            --pna_lamf 10.0 \
            --pna_ka 5 \
            --anchor_idx_dir results_pna \
            --verify_anchors \
            --npy_dir results_our \
            --output_file "results_pna/eval_anchor/${data}_baselines_f${fold}.csv" \
            --methods "${METHODS[@]}" \
            2>&1 | tee "logs/eval_anchor/${data}_baselines_f${fold}.log" &
    done
    wait   # 한 데이터셋의 5 fold 다 끝나면 다음 데이터셋
    cat results_pna/eval_anchor/${data}_baselines_f*.csv \
        > results_pna/eval_anchor/${data}_baselines.csv
done