#!/usr/bin/env bash
set -u

LOG_ROOT="logs/viz_hidden"
OUT_ROOT="viz_hidden"

mkdir -p "${LOG_ROOT}"
mkdir -p "${OUT_ROOT}/epilepsy"
mkdir -p "${OUT_ROOT}/wafer"
mkdir -p "${OUT_ROOT}/PAM"
mkdir -p "${OUT_ROOT}/boiler"

run_dataset() {
    local data="$1"
    local gpu="$2"
    local out_dir="$3"
    local failed=0

    for fold in 0 1 2 3 4; do
        # 그림은 fold 0에서만 3개 생성하고,
        # fold 1~4에서는 정량 summary만 계산
        if (( fold == 0 )); then
            n_samples=3
        else
            n_samples=0
        fi

        log_file="${LOG_ROOT}/${data}_f${fold}.log"
        echo "[START] GPU=${gpu}, data=${data}, fold=${fold}, n_samples=${n_samples}"

        if CUDA_VISIBLE_DEVICES="${gpu}" \
            python viz/viz_hidden_path.py \
                --data "${data}" \
                --fold "${fold}" \
                --device cuda:0 \
                --pna_lam0 10.0 \
                --pna_lamf 10.0 \
                --pna_ka 5 \
                --n_alphas 50 \
                --n_samples "${n_samples}" \
                --summary_n 100 \
                --knn_k 5 \
                --out_dir "${out_dir}" \
                > "${log_file}" 2>&1
        then
            echo "[SUCCESS] GPU=${gpu}, data=${data}, fold=${fold}"
        else
            echo "[FAILED] GPU=${gpu}, data=${data}, fold=${fold}"
            echo "         log: ${log_file}"
            failed=$((failed + 1))
        fi
    done

    echo "[DATASET DONE] data=${data}, failed=${failed}"
    return "${failed}"
}

# 데이터셋 하나당 물리 GPU 하나 배정
run_dataset epilepsy 0 "${OUT_ROOT}/epilepsy" &
pid_epilepsy=$!
run_dataset wafer 1 "${OUT_ROOT}/wafer" &
pid_wafer=$!
run_dataset PAM 2 "${OUT_ROOT}/PAM" &
pid_pam=$!
run_dataset boiler 3 "${OUT_ROOT}/boiler" &
pid_boiler=$!

total_failed=0
wait "${pid_epilepsy}" || total_failed=$((total_failed + 1))
wait "${pid_wafer}"    || total_failed=$((total_failed + 1))
wait "${pid_pam}"      || total_failed=$((total_failed + 1))
wait "${pid_boiler}"   || total_failed=$((total_failed + 1))

echo "[ALL DONE] failed dataset workers=${total_failed}"