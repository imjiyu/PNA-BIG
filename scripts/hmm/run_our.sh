wait_n() {
    background=($(jobs -p))
    echo ${num_max_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}

GPUS=(4 5 6)
NUM_GPUS=${#GPUS[@]}
i=0
num_max_jobs=3

# boiler epilepsy
for cv in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python synthetic/hmm/main.py \
        --explainers our \
        --fold $cv \
        --device cuda:0 \
        --output-file result/hmm_${cv}_results.csv \
        2>&1 &
    wait_n
    i=$((i + 1))
done
