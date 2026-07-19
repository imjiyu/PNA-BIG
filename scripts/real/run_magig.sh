wait_n() {
    background=($(jobs -p))
    echo ${num_max_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}

GPUS=(0 1 2 3 4 5 6)
NUM_GPUS=${#GPUS[@]}
i=0
num_max_jobs=7

FOLD=0
SEED=42

for data in epilepsy
do
    CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
        --explainers magig \
        --data $data \
        --fold $FOLD --seed $SEED \
        --device cuda:0 \
        --model_type state \
        --testbs 64 \
        --output-file magig_${data}.csv \
        2>&1 &
    wait_n
    i=$((i + 1))
done
