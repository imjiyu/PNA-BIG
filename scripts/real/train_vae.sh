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


for data in epilepsy freezer PAM wafer boiler
do
    for cv in 0 1 2 3 4
    do
        CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/train_vae.py \
            --data $data \
            --fold $cv \
            --seed 42 \
            --device cuda:0 \
            2>&1 &
        wait_n
        i=$((i + 1))
    done
done
