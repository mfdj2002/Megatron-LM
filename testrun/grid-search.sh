#!/bin/bash

# grid search over common hyperparameters/parallelization strategies

#first create a new directory for each run
#!/bin/bash
set -x

eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa

WORKDIR="/workspace/Megatron-LM"
HOSTFILE="../hostfile.txt"
LOGDIR="/mnt/logs"
IMAGE_NAME="mfdj2002/mds:latest"

MAX_PROFILE_TIME_PER_EXPERIMENT=4.5 #minutes
MAX_RUNTIME_PER_EXPERIMENT=5        #minutes

NNODES=$(wc -l <"$HOSTFILE")
GPUS_PER_NODE=2
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
MASTER_ADDR=$(ssh -n $(head -n 1 "$HOSTFILE") "hostname")
MASTER_PORT=6000

NUM_LAYERS=24
MICRO_BATCH_SIZE=1

CHECKPOINT_PATH=/mnt/checkpoints/gpt2_345m
VOCAB_FILE=testrun/gpt2-vocab.json
MERGE_FILE=testrun/gpt2-merges.txt
DATA_PATH=testrun/dataset/pile_gpt_train_text_document

# Function to generate powers of two up to a maximum value
powers_of_two() {
    local max_value=$1
    local n=1
    while [ $n -lt $max_value ]; do #assuming using single strategy is suboptimal
        echo $n
        n=$((n * 2))
    done
}

#--use-flash-attn \
#cannot use flash attention because only supported in A100 GPUs

FIXED_ARGS="
--recompute-activations \
--distribute-saved-activations \
--recompute-method=uniform \
--use-mcore-models \

--use-distributed-optimizer \
--overlap-grad-reduce \
--overlap-param-gather \

--no-delay-grad-reduce \
--empty-unused-memory-level=1 \

--exit-duration-in-mins $MAX_PROFILE_TIME_PER_EXPERIMENT
"

TORCHRUN_ARGS="
--nproc_per_node $GPUS_PER_NODE \
--nnodes $NNODES \
--node_rank $NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT
"

GPT_ARGS="
    --num-layers $NUM_LAYERS \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

NSYS_CMD="
nsys profile -w true \
-t cuda,nvtx,osrt,cudnn,cublas \
-s none \
-o $LOGDIR/$RUN_UID/nsys-profile-rank${NODE_RANK} \
-f true -x true
"

set_configs() {
    # Ensure the directory exists
    # mkdir -p "$LOGDIR/$RUN_UID"

    for arg in WORKDIR LOGDIR RUN_UID USE_NSYS NODE_RANK FIXED_ARGS SEARCH_ARGS TORCHRUN_ARGS GPT_ARGS DATA_ARGS OUTPUT_ARGS; do
        if [ -n "${!arg}" ]; then # Check if the variable is set and not empty
            echo "$arg='${!arg}'" | sed 's/^[ \t]*//'
        fi
    done
}

launch() {
    sudo mkdir -p orchestrator_logs/$RUN_UID
    local pids=()
    NODE_RANK=0
    while IFS= read -r addr; do
        # Execute the SSH command in the background
        (set_configs | ssh "$addr" \
            "
            sudo systemctl stop docker
            sudo mount /dev/sda4 /mnt
            sudo systemctl start docker
            sudo modprobe nvidia-peermem
            sudo mkdir -p $LOGDIR/$RUN_UID
            cat >> $LOGDIR/$RUN_UID/configs.env

            image_exists=\$(docker images -q $IMAGE_NAME)
            if [[ -z \"\$image_exists\" ]]; then
                docker pull $IMAGE_NAME
            fi

            timeout ${MAX_RUNTIME_PER_EXPERIMENT}m docker run --privileged --gpus all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                --env-file $LOGDIR/$RUN_UID/configs.env --stop-timeout 30\
                -v ~/Megatron-LM:/workspace/Megatron-LM -v /mnt:/mnt $IMAGE_NAME
            " >orchestrator_logs/$RUN_UID/ssh_node$NODE_RANK.log 2>&1) &

        pids+=("$!")
        NODE_RANK=$((NODE_RANK + 1))
    done <"$HOSTFILE"

    local failure_flag=0
    for idx in "${!pids[@]}"; do
        pid=${pids[$idx]}
        if ! wait "$pid"; then
            addr=$(awk "NR==$((idx + 1))" "$HOSTFILE")
            echo "Subprocess with address $addr in RUN $RUN_UID failed."
            failure_flag=1
        fi
    done

    if [ "$failure_flag" -eq 1 ]; then
        return 1
    else
        echo "All subprocesses completed successfully in RUN $RUN_UID."
        return 0
    fi

}

# launch() {
#     counter=0
#     while IFS= read -r addr; do
#         set_configs | ssh "$addr" \
#             "
#             sudo systemctl stop docker
#             sudo mount /dev/sda4 /mnt
#             sudo systemctl start docker
#             sudo modprobe nvidia-peermem
#             sudo mkdir -p $LOGDIR/$RUN_UID && \
#             cat >> $LOGDIR/$RUN_UID/configs.env;

#             image_exists=$(docker images -q '$IMAGE_NAME')

#             if [[ -z \"$image_exists\" ]]; then
#             docker pull $IMAGE_NAME
#             fi

#             docker run -d --privileged --gpus all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
#                 --env-file $LOGDIR/$RUN_UID/configs.env \
#                 -v ~/Megatron-LM:/workspace/Megatron-LM -v /mnt:/mnt $IMAGE_NAME
#             if [ $USE_NSYS -eq 0]; then
#                 nohup bash $WORKDIR/testrun/basic-profiling.sh &
#             fi
#             exit 0
#             "
#         status=$?
#         if [ $status -eq 0 ]; then
#             echo "Starting launch to node $addr in RUN $RUN_UID successful."
#         else
#             echo "Starting launch on node $addr in RUN $RUN_UID failed. Status: $status."
#         fi
#         counter=$((counter + 1))
#     done <"$HOSTFILE"
# }

# monitor_and_cleanup() {
#     local check_interval=30 # How often to check the containers, in seconds.

#     while :; do
#         local all_running=true

#         for detail in "${details[@]}"; do
#             # Split the detail string into its components.
#             IFS=' ' read -r addr docker_id bash_pid <<<"$detail"

#             # Check if the Docker container is still running.
#             if ssh "$addr" "[[ \$(docker ps -q -f id=\"$docker_id\") == \"\" ]]"; then
#                 all_running=false
#                 break # Exit the loop early if any container has stopped.
#             fi
#         done

#         if [ "$all_running" = false ]; then
#             echo "A container has stopped. Initiating cleanup..."

#             for detail in "${details[@]}"; do
#                 IFS=' ' read -r addr docker_id bash_pid <<<"$detail"

#                 # Stop the Docker container.
#                 ssh "$addr" "docker stop \"$docker_id\""

#                 # If a Bash PID exists, kill it.
#                 if [ -n "$bash_pid" ]; then
#                     ssh "$addr" "kill \"$bash_pid\" 2>/dev/null || true"
#                 fi
#             done

#             echo "Cleanup completed."
#             return 0 # Exit the function successfully after cleanup.
#         fi

#         # Wait for the next check.
#         sleep "$check_interval"
#     done
# }

SEARCH_ARGS=""
# Main search loop with boolean parameters included
for context_size in $(powers_of_two $WORLD_SIZE); do
    for pipeline_size in $(powers_of_two $WORLD_SIZE); do
        for tensor_size in $(powers_of_two $WORLD_SIZE); do
            for ((layers_per_virtual_stage = 1; layers_per_virtual_stage < $NUM_LAYERS; layers_per_virtual_stage++)); do
                for sequence_parallel in 0 1; do
                    for no_clone_scatter_output_in_embedding in 0 1; do
                        for cpu_init in 0 1; do
                            for USE_NSYS in 0 1; do
                                if [ $((context_size * pipeline_size * tensor_size)) -le $WORLD_SIZE ]; then
                                    # Check if the combination fits the layers constraint
                                    if [ $virtual_stages -gt 1]; then
                                        if [ $pipeline_size -gt 2]; then
                                            if [ $((virtual_stages * pipeline_size)) -le $NUM_LAYERS ]; then
                                                RUN_UID=$(date +%Y%m%d%H%M%S)
                                                SEARCH_ARGS+="--context-parallel-size $context_size --pipeline-model-parallel-size $pipeline_size --tensor-model-parallel-size $tensor_size --num-layers-per-virtual-pipeline-stage $layers_per_virtual_stage"
                                                if [ $sequence_parallel -eq 1 ]; then
                                                    SEARCH_ARGS+="--sequence-parallel"
                                                fi
                                                if [ $no_clone_scatter_output_in_embedding -eq 1]; then
                                                    SEARCH_ARGS+="--no-clone-scatter-output-in-embedding"
                                                fi
                                                if [ $cpu_init -eq 1]; then
                                                    SEARCH_ARGS+="--use-cpu-initialization"
                                                fi
                                                launch
                                                sleep 5
                                            fi
                                        fi
                                    else #dont set virtual stages argument..
                                        RUN_UID=$(date +%Y%m%d%H%M%S)
                                        SEARCH_ARGS+="--context-parallel-size $context_size --pipeline-model-parallel-size $pipeline_size --tensor-model-parallel-size $tensor_size"
                                        if [ $sequence_parallel -eq 1 ]; then
                                            SEARCH_ARGS+="--sequence-parallel"
                                        fi
                                        if [ $no_clone_scatter_output_in_embedding -eq 1]; then
                                            SEARCH_ARGS+="--no-clone-scatter-output-in-embedding"
                                        fi
                                        if [ $cpu_init -eq 1]; then
                                            SEARCH_ARGS+="--use-cpu-initialization"
                                        fi
                                        launch
                                        sleep 5
                                    fi
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done

eval $(ssh-agent -k)

## GPT-3 Small 125M
# model_size=0.125
# num_layers=12
# hidden_size=768
# num_attn_heads=12
# global_batch_size=256
# lr=6.0e-4
# min_lr=1.0e-6
# init_std=0.02

## GPT-3 Medium 350M
# model_size=0.35
# num_layers=24
# hidden_size=1024
# num_attn_heads=16
# global_batch_size=256
# lr=3.0e-4
# min_lr=1.0e-6
# init_std=0.018

## GPT-3 Large 760M
# model_size=0.76
# num_layers=24
# hidden_size=1536
# num_attn_heads=16
# global_batch_size=256
# lr=2.5e-4
# min_lr=1.0e-6
# init_std=0.015

## GPT-3 XL 1.3B
# model_size=1.3
# num_layers=24
# hidden_size=2048
# num_attn_heads=16
# global_batch_size=512
# lr=2.0e-4
# min_lr=1.0e-6
# init_std=0.013

## GPT-3 2.7B
# model_size=2.7
# num_layers=32
# hidden_size=2560
# num_attn_heads=32
# global_batch_size=512
# lr=1.6e-4
# min_lr=1.0e-6
# init_std=0.011

## GPT-3 6.7B
# model_size=6.7
# num_layers=32
# hidden_size=4096
# num_attn_heads=32
# global_batch_size=1024
# lr=1.2e-4
# min_lr=1.0e-6
# init_std=0.009

## GPT-3 13B
# model_size=13
# num_layers=40
# hidden_size=5120
# num_attn_heads=40
# global_batch_size=1024
# lr=1.0e-4
# min_lr=1.0e-6
# init_std=0.008
