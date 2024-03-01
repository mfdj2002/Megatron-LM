#!/bin/bash

# grid search over common hyperparameters/parallelization strategies

set -x

eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa

WORKDIR="/workspace/Megatron-LM"
HOSTFILE="../hostfile.txt"
LOGDIR="/mnt/logs"
IMAGE_NAME="mfdj2002/mds:latest"

ORCHESTRATOR_HOSTNAME=$(hostname)
ADDR_SUFFIX="${ORCHESTRATOR_HOSTNAME#*.}"

ORCHESTRATOR_LOGDIR="$HOME/orchestrator_logs-at-${ORCHESTRATOR_HOSTNAME}"

MAX_RUNTIME_PER_EXPERIMENT=5 #minutes

NNODES=$(wc -l <"$HOSTFILE")
# NNODES=4
GPUS_PER_NODE=2
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
MASTER_ADDR=$(ssh -n $(head -n 1 "$HOSTFILE") "hostname")
# MASTER_ADDR="0.0.0.0"
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

--exit-duration-in-mins $MAX_RUNTIME_PER_EXPERIMENT
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



# set_configs() {
#     # Ensure the directory exists
#     mkdir -p "test_configs/$RUN_UID"

#     for arg in WORKDIR LOGDIR RUN_UID USE_NSYS NSYS_CMD NODE_RANK FIXED_ARGS SEARCH_ARGS TORCHRUN_ARGS GPT_ARGS DATA_ARGS OUTPUT_ARGS; do
#         if [ -n "${!arg}" ]; then # Check if the variable is set and not empty
#             echo "$arg='${!arg}'" | sed 's/^[ \t]*//' >>"test_configs/$RUN_UID/configs.env"
#         fi
#     done
# }


env_vars=("WORKDIR" "LOGDIR" "RUNNAME" "USE_NSYS" "NSYS_CMD" "NODE_RANK" "MAX_RUNTIME_PER_EXPERIMENT" "FIXED_ARGS" "SEARCH_ARGS" "TORCHRUN_ARGS" "GPT_ARGS" "DATA_ARGS" "OUTPUT_ARGS")

launch() {
    mkdir -p $ORCHESTRATOR_LOGDIR/$RUNNAME
    local pids=()
    NODE_RANK=0
    TORCHRUN_ARGS="
        --nproc_per_node $GPUS_PER_NODE \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT
        "
    NSYS_CMD="
        nsys profile -w true \
        -t cuda,nvtx,osrt,cudnn,cublas \
        -s none \
        -o $LOGDIR/$RUNNAME/nsys-profile-rank${NODE_RANK} \
        -f true -x true
        "
    while IFS= read -r addr; do
        docker_cmd="docker run"
        for var in "${env_vars[@]}"; do
            docker_cmd+=" -e $var=\"${!var}\""
        done
        docker_cmd+=" --privileged --gpus all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ~/Megatron-LM:/workspace/Megatron-LM -v /mnt:/mnt --rm $IMAGE_NAME"

        echo $docker_cmd > $ORCHESTRATOR_LOGDIR/$RUNNAME/master_docker_command.txt
        # Execute the SSH command in the background
        ssh -n "$addr" \
            "
            mkdir -p $LOGDIR/$RUNNAME
            sudo systemctl stop docker
            sudo mount /dev/sda4 /mnt
            sudo systemctl start docker
            sudo modprobe nvidia-peermem

            if [[ -z \"\$(docker images -q $IMAGE_NAME)\" ]]; then
                docker pull $IMAGE_NAME
            fi

            if [ \"$USE_NSYS\" -eq 1 ]; then
                nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw,pstate,pcie.link.gen.max,pcie.link.gen.current --format=csv -l 5 >"$LOGDIR/$RUNNAME/nvidia-smi-rank${NODE_RANK}.csv" &
                nvidia_smi_pid=$!
                dool --more --output "$LOGDIR/$RUNNAME/dool-rank${NODE_RANK}.csv" 5 &
                dool_pid=$!
            fi

            \$docker_cmd

            if [ \"$USE_NSYS\" -eq 1 ]; then
                kill $nvidia_smi_pid $dool_pid 2>/dev/null || true
            fi
                
        " >$ORCHESTRATOR_LOGDIR/$RUNNAME/ssh_node$NODE_RANK.log 2>&1 &


        
        pids+=("$!")
        NODE_RANK=$((NODE_RANK + 1))
    done <"$HOSTFILE"

    local failure_flag=0
    for idx in "${!pids[@]}"; do
        pid=${pids[$idx]}
        if ! wait "$pid"; then
            addr=$(awk "NR==$((idx + 1))" "$HOSTFILE")
            echo "Subprocess with address $addr in RUN $RUNNAME failed."
            failure_flag=1
        fi
    done

    if [ "$failure_flag" -eq 1 ]; then
        return 1
    else
        echo "All subprocesses completed successfully in RUN $RUNNAME."
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

# Main search loop with boolean parameters included
# count=0
for cpu_init in 0 1; do
    for USE_NSYS in 0 1; do
        for tensor_size in $(powers_of_two $WORLD_SIZE); do
            for pipeline_size in $(powers_of_two $WORLD_SIZE); do
                for ((layers_per_virtual_stage = 1; layers_per_virtual_stage < $NUM_LAYERS; layers_per_virtual_stage++)); do
                    for context_size in $(powers_of_two $WORLD_SIZE); do
                        for sequence_parallel in 0 1; do
                            for no_clone_scatter_output_in_embedding in 0 1; do
                                SEARCH_ARGS=""
                                RUNNAME="$(date +%m%d%H)_"
                                if [ $((context_size * pipeline_size * tensor_size)) -le $WORLD_SIZE ]; then
                                    # Check if the combination fits the layers constraint
                                    if [ $layers_per_virtual_stage -gt 1 ]; then
                                        if [ $pipeline_size -gt 2 ]; then
                                            if [ $((layers_per_virtual_stage * pipeline_size)) -le $NUM_LAYERS ]; then
                                                SEARCH_ARGS+="--context-parallel-size $context_size --pipeline-model-parallel-size $pipeline_size --tensor-model-parallel-size $tensor_size --num-layers-per-virtual-pipeline-stage $layers_per_virtual_stage"
                                                RUNNAME+="cp${context_size}-pp${pipeline_size}-tp${tensor_size}-lvpvs${layers_per_virtual_stage}"
                                                if [ $sequence_parallel -eq 1 ]; then
                                                    SEARCH_ARGS+=" --sequence-parallel"
                                                    RUNNAME+="_sp"
                                                fi
                                                if [ $no_clone_scatter_output_in_embedding -eq 1 ]; then
                                                    SEARCH_ARGS+=" --no-clone-scatter-output-in-embedding"
                                                    RUNNAME+="_ncso"
                                                fi
                                                if [ $cpu_init -eq 1 ]; then
                                                    SEARCH_ARGS+=" --use-cpu-initialization"
                                                    RUNNAME+="_cpuinit"
                                                fi
                                                launch
                                                sleep 0.1
                                                # count=$((count + 1))
                                            fi
                                        fi
                                    else #dont set virtual stages argument..
                                        SEARCH_ARGS+="--context-parallel-size $context_size --pipeline-model-parallel-size $pipeline_size --tensor-model-parallel-size $tensor_size"
                                        RUNNAME+="cp${context_size}-pp${pipeline_size}-tp${tensor_size}"
                                        if [ $sequence_parallel -eq 1 ]; then
                                            SEARCH_ARGS+=" --sequence-parallel"
                                            RUNNAME+="_sp"
                                        fi
                                        if [ $no_clone_scatter_output_in_embedding -eq 1 ]; then
                                            SEARCH_ARGS+=" --no-clone-scatter-output-in-embedding"
                                            RUNNAME+="_ncso"
                                        fi
                                        if [ $cpu_init -eq 1 ]; then
                                            SEARCH_ARGS+=" --use-cpu-initialization"
                                            RUNNAME+="_cpuinit"
                                        fi
                                        # set_configs
                                        # sleep 1
                                        # count=$((count + 1))
                                        launch
                                        sleep 0.1
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

echo $count

# eval $(ssh-agent -k)

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
