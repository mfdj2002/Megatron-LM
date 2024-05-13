#!/bin/bash

# grid search over common hyperparameters/parallelization strategies

# set -x

# eval $(ssh-agent -s)
# ssh-add ~/.ssh/id_rsa

## GPT-3 13B
# MODEL_SIZE=13
# NUM_LAYERS=40
# HIDDEN_SIZE=5120
# MICRO_BATCH_SIZE=32
# NUM_ATTN_HEADS=40
# LR=1.0e-4
# MIN_LR=1.0e-6
# INIT_STD=0.008
# SEQ_LEN=1024

# # GPT-3 1.3B
# NUM_LAYERS=24
# HIDDEN_SIZE=2048
# NUM_ATTN_HEADS=32
# # MICRO_BATCH_SIZE=1
# LR=2.0e-4
# MIN_LR=1.0e-6
# SEQ_LEN=1024
# # init_std=0.013

# GPT-3 Large 760M
NUM_LAYERS=24
HIDDEN_SIZE=1536
NUM_ATTN_HEADS=16
LR=2.5e-4
MIN_LR=1.0e-6
SEQ_LEN=1024
# init_std=0.015

# GPT-3 Small 125M
# MODEL_SIZE=0.125
# NUM_LAYERS=12
# HIDDEN_SIZE=768
# NUM_ATTN_HEADS=12
# MICRO_BATCH_SIZE=1
# LR=6.0e-4
# MIN_LR=1.0e-6
# SEQ_LEN=1024
# init_std=0.02

WORKDIR="/workspace/Megatron-LM"
HOSTFILE="../hostfile.txt"

IMAGE_NAME="mfdj2002/mds:gcp"

MAX_RUNTIME_PER_EXPERIMENT=5 #minutes

if [ -z "$NNODES" ] || [ -z "$GPUS_PER_NODE" ]; then
    echo "Error: NNODES and GPUS_PER_NODE are required."
    exit -1
fi
#assumes master is also orchestrator

# # NNODES=$(wc -l <"$HOSTFILE")
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
# MASTER_ADDR=$(ssh -n $(head -n 1 "$HOSTFILE") "hostname")
MASTER_ADDR=$(hostname)
MASTER_PORT=6000

# ADDR_SUFFIX="${MASTER_ADDR#*.}"

# CHECKPOINT_PATH=${HOME}/checkpoints/${JOB_NAME}
VOCAB_FILE=scripts/testrun/gpt2-vocab.json
MERGE_FILE=scripts/testrun/gpt2-merges.txt
DATA_PATH=scripts/testrun/dataset/pile_gpt_train_text_document

# Function to generate powers of two up to a maximum value
powers_of_two() {
    local max_value=$1
    local n=1
    while [ $n -le $max_value ]; do #assuming using single strategy is suboptimal -le or -lt
        echo $n
        n=$((n * 2))
    done
}

#--use-flash-attn \
#cannot use flash attention because only supported in A100 GPUs

# --recompute-activations \
# --distribute-saved-activations \
# --recompute-method=uniform \
# --use-distributed-optimizer \
# --overlap-grad-reduce \
# --overlap-param-gather \

#--use-mcore-models --no-delay-grad-reduce --empty-unused-memory-level=1

FIXED_ARGS="

--exit-duration-in-mins $MAX_RUNTIME_PER_EXPERIMENT
"

# --lr-decay-iters 320000 \
# --lr-decay-style cosine \
# --weight-decay 0.1 \
# --clip-grad 1.0 \
# --hysteresis 2 \
# --min-lr $MIN_LR \

GPT_ARGS="
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTN_HEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --lr $LR \
    --train-iters 100
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 5 \
    --save-interval 100 \
    --eval-interval 10 \
    --eval-iters 10
"

OMP_NUM_THREADS=$(($(nproc --all) / $GPUS_PER_NODE))

############# only for basic profiling runs:
# USE_NSYS=0
# use for nvprof runs:
# USE_NSYS=1

env_vars=("WORKDIR" "LOGDIR" "RUNNAME" "OMP_NUM_THREADS" "USE_NSYS" "NSYS_CMD" "NODE_RANK" "MAX_RUNTIME_PER_EXPERIMENT" "FIXED_ARGS" "SEARCH_ARGS" "TORCHRUN_ARGS" "GPT_ARGS" "DATA_ARGS" "OUTPUT_ARGS")

launch() {
    echo "$(date +%y-%m-%d,%H:%M:%S) Launching RUN $RUNNAME..."
    mkdir -p $LOGDIR/$RUNNAME/orchestrator-log
    local pids=()

    # for ((NODE_RANK = 0; NODE_RANK < $NNODES; NODE_RANK++)); do
    NODE_RANK=0
    while IFS= read -r addr; do
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
        docker_cmd="docker run"
        for var in "${env_vars[@]}"; do
            docker_cmd+=" -e $var=\"${!var}\""
        done
        docker_cmd+=" --privileged --gpus all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ~/Megatron-LM:/workspace/Megatron-LM -v /$LOGDIR:/$LOGDIR --rm $IMAGE_NAME"

        if [ $NODE_RANK -eq 0 ]; then
            echo $docker_cmd >$LOGDIR/$RUNNAME/orchestrator-log/master_docker_command.txt
            echo "docker command to run: $docker_cmd"
        fi

        # addr="node$NODE_RANK.$ADDR_SUFFIX"

        # Execute the SSH command in the background
        # ssh-keygen -F $addr >/dev/null

        # $? is the exit code of the last command (ssh-keygen -F)
        # If the exit code is 0, the host already exists in known_hosts
        # if [ $? -ne 0 ]; then
        # If the host doesn't exist, add it
        # touch ~/.ssh/known_hosts
        # chmod 644 ~/.ssh/known_hosts
        # ssh-keyscan -H $addr >>~/.ssh/known_hosts
        # echo "Host $addr added to known_hosts."
        # fi
        max_time="8m"
        if [ $counter -eq 0 ]; then
            max_time="15m"
        fi
        #8m for basic profiling runs, 10m for nvprof?
        #timeout "8m" ssh -n "$addr" \
        #assuming all the instances are on the same zone...
        timeout $max_time gcloud compute ssh $addr \
            --command="
            sudo mkdir -p $LOGDIR/$RUNNAME
            sudo chmod \$USER:\$USER $LOGDIR/$RUNNAME
            echo \"\$(date +%y-%m-%d,%H:%M:%S) Docker running on node $addr...\"
            docker pull $IMAGE_NAME
            $docker_cmd || exit 1
            echo \"\$(date +%y-%m-%d,%H:%M:%S) Docker finished on node $addr...\"
            " --zone $(gcloud compute instances list $(hostname) --format 'csv[no-heading](zone)') --quiet >$LOGDIR/$RUNNAME/orchestrator-log/ssh_node${NODE_RANK}.log 2>&1 &
        pids+=("$!")
        NODE_RANK=$((NODE_RANK + 1))
    done <"$HOSTFILE"
    echo "$(date +%y-%m-%d,%H:%M:%S) Waiting for subprocesses to finish in RUN $RUNNAME..."

    # nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw,pstate,pcie.link.gen.max,pcie.link.gen.current --format=csv -l 5 | sudo tee "\""$LOGDIR/$RUNNAME/nvidia-smi-rank${NODE_RANK}.csv"\"" > /dev/null & PGID=\$!
    # dool --more --output "\""$LOGDIR/$RUNNAME/dool-rank${NODE_RANK}.csv"\"" 5 --pgid=\$PGID &

    local failure_flag=0
    for rank in "${!pids[@]}"; do
        pid=${pids[$rank]}
        wait "$pid"
        exit_status=$?

        # if [ $exit_status -eq 124 ]; then
        #     echo "$(date +%y-%m-%d,%H:%M:%S) Node $rank in RUN $RUNNAME timed out. Sending sudo killall"
        #     gcloud compute ssh $addr --command="sudo killall -9 nvidia-smi; pkill -f dool; docker stop \$(docker ps -q)" --zone $(gcloud compute instances list $(hostname) --format 'csv[no-heading](zone)') --quiet
        #     failure_flag=1
        # fi
        if [ $exit_status -ne 0 ]; then
            echo "$(date +%y-%m-%d,%H:%M:%S) Node $rank in RUN $RUNNAME failed with exit status $exit_status."
            failure_flag=1
        fi
    done

    if [ "$failure_flag" -eq 1 ]; then
        return 1
    else
        echo "$(date +%y-%m-%d,%H:%M:%S) All subprocesses completed successfully in RUN $RUNNAME."
        return 0
    fi
}

USE_NSYS=0

counter=0
# for distribute_saved_activations in 0 1; do
#for USE_NSYS in 0 1; do
for global_batch_size in 32 64 128 256 512 1024; do #8 16
    for num_microbatches in 1 2 4 8 16 32; do
        # for recompute_activation in 0 1; do
        # for standalone_embedding in 0 1; do
        #     for no_clone_scatter_output_in_embedding in 0 1; do
        # for sequence_parallel in 0 1; do
        # for context_size in $(powers_of_two $WORLD_SIZE); do
        # for ((layers_per_virtual_stage = 1; layers_per_virtual_stage <= $NUM_LAYERS / 2; layers_per_virtual_stage++)); do
        for pipeline_size in $(powers_of_two $WORLD_SIZE); do
            for tensor_size in $(powers_of_two $WORLD_SIZE); do
                # for USE_NSYS in 0 1; do
                # for cpu_init in 0 1; do
                if [ $((pipeline_size * tensor_size)) -gt $WORLD_SIZE ]; then
                    continue
                fi
                micro_batch_size=$((global_batch_size / num_microbatches / pipeline_size / tensor_size))
                if [ $micro_batch_size -lt 1 ]; then
                    continue
                fi
                SEARCH_ARGS="--global-batch-size $global_batch_size --micro-batch-size $micro_batch_size"
                RUNNAME="$(date +%y%m%d%H%M%S)-gb${global_batch_size}-mb${micro_batch_size}"
                if [ $USE_NSYS -eq 1 ]; then
                    RUNNAME+="-nv_prof"
                else
                    RUNNAME+="-basic_prof"
                fi
                # if [ $((context_size * pipeline_size * tensor_size)) -gt $WORLD_SIZE ]; then
                #     continue
                # fi
                # if [ $recompute_activation -eq 1 ]; then
                SEARCH_ARGS+=" --recompute-granularity=full --recompute-method=uniform --recompute-num-layers=$(($NUM_LAYERS / $pipeline_size))" #" --recompute-activations" #--recompute-granularity=selective"
                RUNNAME+="-rm"
                # fi
                # if [ $distribute_saved_activations -eq 1 ]; then
                #     if [ $tensor_size -eq 1 ]; then
                #         continue
                #     fi
                #     if [ $recompute_activation -eq 1 ]; then
                #         continue
                #     fi
                #     SEARCH_ARGS+=" --distribute-saved-activations --recompute-granularity=full --recompute-method=uniform" #defaults to uniform, saves block for future exploration
                #     RUNNAME+="-dsa"
                # fi
                # if [ $tensor_size -eq 1 ]; then # Added missing space before 'then'
                # if [ $sequence_parallel -eq 1 ]; then
                #     continue
                # fi
                # fi
                # if [ $sequence_parallel -eq 1 ]; then
                #     SEARCH_ARGS+=" --sequence-parallel --no-async-tensor-model-parallel-allreduce"
                #     RUNNAME+="-sp"
                # fi
                # if [ $no_clone_scatter_output_in_embedding -eq 1 ]; then
                #     SEARCH_ARGS+=" --no-clone-scatter-output-in-embedding"
                #     RUNNAME+="-ncso"
                # fi
                # if [ $cpu_init -eq 1 ]; then
                #     SEARCH_ARGS+=" --use-cpu-initialization"
                #     RUNNAME+="-cpuinit"
                # fi

                SEARCH_ARGS+=" --pipeline-model-parallel-size $pipeline_size --tensor-model-parallel-size $tensor_size"
                RUNNAME+="-pp${pipeline_size}-tp${tensor_size}"
                launch
                sleep 10
                counter=$((counter + 1))

                # if [ $layers_per_virtual_stage -gt 1 ]; then
                #     if [ $pipeline_size -lt 2 ]; then
                #         continue
                #     fi
                #     # if [ $standalone_embedding -eq 1 ]; then
                #     #     if [ $(($NUM_LAYERS % (layers_per_virtual_stage - 1))) -ne 0 ]; then
                #     #         continue
                #     #     fi
                #     #     SEARCH_ARGS+=" --standalone-embedding-stage"
                #     #     RUNNAME+="-se"
                #     # else
                #     if [ $(($NUM_LAYERS % layers_per_virtual_stage)) -ne 0 ]; then
                #         continue
                #     fi
                #     # fi
                #     SEARCH_ARGS+=" --context-parallel-size $context_size --pipeline-model-parallel-size $pipeline_size --tensor-model-parallel-size $tensor_size --num-layers-per-virtual-pipeline-stage $layers_per_virtual_stage"
                #     # SEARCH_ARGS+=" --pipeline-model-parallel-size $pipeline_size --tensor-model-parallel-size $tensor_size --num-layers-per-virtual-pipeline-stage $layers_per_virtual_stage"

                #     RUNNAME+="-cp${context_size}-pp${pipeline_size}-tp${tensor_size}-lvpvs${layers_per_virtual_stage}"
                #     # RUNNAME+="-pp${pipeline_size}-tp${tensor_size}-lvpvs${layers_per_virtual_stage}"

                #     # launch
                #     # sleep 30
                #     counter=$((counter + 1))

                # else #dont set virtual stages argument..
                #     SEARCH_ARGS+=" --context-parallel-size $context_size --pipeline-model-parallel-size $pipeline_size --tensor-model-parallel-size $tensor_size"
                #     RUNNAME+="-cp${context_size}-pp${pipeline_size}-tp${tensor_size}"
                #     # if [ $standalone_embedding -eq 1 ]; then
                #     #     SEARCH_ARGS+=" --standalone-embedding-stage"
                #     #     RUNNAME+="-se"
                #     # fi
                #     # launch
                #     # sleep 30
                #     counter=$((counter + 1))
                # fi
                # done
                # done
            done
            #done
            # done
            # done
            # done
            # done
            # done
        done
    done
done
#done
# done
counter=$((counter + 1))
echo "Total number of runs: $counter"

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
