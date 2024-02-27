#!/bin/bash

# grid search over common hyperparameters/parallelization strategies

#first create a new directory for each run
#!/bin/bash
set -x

eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa

export WORKDIR="/workspace/Megatron-LM"
export HOSTFILE="../hostfile.txt"
LOGDIR="/logs"
IMAGE_NAME="mfdj2002/mds:latest"

MAX_RUNTIME_PER_EXPERIMENT=10 #minutes

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

server_counter=0

# Function to generate powers of two up to a maximum value
powers_of_two() {
    local max_value=$1
    local n=1
    while [ $n -le $max_value ]; do
        echo $n
        n=$((n * 2))
    done
}

FIXED_ARGS="
--use-cpu-initialization \
--recompute-activations \
--distribute-saved-activations \
--recompute-method=uniform \
--use-flash-attn \
--use-mcore-models \

--use-distributed-optimizer \
--overlap-grad-reduce \
--overlap-param-gather \

--no-delay-grad-reduce \
--empty-unused-memory-level=1
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

PROFILING_ARGS="
    --exit-duration-in-mins $MAX_RUNTIME_PER_EXPERIMENT
"

set_config() {
    # Ensure the directory exists
    mkdir -p "$LOGDIR/$RUN_UID"

    # Append each environment variable to the file
    for arg in WORKDIR LOGDIR RUN_UID FIXED_ARGS SEARCH_ARGS TORCHRUN_ARGS GPT_ARGS DATA_ARGS OUTPUT_ARGS PROFILING_ARGS; do
        if [ -n "${!arg}" ]; then # Check if the variable is set and not empty
            echo "export $arg='${!arg}' && \\" >>"$LOGDIR/$RUN_UID/configs.env"
        fi
    done
}

SEARCH_ARGS=""

# Main search loop with boolean parameters included
for context_size in $(powers_of_two $WORLD_SIZE); do
    for pipeline_size in $(powers_of_two $WORLD_SIZE); do
        for tensor_size in $(powers_of_two $WORLD_SIZE); do
            for layers_per_virtual_stage in $(powers_of_two $NUM_LAYERS); do
                for sequence_parallel in 0 1; do
                    for no_clone_scatter_output_in_embedding in 0 1; do
                        if [ $((context_size * pipeline_size * tensor_size)) -le $WORLD_SIZE ]; then
                            # Check if the combination fits the layers constraint
                            if [ $virtual_stages -gt 1]; then
                                if [ $pipeline_size -gt 2]; then
                                    if [ $((virtual_stages * pipeline_size)) -le $NUM_LAYERS ]; then
                                        RUN_UID=$(date +%Y%m%d%H%M%S)
                                        SEARCH_ARGS+="
                                            --context-parallel-size $context_size \
                                            --pipeline-model-parallel-size $pipeline_size \
                                            --tensor-model-parallel-size $tensor_size \
                                            --num-layers-per-virtual-pipeline-stage $layers_per_virtual_stage
                                            "
                                        if [ $sequence_parallel -eq 1 ]; then
                                            SEARCH_ARGS+="--sequence-parallel"
                                        fi
                                        if [ $no_clone_scatter_output_in_embedding -eq 1]; then
                                            SEARCH_ARGS+="--no-clone-scatter-output-in-embedding"
                                        fi
                                        set_config
                                    fi
                                fi
                            else #dont set virtual stages argument..
                                RUN_UID=$(date +%Y%m%d%H%M%S)
                                SEARCH_ARGS+="
                                            --context-parallel-size $context_size \
                                            --pipeline-model-parallel-size $pipeline_size \
                                            --tensor-model-parallel-size $tensor_size
                                            "
                                if [ $sequence_parallel -eq 1 ]; then
                                    SEARCH_ARGS+="--sequence-parallel"
                                fi
                                if [ $no_clone_scatter_output_in_embedding -eq 1]; then
                                    SEARCH_ARGS+="--no-clone-scatter-output-in-embedding"
                                fi
                                set_config
                            fi
                        fi

                    done
                done
            done
        done
    done
done

# # Check if the combination is within the device limit

# while IFS= read -r addr; do
#     ssh -n "$addr" \
#         "
#         export NNODES='$NNODES' && \
#         export WORKDIR='$WORKDIR' && \
#         export GPUS_PER_NODE='$GPUS_PER_NODE' && \
#         export WORLD_SIZE='$WORLD_SIZE' && \
#         export MASTER_ADDR='$MASTER_ADDR' && \
#         export MASTER_PORT='$MASTER_PORT' && \
#         export NODE_RANK='$server_counter' && \
#         cd $WORKDIR && \
#         nohup bash '$JOB'.sh </dev/null &
#         exit 0
#         "
#     status=$?
#     if [ $status -eq 0 ]; then
#         echo "Starting '$JOB' to node $addr successful."
#     else
#         echo "Starting '$JOB' on node $addr failed. Status: $status."
#     fi
#     server_counter=$((server_counter + 1))
# done <"$HOSTFILE"

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
