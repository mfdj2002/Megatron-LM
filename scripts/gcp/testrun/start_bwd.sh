MODEL_SIZE=0.125
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTN_HEADS=12
MICRO_BATCH_SIZE=4
LR=6.0e-4
MIN_LR=1.0e-6
SEQ_LEN=1024

CHECKPOINT_PATH=scripts/gcp/testrun/test
VOCAB_FILE=scripts/testrun/gpt2-vocab.json
MERGE_FILE=scripts/testrun/gpt2-merges.txt
DATA_PATH=scripts/testrun/dataset/pile_gpt_train_text_document

GPT_ARGS="
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTN_HEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --lr $LR \
    --train-iters 10 \
    --fp16 \
    --micro-batch-size $MICRO_BATCH_SIZE
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --save $CHECKPOINT_PATH
    --log-interval 5 \
    --save-interval 10 \
    --eval-interval 10 \
    --eval-iters 10
"

WORKDIR="/workspace/Megatron-LM"

cd $WORKDIR
export CUDA_DEVICE_MAX_CONNECTIONS=1

export MASTER_ADDR="megatron-gpt2-0"
export MASTER_PORT=6000

python pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS
    #--master_addr test0 \
   # --master_port 6000

