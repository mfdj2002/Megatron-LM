#!/bin/bash

MODEL_SIZE=1.3
export NNODES=4
export GPUS_PER_NODE=2
export TORCH_CPP_LOG_LEVEL=INFO
export NCCL_DEBUG=INFO

export JOB_NAME="gpt3-${MODEL_SIZE}B-$(date +%y%m%d%H%M%S)"
export LOGDIR="/mnt/logs/${JOB_NAME}"

sudo mkdir -p $LOGDIR
sudo chmod 777 $LOGDIR

echo "Starting grid search for ${MODEL_SIZE}B model..."
echo "Logging to $LOGDIR/grid-search.log"

nohup bash grid-search.sh >"$LOGDIR/grid-search.log" 2>&1 &
echo $! >"$LOGDIR/grid-search.pid"
