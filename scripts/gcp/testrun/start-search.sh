#!/bin/bash

MODEL_SIZE=0.76
export NNODES=1
export GPUS_PER_NODE=8
#export USE_NSYS=0
export JOB_NAME="gpt2-${MODEL_SIZE}B-$(date +%y%m%d%H%M%S)"
export LOGDIR="/logs/${JOB_NAME}"

sudo mkdir -p $LOGDIR
sudo chmod 777 $LOGDIR

echo "Starting grid search for ${MODEL_SIZE}B model..."
echo "Logging to $LOGDIR/grid-search.log"

nohup bash grid-search.sh >"$LOGDIR/grid-search.log" 2>&1 &
echo $! >"$LOGDIR/grid-search.pid"
