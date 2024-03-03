#!/bin/bash

MODEL_SIZE=1.3
export NNODES=4
export GPUS_PER_NODE=2

export JOB_NAME="gpt3-${MODEL_SIZE}B-$(date +%y%m%d%H%M%S)"
export LOGDIR="/mnt/logs/${JOB_NAME}"

sudo mkdir -p $LOGDIR

nohup bash grid-search.sh >"$LOGDIR/grid-search.log" 2>&1 &
echo $! >"$LOGDIR/grid-search.pid"
