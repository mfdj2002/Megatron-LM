#!/bin/bash
set -x

eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa

IMAGE_NAME="mfdj2002/mds:latest"
HOSTFILE="../hostfile.txt"

WORKDIR="Megatron-LM/testrun"
JOB="launch"

NNODES=$(wc -l <"$HOSTFILE")
GPUS_PER_NODE=2
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
MASTER_ADDR=$(ssh -n $(head -n 1 "$HOSTFILE") "hostname")
MASTER_PORT=6000

counter=0

while IFS= read -r addr; do
    ssh -n "$addr" \
        "
        export NNODES='$NNODES' && \
        export WORKDIR='$WORKDIR' && \
        export GPUS_PER_NODE='$GPUS_PER_NODE' && \
        export WORLD_SIZE='$WORLD_SIZE' && \
        export MASTER_ADDR='$MASTER_ADDR' && \
        export MASTER_PORT='$MASTER_PORT' && \
        export NODE_RANK='$counter' && \
        cd $WORKDIR && \
        nohup bash '$JOB'.sh </dev/null &
        exit 0
        "
    status=$?
    if [ $status -eq 0 ]; then
        echo "Starting '$JOB' to node $addr successful."
    else
        echo "Starting '$JOB' on node $addr failed. Status: $status."
    fi
    counter=$((counter + 1))
done <"$HOSTFILE"

eval $(ssh-agent -k)
