#!bin/bash
set -x

if [ $# -lt 1 ]; then
    echo "Usage: $0 <setup job>"
    exit 1
fi

job=$1

#TODO: instead of always cd to Megatron-LM/r7525, check whether we are already at the correct directory
cd Megatron-LM/r7525 &&
    mkdir -p logs &&
    nohup bash $job.sh >logs/$job.log 2>&1 &
