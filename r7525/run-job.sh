#!bin/bash
set -ex

if [ $# -lt 1 ]; then
    echo "Usage: $0 <setup job>"
    exit 1
fi

job=$1

cd Megatron-LM/r7525 &&
    mkdir -p logs &&
    nohup bash $job.sh >logs/$job.log 2>&1 &
