#!/bin/bash
set -x

if [ $# -lt 2 ]; then
    echo "Usage: $0 <master node address> <number of nodes>"
    exit 1
fi

eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa

master_external_addr=$1
NNODES=$2
WORKDIR="~/Megatron-LM/testrun"
job="launch"
GPUS_PER_NODE=2
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
IMAGE_NAME="mfdj2002/mds:r7525"

ssh $master_external_addr \
    "
    export NNODES='$NNODES' && \
    export JOB='$job' && \
    export WORK_DIR='$WORKDIR' && \
    export GPUS_PER_NODE='$GPUS_PER_NODE' && \
    export WORLD_SIZE='$WORLD_SIZE' && \
    export MASTER_ADDR=$(hostname) && \
    export MASTER_PORT=6000 && \
    export NODE_RANK=0 && \
    master_hostname=$(hostname)
    master_name=\"${master_hostname%%.*}\"
    echo \"master name: $master_name\"

    addr_suffix=\"${master_hostname#*.}\"
    echo \"addr suffix: $addr_suffix\"

    master_rank=\"${master_name#node}\"
    echo \"master rank: $master_rank\"
    bash $WORKDIR/start-remote-training-job.sh
    "

# master_name="${master_hostname%%.*}"
# echo "master name: $master_name"

# addr_suffix="${master_hostname#*.}"
# echo "addr suffix: $addr_suffix"

# master_rank="${master_name#node}"
# echo "master rank: $master_rank"

# for ((n = 0; n <= $nnodes; n++)); do
#     if [ $n -ne $master_rank ]; then
#         echo "Trying to SSH into node $n for job $job..."
#         ssh node$n.$addr_suffix \
#             "
#                 if [[ ! -d \"Megatron-LM\" ]]; then
#                 sudo apt-get update && sudo apt-get install -y git && \
#                 git clone https://github.com/mfdj2002/Megatron-LM.git
#                 fi
#                 bash run-job.sh '$job'
#                 "

#         status=$?

#         if [ $status -eq 0 ]; then
#             echo "SSH connection to node $n successful."
#             break
#         else
#             echo "SSH connection failed. Status: $status."

#         fi

#     fi
# done

eval $(ssh-agent -k)
