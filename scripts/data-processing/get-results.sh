#!/bin/bash
# set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <runname> <download_dir> (<hostfile>, default=./hostfile.txt) (<master_node_idx>, default=0)"
    exit 1
fi

RUNNAME=$1
DATA_DIR=${2:-"."}
MASTER_NODE_IDX=${3:-"0"}
HOSTFILE=${4:-"../hostfile.txt"}

LOGDIR=/mnt/logs
DOWNLOAD_DIR=${DATA_DIR}/runs/${RUNNAME}

eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa

mkdir -p "${DOWNLOAD_DIR}/runs"
master_addr=$(head -n $((MASTER_NODE_IDX + 1)) "$HOSTFILE")
echo "Processing master node on $master_addr "
scp -r "${master_addr}:${LOGDIR}/${RUNNAME}" "${DATA_DIR}/runs" >/dev/null 2>&1

subdirs=($(ls -d ${DOWNLOAD_DIR}/*/))
counter=0
while IFS= read -r addr; do
    if [ -n "$addr" ]; then
        if [ $counter -ne $MASTER_NODE_IDX ]; then
            echo "Processing node on $addr"
            for subdir_path in "${subdirs[@]}"; do
                # Extract the name of the subdirectory
                subdir_name=$(basename "$subdir_path")
                # Copy the contents of the subdirectory from the node to the local machine
                scp -r "${addr}:${LOGDIR}/${RUNNAME}/${subdir_name}/*" "$DOWNLOAD_DIR/${subdir_name}/" >/dev/null 2>&1
            done
        fi

        counter=$((counter + 1))
    fi
done <"$HOSTFILE"

eval $(ssh-agent -k)
