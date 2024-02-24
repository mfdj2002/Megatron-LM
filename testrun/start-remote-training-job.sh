#!bin/bash
set -x

# retry_enabled=${retry_enabled:-0} # Default to 0 (disabled). Set to 1 to enable retries.

for ((n = 0; n <= $NNODES; n++)); do
    if [ $n -ne $master_rank ]; then
        echo "Trying to SSH into node $n for job $JOB..."
        ssh node$n.$addr_suffix \
            "
            export NNODES='$NNODES' && \
            export JOB='$job' && \
            export WORK_DIR='$WORKDIR' && \
            export GPUS_PER_NODE='$GPUS_PER_NODE' && \
            export WORLD_SIZE='$WORLD_SIZE' && \
            export MASTER_ADDR='$master_hostname' && \
            export MASTER_PORT=6000 && \
            export NODE_RANK=$n && \
            export IMAGE_NAME='$IMAGE_NAME' && \
            cd $WORKDIR && \
            mkdir -p logs && \
            nohup bash '$JOB'.sh >logs/'$JOB'.log 2>&1 &
            "

        status=$?

        if [ $status -eq 0 ]; then
            echo "SSH connection to node$n successful."
            break
        else
            echo "SSH connection failed. Status: $status."
        fi
    fi
done
