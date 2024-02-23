#!bin/bash
set -ex

if [ $# -lt 2 ]; then
    echo "Usage: $0 <number of nodes> <setup job>"
    exit 1
fi

nnodes=$1
job=$2

master_hostname=$(hostname)

master_name="${master_hostname%%.*}"
echo "master name: $master_name"

addr_suffix="${master_hostname#*.}"
echo "addr suffix: $addr_suffix"

master_rank="${master_name#node}"
echo "master rank: $master_rank"

# Maximum number of attempts
MAX_ATTEMPTS=5

# Delay between attempts in seconds
RETRY_DELAY=60

for ((n = 0; n <= $nnodes; n++)); do
    if [ $n -ne $master_rank ]; then
        while [ $attempt -le $MAX_ATTEMPTS ]; do
            echo "Attempt $attempt of $MAX_ATTEMPTS: Trying to SSH into node $n for job $job..."
            ssh node$n.$addr_suffix \
                "
                if [[ ! -d \"Megatron-LM\" ]]; then
                sudo apt-get update && sudo apt-get install -y git && \
                git clone https://github.com/mfdj2002/Megatron-LM.git
                fi
                bash run-job.sh '$job'
                "

            status=$?

            if [ $status -eq 0 ]; then
                echo "SSH connection to node $n successful."
                break
            else
                echo "SSH connection failed. Status: $status. Retrying in $RETRY_DELAY seconds..."
                sleep $RETRY_DELAY
            fi

            attempt=$((attempt + 1))
        done

        if [ $attempt -gt $MAX_ATTEMPTS ]; then
            echo "Reached maximum number of retries. Giving up on node $n."
        fi
    fi
done
