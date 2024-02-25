#!bin/bash
set -x

master_hostname=$(hostname)

master_name="${master_hostname%%.*}"
echo "master name: $master_name"

addr_suffix="${master_hostname#*.}"
echo "addr suffix: $addr_suffix"

master_rank="${master_name#node}"
echo "master rank: $master_rank"

# retry_enabled=${retry_enabled:-0} # Default to 0 (disabled). Set to 1 to enable retries.

for ((n = 0; n <= $NNODES; n++)); do
    if [ $n -ne $master_rank ]; then
        echo "Trying to SSH into node $n for job $JOB..."
        ssh-keyscan -H node$n.$addr_suffix >>~/.ssh/known_hosts
        ssh node$n.$addr_suffix \
            "
            if [ -f '$WORKDIR'/logs/'$JOB'.log ]; then
                echo \"job $JOB already executed on node '$n'.\"
            else
                if [[ ! -d \"Megatron-LM\" ]]; then
                    sudo apt-get update && sudo apt-get install -y git &&
                        git clone https://github.com/mfdj2002/Megatron-LM.git
                fi
                cd '$WORKDIR' && \
                bash gen-ssh-keys.sh && \
                mkdir -p logs && \
                nohup bash '$JOB'.sh >logs/'$JOB'.log 2>&1 &
            fi
            "

        status=$?

        if [ $status -eq 0 ]; then
            echo "SSH connection to master node successful."
        else
            echo "SSH connection failed. Status: $status."
            exit 1
        fi
    fi
done
