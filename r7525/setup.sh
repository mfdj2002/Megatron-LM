#!bin/bash
set -ex

if [ $# -lt 2 ]; then
	echo "Usage: $0 <master node address> <number of nodes>"
	exit 1
fi

eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa

master_external_addr=$1
NNODES=$2
WORKDIR="~/Megatron-LM/r7525"

# Maximum number of attempts
MAX_ATTEMPTS=5

# Delay between attempts in seconds
RETRY_DELAY=60

# Current attempt counter

for job in toy; do
	attempt=1
	while [ $attempt -le $MAX_ATTEMPTS ]; do
		if [ $attempt -gt 1 ]; then
			echo "Last attempt to connect to master failed. Retrying in $RETRY_DELAY seconds..."
			sleep $RETRY_DELAY
		fi
		echo "Attempt $attempt of $MAX_ATTEMPTS: Trying to SSH into master node at $master_external_addr for job $job..."
		ssh $master_external_addr \
			"
			export NNODES='$NNODES' && \
			export JOB='$job' && \
			export WORK_DIR='$WORKDIR' && \
			if [[ ! -d \"Megatron-LM\" ]]; then
				sudo apt-get update && sudo apt-get install -y git && \
				git clone https://github.com/mfdj2002/Megatron-LM.git
			fi
			cd '$WORKDIR' && \
			bash gen-ssh-keys.sh && \
			mkdir -p logs && \
			bash start-remote-job.sh && \
			if [ -f '$WORKDIR'/logs/'$JOB'.log ]; then
				echo \"job $JOB already executed on node '$n'.\"
			else
				nohup bash '$job'.sh >logs/'$job'.log 2>&1 &
			fi
			"
		status=$?

		if [ $status -eq 0 ]; then
			echo "SSH connection to master node successful."
			break
		else
			echo "SSH connection failed. Status: $status."
			attempt=$((attempt + 1))
		fi
	done

	if [ $attempt -gt $MAX_ATTEMPTS ]; then
		echo "Reached maximum number of retries. Giving up."
	else
		echo "Job $job completed successfully."
	fi
	echo "Sleeping 300 seconds before next job..."
	sleep 300
done

eval $(ssh-agent -k)
