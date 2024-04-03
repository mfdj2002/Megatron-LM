#!bin/bash
set -x

eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa

WORKDIR="Megatron-LM/scripts/r7525"
HOSTFILE="./hostfile.txt"

# Maximum number of attempts
MAX_ATTEMPTS=5

# Delay between attempts in seconds
RETRY_DELAY=60

jobs=("init-setup" "install-deps")
total_jobs=${#jobs[@]}

for i in "${!jobs[@]}"; do
	job=${jobs[$i]}
	while IFS= read -r addr; do
		if [ -n "$addr" ]; then
			attempt=1
			while [ $attempt -le $MAX_ATTEMPTS ]; do
				if [ $attempt -gt 1 ]; then
					echo "Last attempt to connect to master failed. Retrying in $RETRY_DELAY seconds..."
					sleep $RETRY_DELAY
				fi
				echo "Attempt $attempt of $MAX_ATTEMPTS: Trying to SSH into node $addr for job $job..."
				ssh -o StrictHostKeyChecking=no -n "$addr" \
					"
                if [[ ! -d \"Megatron-LM\" ]]; then
                sudo apt-get update && sudo apt-get install -y git && \
                git clone https://github.com/mfdj2002/Megatron-LM.git
                fi
                cd $WORKDIR && \
                mkdir -p setup-logs && \
                if [ -f setup-logs/$job.log ]; then
                echo \"job $job already executed on node $addr.\"
                else
                nohup bash $job.sh >setup-logs/$job.log 2>&1 </dev/null &
                exit 0
                fi
                "
				status=$?
				if [ $status -eq 0 ]; then
					echo "SSH connection to node $addr successful."
					break
				else
					echo "SSH connection failed. Status: $status."
					attempt=$((attempt + 1))
				fi
			done
			if [ $attempt -gt $MAX_ATTEMPTS ]; then
				echo "Reached maximum number of retries. Giving up."
			else
				echo "Job $job sent successfully."
			fi
		fi
	done <"$HOSTFILE"
	# Check if not last job then sleep
	if [[ $i -lt $((total_jobs - 1)) ]]; then
		sleep 300
	fi
done

eval $(ssh-agent -k)

# bash gen-ssh-keys.sh && \
