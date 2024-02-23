#!bin/bash
set -ex

if [ $# -lt 2 ]; then
	echo "Usage: $0 <master node address> <number of nodes>"
	exit 1
fi

eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa

master_external_addr=$1
nnodes='$2'

# ssh $master_external_addr \
# 	"
# 	bash master/gen-ssh-key.sh
# 	bash start-remote-job.sh '$nnodes' init-setup
# 	if [[ ! -d \"Megatron-LM\" ]]; then
# 		sudo apt-get update && sudo apt-get install -y git && \
# 		git clone https://github.com/mfdj2002/Megatron-LM.git
# 	fi
# 	bash run-job.sh init-setup
# 	"

# Maximum number of attempts
MAX_ATTEMPTS=5

# Delay between attempts in seconds
RETRY_DELAY=60

# Current attempt counter
for job in toy; do
	attempt=1
	while [ $attempt -le $MAX_ATTEMPTS ]; do
		echo "Attempt $attempt of $MAX_ATTEMPTS: Trying to SSH into master node at $master_external_addr for job $job..."
		ssh $master_external_addr \
			"
			if [[ ! -d \"Megatron-LM\" ]]; then
				sudo apt-get update && sudo apt-get install -y git && \
				git clone https://github.com/mfdj2002/Megatron-LM.git
			fi
			cd Megatron-LM/r7525 && \
			bash gen-ssh-keys.sh && \
			bash start-remote-job.sh '$job' && \
			bash run-job.sh '$job'
			"
		status=$?

		if [ $status -eq 0 ]; then
			echo "SSH connection to master node successful."
			break
		else
			echo "SSH connection failed. Status: $status. Retrying in $RETRY_DELAY seconds..."
			sleep $RETRY_DELAY
		fi

		attempt=$((attempt + 1))
	done

	if [ $attempt -gt $MAX_ATTEMPTS ]; then
		echo "Reached maximum number of retries. Giving up."
	else
		echo "Job $job completed successfully."
		echo "sleeping 300 seconds before next job..."
		sleep 300
	fi
done

##############################################################################
# add nodes to known hosts and begin setup
##############################################################################

# for ((n=0; n<=$nnodes; n++)); do
#   if [ "$n" -ne "$master_rank" ]; then
#     # Replace the echo command with whatever action you want to perform
#     echo "begin ssh into node$n"
#     ssh node$n.$addr_suffix \
#         "sudo apt-get update && sudo apt-get install -y git && \
#         git clone https://github.com/mfdj2002/Megatron-LM.git && \
#         cd Megatron-LM/r7525 && \
#         nohup bash init-setup.sh > init-setup.log 2>&1 &"
#   fi
# done

# echo "begin setup on master node"
# sudo apt-get update && sudo apt-get install -y git && \
#     git clone https://github.com/mfdj2002/Megatron-LM.git && \
#     cd Megatron-LM/r7525 && \
#     bash init-setup.sh > init-setup.log 2>&1
eval $(ssh-agent -k)
