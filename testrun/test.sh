#!/bin/bash
set -x

IMAGE_NAME=mfdj2002/mds:latest
MAX_RUNTIME_PER_EXPERIMENT=5
eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa

ssh -n node2.mds1.hetkv-pg0.clemson.cloudlab.us \
    "
    mkdir -p /users/jf3516/orchestrator_logs/030103_cp1-pp2-tp2_sp_ncso_cpuinit
    sudo systemctl stop docker
    sudo mount /dev/sda4 /mnt
    sudo systemctl start docker
    sudo modprobe nvidia-peermem

    docker pull $IMAGE_NAME

    docker_cmd=\$(head -n 1 /users/jf3516/orchestrator_logs/030103_cp1-pp2-tp2_sp_ncso_cpuinit/master_docker_command.txt)

    \$docker_cmd
    "

#     " >$ORCHESTRATOR_LOGDIR/$RUN_UID/ssh_node$NODE_RANK.log 2>&1) &

eval $(ssh-agent -k)

#

# sudo mkdir -p $test_configs/
# cat >>$test_configs/$RUN_UID/configs.env
# /users/jf3516/Megatron-LM/testrun/test_configs/20240301012510
# image_exists=$(docker images -q $IMAGE_NAME)
