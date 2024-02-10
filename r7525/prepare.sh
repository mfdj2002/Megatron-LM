#!/bin/sh
set -e

MNT_DIR=/mnt

# sudo mkfs.ext4 /dev/sdb #will be quite different if on a different machine
sudo mkfs.ext4 /dev/sda4
# # sudo mount /dev/sdb /mnt
sudo mount /dev/sda4 $MNT_DIR #for r7525

# Update and upgrade the system
sudo apt update

sudo apt install docker.io -y

sudo groupadd docker || true
sudo usermod -aG docker $USER || true



distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Install Python development packages and pip
# sudo apt install python3-dev python3-pip -y

STAGE_DIR=$MNT_DIR/tmp

sudo mkdir -p $STAGE_DIR && sudo chmod 1777 $STAGE_DIR
cd $STAGE_DIR


group=$(id -gn)

sudo chown -R $USER:$group $MNT_DIR

for dir in .vscode-server .debug .cache .local; do
    sudo mkdir -p $MNT_DIR/$dir
    sudo chown -R $USER:$group $MNT_DIR/$dir
    rm -rf ~/$dir
    ln -s $MNT_DIR/$dir ~/$dir
done

# ##############################################################################
# MLNX_OFED
##############################################################################

# ENV MLNX_OFED_VERSION=4.9-7.1.0.0
MLNX_OFED_VERSION=5.4-3.7.5.0
# RUN apt-get update && \
sudo apt-get install -y --no-install-recommends libcap2

cd ${STAGE_DIR} && \
wget -q -O - http://www.mellanox.com/downloads/ofed/MLNX_OFED-${MLNX_OFED_VERSION}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu20.04-x86_64.tgz | tar xzf - && \
cd MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu20.04-x86_64 && \
./mlnxofedinstall --force --user-space-only --add-kernel-support --without-fw-update --all -q && \
cd ${STAGE_DIR} && \
rm -rf ${STAGE_DIR}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu20.04-x86_64*

##############################################################################
# CUDA driver and toolkit
##############################################################################

cd $STAGE_DIR && \
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run && \
sudo sh cuda_12.1.0_530.30.02_linux.run --installpath=/mnt/cuda --tmpdir=/mnt/tmp && \
rm cuda_12.1.0_530.30.02_linux.run && \
sudo dkms autoinstall && sudo modprobe nvidia && \
sudo modprobe nvidia-peermem && sudo reboot
