#!/bin/sh
set -e

# sudo mkfs.ext4 /dev/sdb #will be quite different if on a different machine
sudo mkfs.ext4 /dev/sda4
# # sudo mount /dev/sdb /mnt
sudo mount /dev/sda4 /mnt #for r7525

# Update and upgrade the system
sudo apt update

sudo apt install docker.io -y


# sudo groupadd docker
# sudo usermod -aG docker $USER
# newgrp docker
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

sudo mkdir /mnt/tmp && sudo chmod 1777 /mnt/tmp
cd /mnt/tmp

mnt=/mnt

group=$(id -gn)

sudo chown -R $USER:$group $mnt

for dir in .vscode-server .debug .cache .local; do
    sudo mkdir -p $mnt/$dir
    sudo chown -R $USER:$group $mnt/$dir
    rm -rf ~/$dir
    ln -s $mnt/$dir ~/$dir
done

wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
# wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --installpath=/mnt/cuda --tmpdir=/mnt/tmp
# sudo sh cuda_11.8.0_520.61.05_linux.run --installpath=/mnt/cuda --tmpdir=/mnt/tmp

rm cuda_12.1.0_530.30.02_linux.run
# rm cuda_11.8.0_520.61.05_linux.run

sudo dkms autoinstall && sudo modprobe nvidia && sudo reboot
