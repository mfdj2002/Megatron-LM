#!/bin/bash

# image_name=$1 #mfdj2002/mds:latest
sudo systemctl stop docker
sudo mount /dev/sda4 /mnt #for r7525
sudo systemctl start docker
sudo modprobe nvidia-peermem

sudo mkdir -p /mnt/logs

# Check if the Docker image exists
image_exists=$(docker images -q $IMAGE_NAME)

# If the image does not exist, pull it
if [[ -z "$image_exists" ]]; then
  echo "Image $IMAGE_NAME does not exist. Pulling..."
  docker pull $IMAGE_NAME
fi

# Run the Docker image
echo "Running $IMAGE_NAME..."
docker run --privileged --gpus all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --env-file env.list \
  -v ~/Megatron-LM:/workspace/Megatron-LM -v /mnt:/mnt $IMAGE_NAME
