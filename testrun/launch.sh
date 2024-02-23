#!/bin/bash

# image_name=$1 #mfdj2002/mds:r7525

sudo modprobe nvidia-peermem

# Check if the Docker image exists
image_exists=$(docker images -q $IMAGE_NAME)

# If the image does not exist, pull it
if [[ -z "$image_exists" ]]; then
  echo "Image $IMAGE_NAME does not exist. Pulling..."
  docker pull $IMAGE_NAME
fi

# Run the Docker image
echo "Running $IMAGE_NAME..."
docker run --gpus all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e GPUS_PER_NODE=$GPUS_PER_NODE -e WORLD_SIZE=$WORLD_SIZE -e MASTER_ADDR=$MASTER_ADDR -e MASTER_PORT=$MASTER_PORT -e NNODES=$NNODES -e NODE_RANK=$NODE_RANK \
  -v ~/Megatron-LM:/workspace/Megatron-LM -v /mnt:/mnt -v /mnt/logs:/logs $IMAGE_NAME
