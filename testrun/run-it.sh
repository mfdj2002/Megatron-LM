#!/bin/bash

sudo modprobe nvidia-peermem

# Check if the Docker image exists
image_exists=$(docker images -q mfdj2002/megatron:latest)

# If the image does not exist, pull it
if [[ -z "$image_exists" ]]; then
  echo "Image mfdj2002/megatron:latest does not exist. Pulling..."
  docker pull mfdj2002/megatron:latest
fi

# Run the Docker image
echo "Running mfdj2002/megatron:latest..."
docker run --gpus all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v ~/Megatron-LM:/workspace/Megatron-LM -v /mnt:/mnt -v /mnt/logs:/logs \
        -it mfdj2002/megatron:latest
