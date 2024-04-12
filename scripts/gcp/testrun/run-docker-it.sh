#!/bin/bash

# sudo modprobe nvidia-peermem
IMAGE_NAME="mfdj2002/mds:gcp"

# Check if the Docker image exists
image_exists=$(docker images -q $IMAGE_NAME)

# If the image does not exist, pull it
if [[ -z "$image_exists" ]]; then
  echo "Image $IMAGE_NAME does not exist. Pulling..."
  docker pull $IMAGE_NAME
fi

LOGDIR="/logs"
mkdir -p $LOGDIR

# Run the Docker image
echo "Running $IMAGE_NAME..."
docker run --gpus all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/Megatron-LM:/workspace/Megatron-LM \
  -v /logs:/logs \
  --entrypoint /bin/bash \
  -it $IMAGE_NAME
