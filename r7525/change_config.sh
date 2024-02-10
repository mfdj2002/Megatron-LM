#!/bin/bash
set -e
set -x

# Make sure you are mounted
sudo mount /dev/sda4 /mnt #for r7525

# Move Docker data directory to /mnt/docker
sudo systemctl stop docker
if [ ! -d "/mnt/docker" ]; then
    sudo mv /var/lib/docker /mnt/docker
else
    echo "/mnt/docker already exists, skipping move."
fi
# sudo systemctl start docker

CONFIG_FILE="/etc/docker/daemon.json"
TEMP_FILE=$(sudo mktemp)

# Ensure the Docker service is stopped before modifying the daemon.json
# sudo systemctl stop docker
sudo apt update && sudo apt install -y jq

# Ensure the file exists and is none empty
if [ ! -s "$CONFIG_FILE" ]; then
    echo "{}" | sudo tee "$CONFIG_FILE" > /dev/null
fi

# Use jq to modify the file (example: set "data-root" and enable "experimental" features and BuildKit)
sudo jq '. + {"data-root": "/mnt/docker", "features": {"buildkit": true}}' "$CONFIG_FILE" | sudo tee "$TEMP_FILE" > /dev/null

# Replace the original file with the modified one
sudo mv "$TEMP_FILE" "$CONFIG_FILE"

# Restart Docker to apply changes
sudo systemctl start docker
