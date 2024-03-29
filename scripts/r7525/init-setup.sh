#!/bin/bash
# set -ex
##############################################################################
# disable nouveau
##############################################################################

# Add the blacklist lines to the file
echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf

# Update initramfs
sudo update-initramfs -u

##############################################################################
#disable iommu
##############################################################################

status="off"
parameter="intel_iommu="$status
grub_file="/etc/default/grub"

if dmesg | grep iommu | grep -q "amd"; then
	parameter="amd_iommu="$status
fi

echo "Adding booting parameter ${parameter} to ${grub_file}"

old=$(grep GRUB_CMDLINE_LINUX_DEFAULT ${grub_file})

new=${old::-1}${parameter}\"

sudo sed -i "s/${old}/${new}/g" $grub_file

sudo update-grub

##############################################################################
# Generate SSH keys if not already present
##############################################################################

# Define the SSH key path
SSH_KEY_PATH="$HOME/.ssh/id_rsa"

# Create the user SSH directory, just in case it doesn't exist.
mkdir -p $HOME/.ssh && chmod 700 $HOME/.ssh

# Check if the SSH private key already exists
if [ ! -f "$SSH_KEY_PATH" ]; then
	echo "SSH key not found. Generating new SSH key..."

	# Retrieve the server-generated RSA private key.
	geni-get key >"$SSH_KEY_PATH"
	chmod 600 "$SSH_KEY_PATH"

	# Derive the corresponding public key portion.
	ssh-keygen -y -f "$SSH_KEY_PATH" >"${SSH_KEY_PATH}.pub"
else
	echo "SSH key already exists. Skipping key generation."
fi

# Ensure the authorized_keys2 file exists and is properly appended
touch $HOME/.ssh/authorized_keys2
grep -q -f "${SSH_KEY_PATH}.pub" $HOME/.ssh/authorized_keys2 || cat "${SSH_KEY_PATH}.pub" >>$HOME/.ssh/authorized_keys2

echo "SSH key setup complete."

##############################################################################
# mount the disk and change permission
##############################################################################

MNT_DIR=/mnt
DEVICE="/dev/sda4"

# sudo mkfs.ext4 /dev/sdb #will be quite different if on a different machine
sudo mkfs.ext4 $DEVICE
# # sudo mount /dev/sdb /mnt
sudo mount $DEVICE $MNT_DIR #for r7525

group=$(id -gn)

sudo chown -R $USER:$group $MNT_DIR

for dir in .vscode-server .debug .cache .local; do #tmp logs?
	sudo mkdir -p $MNT_DIR/$dir
	sudo chown -R $USER:$group $MNT_DIR/$dir
	rm -rf ~/$dir
	ln -s $MNT_DIR/$dir ~/$dir
done

##############################################################################
# install docker and change the data directory
##############################################################################

# Update and upgrade the system
sudo apt update && sudo apt install docker.io -y

# Check if the docker group exists
if ! getent group docker >/dev/null; then
	echo "Adding docker group..."
	sudo groupadd docker
else
	echo "Docker group already exists."
fi

# Check if the current user is already a member of the docker group
if ! id -nG "$USER" | grep -qw docker; then
	echo "Adding $USER to the docker group..."
	sudo usermod -aG docker "$USER"
else
	echo "$USER is already a member of the docker group."
fi

distribution=$(
	. /etc/os-release
	echo $ID$VERSION_ID
)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Move Docker data directory to /mnt/docker
sudo systemctl stop docker
if [ ! -d "/mnt/docker" ]; then
	sudo mv /var/lib/docker /mnt/docker
else
	echo "/mnt/docker already exists, skipping move."
fi

CONFIG_FILE="/etc/docker/daemon.json"
TEMP_FILE=$(sudo mktemp)

# Ensure the Docker service is stopped before modifying the daemon.json
sudo apt update && sudo apt install -y jq

# Ensure the file exists and is none empty
if [ ! -s "$CONFIG_FILE" ]; then
	echo "{}" | sudo tee "$CONFIG_FILE" >/dev/null
fi

# Use jq to modify the file (example: set "data-root" and enable "experimental" features and BuildKit)
sudo jq '. + {"data-root": "/mnt/docker", "features": {"buildkit": true}}' "$CONFIG_FILE" | sudo tee "$TEMP_FILE" >/dev/null

# Replace the original file with the modified one
sudo mv "$TEMP_FILE" "$CONFIG_FILE"

# Restart Docker to apply changes
sudo systemctl start docker

##############################################################################
# change github config
##############################################################################

git config --global user.name mfdj2002
git config --global user.email jf3516@columbia.edu

# Reboot the system
echo "Rebooting"

sudo reboot
