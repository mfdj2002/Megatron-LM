#!/bin/sh
set -ex

MNT_DIR=/mnt
DEVICE="/dev/sda4"

# Check if the device is already mounted
if findmnt -rn $DEVICE $MNT_DIR >/dev/null; then
    echo "Device is already mounted."
else
    # Attempt to mount the device
    echo "Mounting device..."
    sudo mount $DEVICE $MNT_DIR
    echo "Device mounted."
fi

STAGE_DIR=$MNT_DIR/tmp

sudo mkdir -p $STAGE_DIR && sudo chmod 1777 $STAGE_DIR
cd $STAGE_DIR

# ##############################################################################
# install MLNX_OFED
################################################################################

# ENV MLNX_OFED_VERSION=4.9-7.1.0.0
MLNX_OFED_VERSION=5.4-3.7.5.0
# RUN apt-get update && \
sudo apt-get install -y --no-install-recommends libcap2

cd $STAGE_DIR &&
    wget -q -O - http://www.mellanox.com/downloads/ofed/MLNX_OFED-${MLNX_OFED_VERSION}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu20.04-x86_64.tgz | tar xzf - &&
    cd MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu20.04-x86_64 &&
    sudo ./mlnxofedinstall --force --user-space-only --add-kernel-support --without-fw-update --all -q &&
    cd ${STAGE_DIR} &&
    rm -rf ${STAGE_DIR}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu20.04-x86_64*

##############################################################################
# install CUDA driver and toolkit
##############################################################################

cd $STAGE_DIR &&
    wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run &&
    sudo sh cuda_12.1.0_530.30.02_linux.run --installpath=/mnt/cuda --tmpdir=/mnt/tmp -silent &&
    rm cuda_12.1.0_530.30.02_linux.run &&
    sudo dkms autoinstall && sudo modprobe nvidia &&
    sudo modprobe nvidia-peermem && sudo reboot
