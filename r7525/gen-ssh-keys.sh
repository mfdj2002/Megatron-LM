#!/bin/bash

# ##############################################################################
# # generate ssh keys
# ##############################################################################

# # Create the user SSH directory, just in case.
# mkdir -p $HOME/.ssh && chmod 700 $HOME/.ssh

# # Retrieve the server-generated RSA private key.
# geni-get key > $HOME/.ssh/id_rsa
# chmod 600 $HOME/.ssh/id_rsa

# # Derive the corresponding public key portion.
# ssh-keygen -y -f $HOME/.ssh/id_rsa > $HOME/.ssh/id_rsa.pub

# # If you want to permit login authenticated by the auto-generated key,
# # then append the public half to the authorized_keys2 file:
# touch $HOME/.ssh/authorized_keys2

# grep -q -f $HOME/.ssh/id_rsa.pub $HOME/.ssh/authorized_keys2 || cat $HOME/.ssh/id_rsa.pub >> $HOME/.ssh/authorized_keys2

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
