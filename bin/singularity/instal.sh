#!/bin/bash

CWD=$(pwd)
cd ~

# Ensure repositories are up-to-date
sudo apt-get update
# Install debian packages for dependencies
sudo apt-get install -y \
    autoconf \
    automake \
    cryptsetup \
    fuse2fs \
    git \
    fuse \
    libfuse-dev \
    libglib2.0-dev \
    libseccomp-dev \
    libtool \
    pkg-config \
    runc \
    squashfs-tools \
    squashfs-tools-ng \
    uidmap \
    wget \
    zlib1g-dev

# Install Go
VERSION=1.21.11
OS=linux
ARCH=amd64  # change this as you need
wget -O /tmp/go${VERSION}.${OS}-${ARCH}.tar.gz \
  https://dl.google.com/go/go${VERSION}.${OS}-${ARCH}.tar.gz
sudo tar -C /usr/local -xzf /tmp/go${VERSION}.${OS}-${ARCH}.tar.gz

# Add /usr/local/go/bin to the PATH environment variable
PATH=$PATH:/usr/local/go/bin

# Clone singulariy repository
git clone --recurse-submodules https://github.com/sylabs/singularity.git
cd singularity

# Checkout a specific version
git checkout --recurse-submodules v4.1.3

# Build Singularity
./mconfig
make -C builddir
sudo make -C builddir install

# Verify the installation
singularity --version

# # Build the image
# cd $CWD
# sudo singularity build image.sif singularity/image.def

# # Install requirements
# singularity run image.sif pdm install

# # Prepare the data
# singularity run image.sif pdm run prepare_lra