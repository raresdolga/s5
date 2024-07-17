#!/bin/bash
export TMPDIR=~/S5
# Wandb options
mkdir $TMPDIR/wandb_cache
export WANDB_CACHE_DIR=$TMPDIR/wandb_cache
mkdir $TMPDIR/wandb
export WANDB_DIR=$TMPDIR/wandb
# export WANDB_API_KEY=$(head -n 1 $HOME/wandb_api_key.txt)
export WANDB_SERVICE_WAIT=300
export WANDB_MODE="online"
export XLA_PYTHON_CLIENT_MEM_FRACTION=100

singularity run --nv img.sif pdm run wandb agent baesian-learning/sweep_rotrnn/792af51x
