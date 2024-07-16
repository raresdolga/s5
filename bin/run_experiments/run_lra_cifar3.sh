#!/bin/bash
XLA_PYTHON_CLIENT_MEM_FRACTION=100 CUDA_VISIBLE_DEVICES="0" pdm run python run_train.py --C_init=lecun_normal --batchnorm=True --bidirectional=False \
                    --blocks=3 --bsz=50 --clip_eigs=True --d_model=512 --dataset=lra-cifar-classification \
                    --epochs=250 --jax_seed=16416 --lr_factor=4.5 --n_layers=6 --opt_config=BfastandCdecay \
                    --p_dropout=0.1 --ssm_lr_base=0.001 --ssm_size_base=384 --warmup_end=1 --weight_decay=0.07 \
                    --wandb_project "sweep_rotrnn" --wandb_entity "baesian-learning" --ssm_type "rotblock" --USE_WANDB "False" # >'lru_mine_causal_099.log' 2>&1 &