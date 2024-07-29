XLA_PYTHON_CLIENT_MEM_FRACTION=100 CUDA_VISIBLE_DEVICES="0" pdm run python3 run_train.py --C_init=complex_normal --batchnorm=True --bidirectional=True \
                    --blocks=16 --bn_momentum=0.9 --bsz=32 --d_model=128 --dataset=pathx-classification \
                    --dt_min=0.0001 --epochs=75 --jax_seed=6429262 --lr_factor=3 --n_layers=6 \
                    --opt_config=BandCdecay --activation_fn half_glu1 --p_dropout=0.0 --ssm_lr_base=0.0006 --ssm_size_base=256 \
                    --warmup_end=1 --weight_decay=0.06 \
                    --conj_sym "False" --ssm_type "lru" --r_min 0.999 --r_max 0.9999 --max_phase 0.31 \
                    --wandb_project "fix_rotrnn" --wandb_entity "baesian-learning" --USE_WANDB "False"  # >'lru_mine_causal.log' 2>&1 & #   noBCdecay