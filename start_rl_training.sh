python train_maniflow_rl.py \
    --config-path=equi_diffpo/config \
    --config-name=train_maniflow_pointcloud_rl \
    task_name=stack_three_d1 \
    rl_training.num_envs=16\
    policy.checkpoint="data/outputs/2026.01.31/01.47.07_train_maniflow_pointcloud_stack_three_d1/checkpoints/1.ckpt"
