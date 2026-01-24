#!/usr/bin/env python3
import torch
import dill

ckpt_10 = torch.load("data/outputs/2026.01.10/19.57.06_train_maniflow_pointcloud_nut_assembly_d0/checkpoints/latest.ckpt", pickle_module=dill, weights_only=False)
ckpt_19 = torch.load("data/outputs/2026.01.19/18.56.39_train_maniflow_pointcloud_nut_assembly_d0/checkpoints/latest.ckpt", pickle_module=dill, weights_only=False)

print("=" * 50)
print("1.10 config - policy:")
print("=" * 50)
print(ckpt_10['cfg'].policy)

print("\n" + "=" * 50)
print("1.19 config - policy:")
print("=" * 50)
print(ckpt_19['cfg'].policy)

print("\n" + "=" * 50)
print("Comparing key parameters:")
print("=" * 50)
cfg_10 = ckpt_10['cfg'].policy
cfg_19 = ckpt_19['cfg'].policy

keys_to_compare = ['num_inference_steps', 'horizon', 'n_action_steps', 'n_obs_steps',
                   'sample_target_t_mode', 'denoise_timesteps', 'flow_batch_ratio',
                   'consistency_batch_ratio', 'n_layer', 'n_head', 'n_emb']

for key in keys_to_compare:
    val_10 = getattr(cfg_10, key, 'N/A')
    val_19 = getattr(cfg_19, key, 'N/A')
    match = "✓" if val_10 == val_19 else "✗ DIFFERENT"
    print(f"{key}: 1.10={val_10}, 1.19={val_19} {match}")
