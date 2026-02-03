#!/usr/bin/env python3
import torch
import dill

ckpt_10 = torch.load("data/outputs/2026.01.10/19.57.06_train_maniflow_pointcloud_nut_assembly_d0/checkpoints/latest.ckpt", pickle_module=dill, weights_only=False)
ckpt_19 = torch.load("data/outputs/2026.01.19/18.56.39_train_maniflow_pointcloud_nut_assembly_d0/checkpoints/latest.ckpt", pickle_module=dill, weights_only=False)

print("=" * 50)
print("Comparing state_dict keys:")
print("=" * 50)
keys_10 = set(ckpt_10['state_dicts']['model'].keys())
keys_19 = set(ckpt_19['state_dicts']['model'].keys())

if keys_10 == keys_19:
    print("✓ Model keys are identical")
else:
    print("✗ Model keys DIFFER!")
    print(f"Only in 1.10: {keys_10 - keys_19}")
    print(f"Only in 1.19: {keys_19 - keys_10}")

print("\n" + "=" * 50)
print("Comparing weight statistics:")
print("=" * 50)
model_10 = ckpt_10['state_dicts']['model']
model_19 = ckpt_19['state_dicts']['model']

# Compare a few key layers
layers_to_check = ['model.blocks.0.attn.qkv.weight', 'model.final_layer.linear.weight', 'obs_encoder.pointnet.mlp1.0.weight']
for layer in layers_to_check:
    if layer in model_10 and layer in model_19:
        w10 = model_10[layer]
        w19 = model_19[layer]
        print(f"\n{layer}:")
        print(f"  1.10: mean={w10.mean():.6f}, std={w10.std():.6f}, shape={w10.shape}")
        print(f"  1.19: mean={w19.mean():.6f}, std={w19.std():.6f}, shape={w19.shape}")
        if w10.shape == w19.shape:
            diff = (w10 - w19).abs().mean()
            print(f"  Diff: {diff:.6f}")

print("\n" + "=" * 50)
print("Comparing normalizer:")
print("=" * 50)
norm_10 = ckpt_10['state_dicts']['model'].get('normalizer.params_dict.action.input_stats.min')
norm_19 = ckpt_19['state_dicts']['model'].get('normalizer.params_dict.action.input_stats.min')
if norm_10 is not None and norm_19 is not None:
    print(f"1.10 action normalizer min: {norm_10}")
    print(f"1.19 action normalizer min: {norm_19}")
    if torch.equal(norm_10, norm_19):
        print("✓ Normalizer identical")
    else:
        print("✗ Normalizer DIFFERENT!")
else:
    print("Could not find normalizer in state_dict")
