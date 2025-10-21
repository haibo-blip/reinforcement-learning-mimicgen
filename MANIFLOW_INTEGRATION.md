# ManiFlow Policy Integration Guide for Equidiff

This guide explains how to use ManiFlow policies with the equidiff project for Robomimic/Mimicgen evaluation.

## What Was Copied

The following ManiFlow components have been integrated into equidiff:

### 1. Policy Files (`equi_diffpo/policy/maniflow/`)
- `maniflow_image_policy.py` - ManiFlow policy for RGB image observations
- `maniflow_pointcloud_policy.py` - ManiFlow policy for 3D point cloud observations

### 2. Model Files
**Diffusion Models** (`equi_diffpo/model/diffusion/`):
- `ditx.py` - DiTX transformer architecture (core of ManiFlow)
- `ditx_block.py` - DiTX transformer blocks
- `positional_embedding.py` - Sinusoidal positional embeddings
- `mask_generator.py` - Mask generation utilities

**Vision Encoders** (`equi_diffpo/model/vision_2d/`, `equi_diffpo/model/vision_3d/`):
- `timm_obs_encoder.py` - Image encoder using timm library
- `pointnet_extractor.py` - PointNet-based 3D encoder
- Supporting files for vision processing

### 3. Utility Files
- `equi_diffpo/model/common/sample_util.py` - Sampling utilities for consistency flow
- `equi_diffpo/common/model_util.py` - Model utilities

## Setup Instructions

### Step 1: Activate Environment
```bash
cd /home/haibo/Desktop/Developer/equidiff
conda activate equidiff
```

### Step 2: Run Setup Script
```bash
./setup_maniflow_policy.sh
```

This will:
- Install `timm` (required for vision encoders)
- Verify all imports work correctly
- Show you the integration structure

### Step 3: Install Additional Packages (if needed)
If you encounter import errors, you may need:
```bash
# If using R3M vision encoder
pip install gdown  # for downloading R3M weights

# If using language conditioning (optional)
pip install transformers sentence-transformers
```

## Using ManiFlow Policies

### Example 1: Import and Create Image Policy

```python
from equi_diffpo.policy.maniflow.manifflow_image_policy import ManiFlowTransformerImagePolicy
from equi_diffpo.model.vision_2d.timm_obs_encoder import TimmObsEncoder
import torch

# Define shape_meta (similar to equidiff format)
shape_meta = {
    'obs': {
        'agentview_image': {
            'shape': [3, 84, 84],
            'type': 'rgb'
        },
        'robot0_eef_pos': {
            'shape': [3],
            'type': 'low_dim'
        },
        'robot0_eef_quat': {
            'shape': [4],
            'type': 'low_dim'
        },
        'robot0_gripper_qpos': {
            'shape': [2],
            'type': 'low_dim'
        }
    },
    'action': {
        'shape': [7]  # [x, y, z, rot, rot, rot, gripper]
    }
}

# Create obs encoder
obs_encoder = TimmObsEncoder(
    shape_meta=shape_meta,
    model_name='resnet34',  # or 'r3m', 'vit_base_patch16_224'
    pretrained=True,
    frozen=False,
    global_pool='',
    downsample_ratio=32
)

# Create policy
policy = ManiFlowTransformerImagePolicy(
    shape_meta=shape_meta,
    obs_encoder=obs_encoder,
    horizon=16,
    n_action_steps=8,
    n_obs_steps=2,
    num_inference_steps=10,
    obs_as_global_cond=True,
    n_layer=12,
    n_head=8,
    n_emb=768,
    visual_cond_len=1024,
    # Consistency flow parameters
    flow_batch_ratio=0.75,
    consistency_batch_ratio=0.25,
    denoise_timesteps=10,
    sample_t_mode_flow="beta",
    sample_t_mode_consistency="discrete"
)

policy.cuda()
policy.eval()
```

### Example 2: Import and Create Pointcloud Policy

```python
from equi_diffpo.policy.maniflow.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy

shape_meta = {
    'obs': {
        'point_cloud': {
            'shape': [1024, 3],  # N points, xyz
            'type': 'point_cloud'
        },
        'agent_pos': {
            'shape': [14],
            'type': 'low_dim'
        }
    },
    'action': {
        'shape': [14]
    }
}

policy = ManiFlowTransformerPointcloudPolicy(
    shape_meta=shape_meta,
    horizon=16,
    n_action_steps=8,
    n_obs_steps=2,
    num_inference_steps=10,
    obs_as_global_cond=True,
    encoder_type="DP3Encoder",
    encoder_output_dim=128,
    visual_cond_len=128,
    n_layer=12,
    n_head=8,
    n_emb=768,
    use_pc_color=False,
    pointnet_type="pointnet",
    pointcloud_encoder_cfg={
        'in_channels': 3,
        'out_channels': 128,
        'use_layernorm': True,
        'final_norm': 'layernorm',
        'normal_channel': False,
        'num_points': 128,
        'pointwise': True
    }
)

policy.cuda()
policy.eval()
```

### Example 3: Inference

```python
import numpy as np

# Prepare observation (for image policy)
obs_dict = {
    'agentview_image': torch.randn(1, 2, 3, 84, 84).cuda(),  # B, T, C, H, W
    'robot0_eef_pos': torch.randn(1, 2, 3).cuda(),
    'robot0_eef_quat': torch.randn(1, 2, 4).cuda(),
    'robot0_gripper_qpos': torch.randn(1, 2, 2).cuda()
}

# Run inference
with torch.no_grad():
    action_dict = policy.predict_action(obs_dict)
    action = action_dict['action']  # Shape: (B, horizon, action_dim)

# Take first n_action_steps
action_to_execute = action[:, :policy.n_action_steps]  # (B, n_action_steps, action_dim)
```

## Adapting robomimic_image_runner.py

You can modify `equi_diffpo/env_runner/robomimic_image_runner.py` to use ManiFlow policies:

```python
# In robomimic_image_runner.py, update the run() method:

from equi_diffpo.policy.maniflow.maniflow_image_policy import ManiFlowTransformerImagePolicy

def run(self, policy: ManiFlowTransformerImagePolicy):
    # The policy should work with the existing runner infrastructure
    # since ManiFlow policies inherit from BaseImagePolicy pattern

    device = policy.device
    dtype = policy.dtype
    env = self.env

    # ... rest of existing code should work ...
```

## Key Differences from Equidiff Policies

1. **Consistency Flow Training**: ManiFlow uses a combination of flow matching and consistency training, enabling 1-2 step inference instead of 10+ steps.

2. **Inference Steps**: Set `num_inference_steps=1` or `num_inference_steps=2` for fast inference with ManiFlow (vs 10-100 for diffusion).

3. **Architecture**: Uses DiTX (Transformer-based) instead of U-Net.

4. **Vision Encoders**: Supports various encoders:
   - `resnet34` - Fast, lightweight
   - `r3m` - Pre-trained on robot data (recommended)
   - `vit_base_patch16_224` - Vision transformer
   - `efficientnet_b0/b3` - Efficient architectures

## Configuration Tips

### For Robomimic Tasks:

**Good starting parameters:**
```python
horizon = 16
n_action_steps = 8
n_obs_steps = 2
num_inference_steps = 10  # Can reduce to 1-2 after training
n_layer = 12
n_head = 8
n_emb = 768
visual_cond_len = 1024  # For images, 128 for pointclouds
```

**Memory optimization (if OOM):**
```python
n_layer = 6
n_emb = 512
visual_cond_len = 512
```

## Training (Optional)

If you want to train ManiFlow policies in equidiff:

1. The policies already have `compute_loss()` methods for training
2. You can adapt existing equidiff training scripts
3. Key parameters for consistency flow training:
   - `flow_batch_ratio=0.75` - 75% of batch uses flow matching
   - `consistency_batch_ratio=0.25` - 25% uses consistency training
   - `denoise_timesteps=10` - Number of discretization steps

## Troubleshooting

### Import Errors
```bash
# If you get "ModuleNotFoundError"
pip install timm einops

# If vision encoder fails
pip install gdown  # for R3M weights
```

### CUDA OOM
- Reduce `n_layer`, `n_emb`, or `visual_cond_len`
- Reduce batch size
- Use `torch.cuda.empty_cache()` between episodes

### Shape Mismatches
- Check that `shape_meta` matches your actual observation shapes
- For images: use (C, H, W) format
- For pointclouds: use (N, 3) or (N, 6) for xyz+rgb

## What's NOT Included

The following ManiFlow components were NOT copied (not needed for Robomimic):

- ❌ RoboTwin environments (requires Sapien 3.0)
- ❌ Language conditioning models (optional feature)
- ❌ Metaworld/Adroit/DexArt runners
- ❌ Training workspaces (can use existing equidiff training)
- ❌ Data generation scripts

## Next Steps

1. **Test the integration:**
   ```bash
   conda activate equidiff
   ./setup_maniflow_policy.sh
   ```

2. **Load a pre-trained ManiFlow checkpoint** (if you have one):
   ```python
   checkpoint = torch.load('path/to/maniflow_checkpoint.ckpt')
   policy.load_state_dict(checkpoint['state_dicts']['model'])
   ```

3. **Run evaluation on Robomimic:**
   - Modify `robomimic_image_runner.py` to instantiate ManiFlow policy
   - Use existing evaluation infrastructure
   - Compare with baseline policies

4. **Train from scratch** (optional):
   - Adapt `train.py` to use ManiFlow policy
   - Use Robomimic dataset loader
   - Train with consistency flow objective

## Support

If you encounter issues:
1. Check that all imports work: `python -c "from equi_diffpo.policy.maniflow.maniflow_image_policy import ManiFlowTransformerImagePolicy"`
2. Verify timm is installed: `pip show timm`
3. Check PyTorch version compatibility: `python -c "import torch; print(torch.__version__)"`

## Summary

✅ **What you can do now:**
- Use ManiFlow image policies with Robomimic environments
- Use ManiFlow pointcloud policies with Robomimic environments
- Leverage 1-2 step fast inference
- Benefit from consistency flow training

✅ **What works out-of-the-box:**
- All ManiFlow policy architectures
- DiTX transformer models
- Vision encoders (2D and 3D)
- Consistency flow sampling

⚠️ **What requires work:**
- Training scripts (can adapt from equidiff)
- Config files (need to create for your tasks)
- Dataset loaders (use existing Robomimic loaders)
