# ManiFlow Training Guide

This guide explains how to train ManiFlow policies (both point cloud and image versions) in the equidiff framework.

## Files Created

### Workspace Files
1. **Point Cloud**: `equi_diffpo/workspace/train_maniflow_pointcloud_workspace.py`
2. **Image**: `equi_diffpo/workspace/train_maniflow_image_workspace.py`

### Config Files
1. **Point Cloud**: `equi_diffpo/config/train_maniflow_pointcloud_workspace.yaml`
2. **Image**: `equi_diffpo/config/train_maniflow_image_workspace.yaml`

## Training Commands

### 1. ManiFlow Point Cloud Policy

Train with default settings (uses `mimicgen_pc_abs.yaml` task):
```bash
python equi_diffpo/workspace/train_maniflow_pointcloud_workspace.py \
    task_name=stack_d1 \
    n_demo=200
```

Train with custom settings:
```bash
python equi_diffpo/workspace/train_maniflow_pointcloud_workspace.py \
    task_name=your_task_name \
    n_demo=200 \
    training.device=cuda:0 \
    policy.n_layer=12 \
    policy.n_head=8 \
    policy.n_emb=768 \
    dataloader.batch_size=64
```

Debug mode (quick test):
```bash
python equi_diffpo/workspace/train_maniflow_pointcloud_workspace.py \
    task_name=stack_d1 \
    n_demo=10 \
    training.debug=True
```

### 2. ManiFlow Image Policy

Train with default settings (uses `mimicgen_abs.yaml` task):
```bash
python equi_diffpo/workspace/train_maniflow_image_workspace.py \
    task_name=stack_d1 \
    n_demo=200
```

Train with custom settings:
```bash
python equi_diffpo/workspace/train_maniflow_image_workspace.py \
    task_name=your_task_name \
    n_demo=200 \
    training.device=cuda:0 \
    policy.obs_encoder.model_name=resnet34 \
    policy.n_layer=12 \
    dataloader.batch_size=64
```

Debug mode:
```bash
python equi_diffpo/workspace/train_maniflow_image_workspace.py \
    task_name=stack_d1 \
    n_demo=10 \
    training.debug=True
```

## Key Configuration Parameters

### ManiFlow-Specific Parameters

```yaml
policy:
  # Architecture
  n_layer: 12              # Number of DiTX transformer layers
  n_head: 8                # Number of attention heads
  n_emb: 768               # Embedding dimension
  visual_cond_len: 1024    # Visual conditioning length

  # Flow training
  flow_batch_ratio: 0.75                    # Ratio of flow training samples
  consistency_batch_ratio: 0.25             # Ratio of consistency samples
  denoise_timesteps: 10                     # Number of denoising steps
  sample_t_mode_flow: "beta"                # Flow time sampling mode
  sample_t_mode_consistency: "discrete"     # Consistency time sampling
  sample_target_t_mode: "relative"          # Target time mode

  # Inference
  num_inference_steps: 10                   # Steps during inference
  obs_as_global_cond: True                  # Use obs as global conditioning
```

### Vision Encoder (Image Policy Only)

```yaml
policy:
  obs_encoder:
    model_name: resnet18           # Options: resnet18, resnet34, resnet50
    pretrained: False              # Use ImageNet pretrained weights
    downsample_ratio: 32           # Spatial downsampling (16 or 32)
    feature_aggregation: spatial_embedding  # How to aggregate features
    use_group_norm: True           # Replace BatchNorm with GroupNorm
```

### Point Cloud Encoder (Point Cloud Policy Only)

```yaml
policy:
  encoder_type: "DP3Encoder"       # Point cloud encoder architecture
  pcd_input_channel: 6             # xyz + rgb
  pcd_down_sample_num: 512         # Number of points after downsampling
  encoder_output_dim: 512          # Encoder output dimension
```

### Training Parameters

```yaml
training:
  device: "cuda:0"                 # GPU device
  seed: 42                         # Random seed
  num_epochs: 250                  # Number of epochs (auto-calculated)
  lr_scheduler: cosine             # Learning rate schedule
  lr_warmup_steps: 500             # Warmup steps
  use_ema: True                    # Use exponential moving average
  gradient_accumulate_every: 1     # Gradient accumulation steps

  # Evaluation frequency
  rollout_every: 5                 # Run policy evaluation every N epochs
  val_every: 1                     # Run validation every N epochs
  checkpoint_every: 5              # Save checkpoint every N epochs
```

### Dataset Parameters

```yaml
horizon: 16                        # Action sequence length
n_obs_steps: 2                     # Number of observation frames
n_action_steps: 8                  # Number of actions to execute
n_demo: 200                        # Number of demonstrations to use

dataloader:
  batch_size: 64                   # Training batch size
  num_workers: 8                   # Data loading workers
```

## Data Preparation

### Point Cloud Data
The point cloud policy expects data in this format:
- Task config: `mimicgen_pc_abs.yaml`
- Dataset path: `data/robomimic/datasets/${task_name}/${task_name}_pc_abs.hdf5`
- Point cloud shape: `[1024, 6]` (1024 points with xyz+rgb)

### Image Data
The image policy expects data in this format:
- Task config: `mimicgen_abs.yaml`
- Dataset path: `data/robomimic/datasets/${task_name}/${task_name}_abs.hdf5`
- Image shapes: `[3, 84, 84]` for RGB images

## Output Structure

Training outputs are saved to:
```
data/outputs/YYYY.MM.DD/HH.MM.SS_train_maniflow_{pointcloud|image}_{task_name}/
├── checkpoints/
│   ├── latest.ckpt
│   └── epoch=XXXX-test_mean_score=X.XXX.ckpt
├── logs.json.txt
└── wandb/
```

## Monitoring Training

Training is logged to Weights & Biases (wandb). Key metrics:
- `train_loss`: Training loss per step
- `val_loss`: Validation loss per epoch
- `test_mean_score`: Success rate on test episodes
- `lr`: Current learning rate

## Resuming Training

Training automatically resumes from the last checkpoint if `training.resume=True` (default):
```bash
python equi_diffpo/workspace/train_maniflow_pointcloud_workspace.py \
    task_name=stack_d1 \
    training.resume=True
```

## Multi-GPU Training

To use a specific GPU:
```bash
python equi_diffpo/workspace/train_maniflow_pointcloud_workspace.py \
    task_name=stack_d1 \
    training.device=cuda:1
```

## Common Issues

### 1. Out of Memory
Reduce batch size:
```bash
dataloader.batch_size=32
```

### 2. Transforms Parameter Missing
The config includes `transforms: null` by default. You can add data augmentation:
```yaml
policy:
  obs_encoder:
    transforms:
      - type: RandomCrop
        ratio: 0.95
```

### 3. Shape Mismatches
Ensure your dataset's observation shapes match the task config's `shape_meta`.

## Performance Tips

1. **Batch Size**: Start with 64, increase if you have enough GPU memory
2. **Learning Rate**: Default 1e-4 works well, adjust if needed
3. **EMA**: Keep `use_ema=True` for better test performance
4. **Number of Layers**: 12 layers for best performance, 6-8 for faster training
5. **Point Cloud Downsampling**: 512 points is a good balance

## Example: Full Training Run

```bash
# Point cloud policy on stack task
python equi_diffpo/workspace/train_maniflow_pointcloud_workspace.py \
    task_name=stack_d1 \
    n_demo=200 \
    horizon=16 \
    n_obs_steps=2 \
    n_action_steps=8 \
    training.device=cuda:0 \
    dataloader.batch_size=64 \
    policy.n_layer=12 \
    policy.n_head=8 \
    policy.n_emb=768 \
    logging.project=my_maniflow_experiment

# Image policy on the same task
python equi_diffpo/workspace/train_maniflow_image_workspace.py \
    task_name=stack_d1 \
    n_demo=200 \
    horizon=16 \
    n_obs_steps=2 \
    n_action_steps=8 \
    training.device=cuda:1 \
    dataloader.batch_size=64 \
    policy.obs_encoder.model_name=resnet34 \
    logging.project=my_maniflow_experiment
```

## Differences from Original ManiFlow

1. **Workspace Structure**: Adapted to equidiff's BaseWorkspace pattern
2. **Dataset Integration**: Uses equidiff's dataset classes
3. **Optimizer**: Simplified to single AdamW optimizer (original had separate optimizers)
4. **Vision Encoder**: Uses equidiff's TimmObsEncoder (compatible with original)
5. **Point Cloud Encoder**: Uses DP3Encoder from equidiff

## Next Steps

After training:
1. Check wandb for training curves and success rates
2. Best checkpoints are saved based on `test_mean_score`
3. Load checkpoint for evaluation or deployment
4. Use the trained policy with `env_runner` for rollouts
