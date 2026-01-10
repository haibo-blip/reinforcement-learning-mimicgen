# ManiFlow RL Training Guide

Complete guide for running RL fine-tuning of ManiFlow policies using PPO.

## 🚀 Quick Start

### 1. Basic Training
```bash
# Run with default configuration
python train_maniflow_rl.py

# Use specific config
python train_maniflow_rl.py --config-name=train_maniflow_pointcloud_rl
```

### 2. With Custom Dataset and Pretrained Policy
```bash
# Specify dataset and checkpoint paths
python train_maniflow_rl.py \
    task.dataset_path=/path/to/your/dataset.hdf5 \
    policy.checkpoint=/path/to/pretrained/checkpoint.ckpt
```

### 3. Custom Training Parameters
```bash
# Override training hyperparameters
python train_maniflow_rl.py \
    training.learning_rate=1e-4 \
    training.num_envs=16 \
    training.total_timesteps=2000000 \
    training.batch_size=1024
```

### 4. Environment-Specific Training
```bash
# Train on specific environment
python train_maniflow_rl.py \
    task.env_runner.env_meta.env_name=PickPlace \
    task.env_runner.env_config.env_name=PickPlace \
    wandb_run_name=ppo_pickplace_v1
```

## 📁 Configuration Structure

```yaml
# Main config file: config/train_maniflow_pointcloud_rl.yaml

policy:                    # ManiFlow policy configuration
  horizon: 16             # Action horizon
  n_action_steps: 8       # Action chunk size
  noise_method: "flow_sde" # Diffusion noise method
  add_value_head: true    # RL value head
  checkpoint: null        # Pretrained checkpoint path

task:
  dataset_path: null      # HDF5 dataset for shape_meta
  env_runner:             # RobomimicRLRunner configuration
    env_meta:
      env_name: "Lift"
    max_steps: 400
    num_episodes: 100

training:                 # PPO training parameters
  total_timesteps: 1000000
  num_envs: 8
  learning_rate: 3e-4
  clip_range: 0.2

advantage:               # GAE parameters
  gamma: 0.99
  gae_lambda: 0.95
```

## 🏗️ What Happens During Training

1. **Initialization**
   - Load pretrained ManiFlow policy (if provided)
   - Add RL value head for PPO
   - Create RobomimicRLRunner with RL data collection
   - Setup normalizers from dataset

2. **Rollout Collection** (RLinf pattern)
   - Use `policy.sample_actions()` to get chains and RL data
   - Store step-by-step: observations, actions, rewards, logprobs, values
   - Compute loss masks for post-termination steps

3. **Advantage Estimation**
   - Calculate GAE advantages and returns
   - Apply loss masking to exclude invalid steps

4. **PPO Training**
   - Mini-batch training with importance sampling
   - Clipped policy loss, value loss, entropy regularization
   - Masked loss computation following RLinf pattern

5. **Logging & Checkpointing**
   - W&B metrics: rewards, losses, KL divergence
   - Video recordings of rollouts
   - Regular checkpoint saves

## 🔧 Advanced Usage

### Custom Environment Configuration

Create a new config file for your specific environment:

```yaml
# config/train_my_env.yaml
defaults:
  - train_maniflow_pointcloud_rl

task:
  env_runner:
    env_meta:
      env_name: "MyCustomEnv"
    env_config:
      env_name: "MyCustomEnv"
      # ... custom environment parameters

# Override other parameters as needed
training:
  total_timesteps: 500000
  num_envs: 4
```

Then run:
```bash
python train_maniflow_rl.py --config-name=train_my_env
```

### Multi-GPU Training

```bash
# Use multiple GPUs (if implemented)
python train_maniflow_rl.py \
    training.num_envs=32 \
    training.batch_size=2048 \
    # Will automatically use available GPUs
```

### Hyperparameter Sweeps with W&B

```bash
# Run sweep
python train_maniflow_rl.py \
    -m \
    training.learning_rate=1e-4,3e-4,1e-3 \
    training.clip_range=0.1,0.2,0.3 \
    wandb_project=maniflow_sweep
```

## 📊 Monitoring Training

### W&B Dashboard
- **Rollout metrics**: Episode rewards, success rates, episode lengths
- **Training metrics**: Policy loss, value loss, entropy loss, KL divergence
- **Performance**: FPS, rollout time, gradient norms
- **Videos**: Episode recordings for visual monitoring

### Local Logs
```
outputs/
├── checkpoints/           # Model checkpoints
├── logs/                 # Training logs
├── videos/               # Episode recordings
└── configs/              # Saved configurations
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```bash
   # Reduce batch size and num_envs
   python train_maniflow_rl.py \
       training.batch_size=256 \
       training.num_envs=4
   ```

2. **Environment setup issues**
   - Check dataset path exists
   - Verify robomimic environment installation
   - Check shape_meta matches your environment

3. **Policy loading errors**
   - Ensure checkpoint path is correct
   - Verify policy architecture matches checkpoint
   - Check device compatibility (CPU vs GPU)

4. **Training instability**
   ```bash
   # Use more conservative hyperparameters
   python train_maniflow_rl.py \
       training.learning_rate=1e-4 \
       training.clip_range=0.1 \
       training.max_grad_norm=0.3
   ```

### Debug Mode

```bash
# Run with minimal configuration for debugging
python train_maniflow_rl.py \
    training.total_timesteps=1000 \
    training.num_envs=2 \
    training.num_steps_per_rollout=20 \
    training.log_interval=1 \
    use_wandb=false
```

## 📚 Implementation Details

- **RL Data Collection**: `RobomimicRLRunner` extends standard environment runner
- **Loss Masking**: Automatic masking of post-termination steps following RLinf
- **Chains Support**: Full diffusion sampling trajectory storage for training
- **PPO Implementation**: Standard PPO with importance sampling and clipping
- **Value Head**: Added to ManiFlow policy for state value estimation

## 🔗 Related Files

- `train_maniflow_rl.py` - Main training script
- `config/train_maniflow_pointcloud_rl.yaml` - Default configuration
- `equi_diffpo/rl_training/` - RL training modules
- `equi_diffpo/env_runner/robomimic_rl_runner.py` - RL-compatible environment runner