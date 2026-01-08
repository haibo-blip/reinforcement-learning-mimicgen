# ManiFlow RL Training

This directory contains the RL training implementation for ManiFlow policies, including rollout collection and PPO training.

## Quick Start

### 1. Test Rollout Collection

Test the rollout collector without training:

```bash
cd maniflow_mimicgen
python test_rl_rollout.py
```

### 2. Test Training Loop

Test a single epoch of RL training:

```bash
cd maniflow_mimicgen
python test_rl_rollout.py training
```

### 3. Full RL Training

Run full RL training with default configuration:

```bash
cd maniflow_mimicgen
python scripts/train_rl.py
```

### 4. Custom Configuration

Override configuration parameters:

```bash
cd maniflow_mimicgen
python scripts/train_rl.py \
    task_name=stack_d0 \
    rl_training.num_epochs=50 \
    rl_training.rollout_collector.n_episodes_per_batch=32
```

### 5. With Pretrained Checkpoint

Train from a pretrained checkpoint:

```bash
cd maniflow_mimicgen
python scripts/train_rl.py \
    pretrained_policy_path=/path/to/checkpoint.ckpt
```

## Configuration

The RL training configuration extends the base ManiFlow training config with additional RL-specific parameters:

### Key Configuration Sections

1. **Rollout Collection** (`rl_training.rollout_collector`)
   - `n_episodes_per_batch`: Episodes to collect per training iteration
   - `n_envs`: Number of parallel environments
   - `max_episode_steps`: Maximum steps per episode
   - `render_episodes`: Number of episodes to record as videos

2. **PPO Training** (`rl_training.ppo_trainer`)
   - `clip_ratio`: PPO clipping parameter (0.2)
   - `ppo_epochs`: Number of PPO update epochs per batch (4)
   - `mini_batch_size`: Mini-batch size for PPO updates
   - `gamma`: Discount factor for GAE (0.99)
   - `gae_lambda`: GAE lambda parameter (0.95)

3. **Learning Rates**
   - `policy_lr`: Policy learning rate (1e-5 for fine-tuning)
   - `critic_lr`: Critic learning rate (1e-4)

## File Structure

```
equi_diffpo/rl_training/
├── __init__.py
├── rl_rollout_collector.py    # Multi-threaded rollout collection
├── ppo_trainer.py             # PPO training with GAE
└── rl_workspace.py            # Main RL training workspace

equi_diffpo/config/
└── train_maniflow_pointcloud_rl.yaml  # RL training configuration

scripts/
└── train_rl.py               # Main training script

test_rl_rollout.py            # Test script
```

## Usage Examples

### Testing Different Environments

```bash
# Test with different task
python test_rl_rollout.py task_name=lift_d0

# Test with more environments
python test_rl_rollout.py rl_training.rollout_collector.n_envs=32
```

### Hyperparameter Tuning

```bash
# Adjust PPO parameters
python scripts/train_rl.py \
    rl_training.ppo_trainer.clip_ratio=0.1 \
    rl_training.ppo_trainer.entropy_coef=0.001 \
    rl_training.ppo_trainer.policy_lr=5e-6

# Adjust GAE parameters
python scripts/train_rl.py \
    rl_training.ppo_trainer.gamma=0.95 \
    rl_training.ppo_trainer.gae_lambda=0.9
```

### Multi-GPU Training

```bash
# Use specific GPU
python scripts/train_rl.py training.device=cuda:1

# For multi-GPU, you may need to modify the code to use DataParallel
```

## Monitoring

The training automatically logs to Weights & Biases with metrics including:
- Episode rewards and lengths
- PPO training metrics (policy loss, value loss, clip fraction)
- GAE statistics (advantages, returns)
- Video recordings of episodes

To disable logging:
```bash
python scripts/train_rl.py logging.mode=disabled
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch sizes
   python scripts/train_rl.py \
       rl_training.rollout_collector.n_episodes_per_batch=8 \
       rl_training.ppo_trainer.mini_batch_size=32
   ```

2. **Environment Initialization Errors**
   - Make sure robomimic dataset exists at the specified path
   - Check that the task configuration matches available environments

3. **Policy Loading Errors**
   - Ensure pretrained checkpoint path is correct
   - Verify checkpoint contains compatible policy configuration

### Performance Tips

1. **Optimize Environment Count**
   - Use `n_envs` equal to the number of CPU cores
   - Balance between CPU and GPU utilization

2. **Batch Size Tuning**
   - Increase `n_episodes_per_batch` for more stable gradients
   - Increase `mini_batch_size` for faster training (if GPU memory allows)

3. **Learning Rate Scheduling**
   - Start with lower learning rates for fine-tuning pretrained models
   - Consider implementing learning rate decay for longer training runs