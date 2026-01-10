# ManiFlow RL Training Pipeline

Complete PPO training pipeline for ManiFlow pointcloud policy following RLinf pattern.

## 🎯 Overview

This implementation provides a full RL training pipeline for the ManiFlow pointcloud policy, including:

- **Rollout Collection**: Parallel environment interaction with chain recording
- **Advantage Calculation**: GAE (Generalized Advantage Estimation) and other methods
- **PPO Training**: Policy optimization with importance sampling and entropy regularization
- **OpenPI Compatibility**: `default_forward` method compatible with RLinf training infrastructure

## 🏗️ Architecture

The pipeline follows the RLinf pattern exactly:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ Rollout         │    │ Advantage        │    │ PPO Training        │
│ Collection      │───►│ Calculation      │───►│                     │
│                 │    │                  │    │                     │
│ • sample_actions│    │ • GAE            │    │ • default_forward   │
│ • chains        │    │ • returns        │    │ • importance ratio  │
│ • denoise_inds  │    │ • normalization  │    │ • clipped loss      │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

## 🚀 Quick Start

### 1. Basic Training

```python
from equi_diffpo.rl_training import create_maniflow_ppo_trainer

# Define observation and action spaces
shape_meta = {
    'obs': {
        'robot0_eye_in_hand_image': {'shape': [3, 84, 84], 'type': 'rgb'},
        'point_cloud': {'shape': [1024, 6], 'type': 'point_cloud'},
        'robot0_eef_pos': {'shape': [3]},
        'robot0_eef_quat': {'shape': [4]},
        'robot0_gripper_qpos': {'shape': [2]},
    },
    'action': {'shape': [10]}
}

# Create trainer with default settings
trainer = create_maniflow_ppo_trainer(
    shape_meta=shape_meta,
    device="cuda"
)

# Start training
trainer.train()
```

### 2. Custom Configuration

```python
from equi_diffpo.rl_training import (
    ManiFlowPPOTrainer, PPOConfig, AdvantageConfig
)

# Custom PPO configuration
ppo_config = PPOConfig(
    total_timesteps=1000000,
    num_envs=16,
    num_steps_per_rollout=256,
    batch_size=1024,
    learning_rate=3e-4,
    clip_range=0.2,
    entropy_coef=0.01,
    wandb_project="my_maniflow_training"
)

# Custom advantage configuration
advantage_config = AdvantageConfig(
    gamma=0.99,
    gae_lambda=0.95,
    advantage_type="gae",
    normalize_advantages=True
)

# Create trainer with custom configs
trainer = create_maniflow_ppo_trainer(
    shape_meta=shape_meta,
    ppo_config=ppo_config.__dict__,
    advantage_config=advantage_config.__dict__,
    device="cuda"
)
```

### 3. With Real Environments

```python
# Replace ManiFlowDummyEnvRunner with your actual environment
from your_env_module import YourEnvRunner

env_runner = YourEnvRunner(
    num_envs=16,
    task_name="PickPlace",
    # ... other env config
)

trainer = ManiFlowPPOTrainer(
    policy=your_policy,
    env_runner=env_runner,
    config=ppo_config,
    advantage_config=advantage_config
)
```

## 📊 Key Features

### RLinf-Compatible Data Flow

The pipeline exactly matches RLinf's training pattern:

1. **Rollout Phase**: `sample_actions()` generates chains and denoise_inds
2. **Training Phase**: `default_forward()` re-evaluates same chains under current policy
3. **PPO Loss**: Importance sampling ratio = `exp(new_logprobs - old_logprobs)`

### OpenPI-Style default_forward

```python
# Input: data with chains and denoise_inds from rollout
policy_outputs = policy.default_forward(
    data={
        'observation': obs_dict,
        'chains': chains,           # [B, N+1, horizon, action_dim]
        'denoise_inds': denoise_inds # [B, N] - which timesteps to evaluate
    },
    compute_values=True
)

# Output: logprobs, values, entropy for PPO
print(policy_outputs['logprobs'].shape)  # [B, action_chunk, action_dim]
print(policy_outputs['values'].shape)    # [B]
print(policy_outputs['entropy'].shape)   # [B, 1]
```

### Advantage Calculation

Supports multiple advantage estimation methods:

- **GAE**: Generalized Advantage Estimation (recommended)
- **Monte Carlo**: Simple discounted returns
- **N-step**: N-step temporal difference

```python
# GAE with custom parameters
advantage_config = AdvantageConfig(
    advantage_type="gae",
    gamma=0.99,
    gae_lambda=0.95,
    normalize_advantages=True
)
```

### PPO Training

Full PPO implementation with:

- **Clipped Policy Loss**: Prevents large policy updates
- **Value Function Loss**: MSE loss for critic training
- **Entropy Regularization**: Encourages exploration
- **KL Divergence Monitoring**: Early stopping for stability

```python
# PPO loss computation
ratio = exp(new_logprobs - old_logprobs)  # Importance sampling
clipped_ratio = clamp(ratio, 1-ε, 1+ε)
policy_loss = -min(ratio * advantages, clipped_ratio * advantages)
```

## 📁 File Structure

```
equi_diffpo/rl_training/
├── __init__.py                           # Package exports
├── maniflow_rollout_collector.py        # Rollout collection with chains
├── maniflow_advantage_calculator.py     # GAE and other advantage methods
├── maniflow_ppo_workspace.py           # Complete PPO training loop
└── README.md                            # This file

equi_diffpo/policy/maniflow/
└── maniflow_pointcloud_rl_policy.py    # Policy with RL support

test_complete_rl_pipeline.py            # Comprehensive test suite
```

## 🔧 Configuration Options

### Policy Configuration

```python
policy_config = {
    'horizon': 16,                # Planning horizon
    'n_action_steps': 8,         # Action chunking
    'n_obs_steps': 2,            # Observation history
    'noise_method': "flow_sde",  # "flow_sde" or "flow_noise"
    'num_inference_steps': 10,   # Denoising steps
    'noise_level': 0.5,          # SDE noise level
    'noise_anneal': True,        # Noise annealing schedule
    'add_value_head': True,      # Enable value function
}
```

### Environment Configuration

```python
# For rollout collection
collector_config = {
    'max_steps_per_episode': 1000,
    'action_chunk_size': 8,
    'obs_chunk_size': 2,
}
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_complete_rl_pipeline.py
```

This tests:
- Individual component functionality
- RLinf-style data flow
- Complete training pipeline
- OpenPI compatibility

## 📈 Monitoring

The pipeline supports various monitoring options:

### Wandb Integration

```python
ppo_config = PPOConfig(
    wandb_project="maniflow_rl",
    wandb_run_name="experiment_1"
)
```

### Checkpointing

```python
# Automatic checkpointing every N rollouts
ppo_config = PPOConfig(
    save_interval=100,        # Save every 100 rollouts
    save_path="checkpoints/maniflow_ppo"
)

# Manual checkpoint loading
trainer.load_checkpoint("checkpoints/maniflow_ppo/checkpoint_500.pt")
```

### Logging

```python
ppo_config = PPOConfig(
    log_interval=10,          # Console logs every 10 rollouts
    eval_interval=50          # Evaluation every 50 rollouts
)
```

## 🎛️ Advanced Usage

### Custom Environment Integration

```python
class YourEnvRunner:
    def reset(self) -> Dict[str, np.ndarray]:
        """Return initial observations."""
        pass

    def step(self, actions: np.ndarray) -> Dict[str, np.ndarray]:
        """Step environments and return obs, rewards, dones."""
        pass
```

### Custom Advantage Function

```python
class CustomAdvantageCalculator(ManiFlowAdvantageCalculator):
    def _calculate_custom_advantages(self, rollout_batch, next_values):
        # Implement your advantage calculation
        pass
```

### Multi-GPU Training

```python
# Use DataParallel for multi-GPU training
if torch.cuda.device_count() > 1:
    trainer.policy = nn.DataParallel(trainer.policy)
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `batch_size` or `num_envs`
2. **Training Instability**: Lower `learning_rate` or `clip_range`
3. **Slow Convergence**: Increase `entropy_coef` or check reward scaling

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 References

- [RLinf Paper/Repo](https://github.com/your-rlinf-repo)
- [Pi0.5 Implementation](https://github.com/your-pi05-repo)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [GAE Paper](https://arxiv.org/abs/1506.02438)

## 🤝 Contributing

When adding new features:

1. Follow the RLinf data flow pattern
2. Maintain OpenPI compatibility
3. Add comprehensive tests
4. Update this documentation

## 📄 License

Same as parent project.