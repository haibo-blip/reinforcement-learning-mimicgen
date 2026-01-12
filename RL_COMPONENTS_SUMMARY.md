# ManiFlow RL Training Components Summary

This document provides an overview of all reinforcement learning (RL) components used in `train_maniflow_rl.py` for fine-tuning ManiFlow policies using PPO.

---

## Architecture Overview

```
train_maniflow_rl.py
        │
        ▼
┌─────────────────────────────────────────────┐
│   create_maniflow_rl_trainer_from_config()  │
│   (Factory function)                        │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│           ManiFlowPPOTrainer                │
│  ┌───────────────────────────────────────┐  │
│  │  • ManiFlowRLPointcloudPolicy         │  │
│  │  • ManiFlowRolloutCollector           │  │
│  │  • ManiFlowAdvantageCalculator        │  │
│  │  • Adam Optimizer + LR Scheduler      │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│         RobomimicRLRunner                   │
│   (Environment interaction)                 │
└─────────────────────────────────────────────┘
```

---

## 1. ManiFlowRLPointcloudPolicy

**File:** `equi_diffpo/policy/maniflow/maniflow_pointcloud_rl_policy.py`

The core policy model that extends ManiFlow for RL training.

### Key Features

| Feature | Description |
|---------|-------------|
| **Flow-based Action Sampling** | Uses flow matching (SDE-style) to generate action trajectories |
| **Value Head** | Neural network head for critic/value estimation |
| **Exploration Noise** | Supports `flow_sde` and `flow_noise` methods for exploration |
| **Log Probability Computation** | Computes log π(a\|s) for PPO training |
| **Action Chunking** | Predicts sequences of actions (horizon) and executes subsets (n_action_steps) |

### Architecture Components

```
Observations ──► DP3Encoder ──► DiTX Transformer ──► Actions
                     │                                  │
                     └──────► ValueHead ────────► V(s)
                     │
                     └──────► ExploreNoiseNet (optional)
```

### Noise Methods

1. **flow_sde** (default): Stochastic differential equation sampling with annealing
   - Adds fresh noise at each denoising step
   - Noise level can anneal from `noise_start` to `noise_end` over training

2. **flow_noise**: Learnable noise network
   - MLP-based noise prediction conditioned on observations
   - Outputs log standard deviations clamped to `noise_logvar_range`

### Key Methods

| Method | Purpose |
|--------|---------|
| `sample_actions()` | Sample actions with chains for RL training |
| `predict_action()` | Predict actions (with optional chain return) |
| `get_log_prob_value()` | Compute log probs, values, entropy from chains |
| `default_forward()` | Forward pass for PPO training |
| `get_step_prediction()` | Single denoising step prediction |

---

## 2. ManiFlowPPOTrainer

**File:** `equi_diffpo/rl_training/maniflow_ppo_workspace.py`

The main PPO training orchestrator following the RLinf pattern.

### Training Loop

```
while global_step < total_timesteps:
    1. Collect Rollouts ──► ManiFlowRolloutCollector
    2. Calculate Advantages ──► ManiFlowAdvantageCalculator
    3. PPO Training ──► run_ppo_training()
    4. Log Metrics ──► W&B / Console
    5. Save Checkpoints (periodic)
    6. Run Evaluation (periodic)
```

### PPOConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 1,000,000 | Total training steps |
| `num_envs` | 8 | Parallel environments |
| `num_steps_per_rollout` | 256 | Steps per rollout per env |
| `batch_size` | 512 | Minibatch size |
| `num_epochs` | 4 | PPO epochs per rollout |
| `clip_range` | 0.2 | PPO clipping parameter |
| `entropy_coef` | 0.0 | Entropy bonus coefficient |
| `value_coef` | 0.5 | Value loss coefficient |
| `learning_rate` | 3e-4 | Adam learning rate |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `target_kl` | 0.01 | KL divergence for early stopping |

### PPO Loss Components

```
Total Loss = Policy Loss + value_coef * Value Loss + entropy_coef * Entropy Loss
```

**Policy Loss (Clipped Surrogate):**
```
L_policy = -min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
where r_t = π_new(a|s) / π_old(a|s)
```

**Value Loss:**
```
L_value = (V_pred - V_target)²
```

---

## 3. ManiFlowRolloutCollector

**File:** `equi_diffpo/rl_training/maniflow_rollout_collector.py`

Collects rollout data from environment interactions.

### Data Structures

**ManiFlowRolloutStep** (single step):
- `observations`: Dict of observation arrays
- `actions`: [action_chunk, action_dim]
- `rewards`: [action_chunk]
- `dones`: [1]
- `prev_logprobs`: [action_chunk, action_dim]
- `prev_values`: [1]
- `forward_inputs`: chains, denoise_inds

**ManiFlowRolloutBatch** (batch):
- All fields have shape `[n_steps, batch_size, ...]`
- Includes `loss_mask` for handling episode boundaries
- `to_torch()` method for GPU transfer

### Collection Flow

```
1. Reset environments
2. For each step:
   a. policy.sample_actions(obs, mode="train")
   b. Extract actions, logprobs, values, chains
   c. Step environments
   d. Store step data
3. Convert to batch format
4. Compute loss masks
```

---

## 4. ManiFlowAdvantageCalculator

**File:** `equi_diffpo/rl_training/maniflow_advantage_calculator.py`

Computes advantages and returns for policy gradient updates.

### Advantage Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| **GAE** (default) | A_t = Σ (γλ)^l δ_{t+l} | Best bias-variance tradeoff |
| **Monte Carlo** | A_t = R_t - V(s_t) | High variance, zero bias |
| **N-Step** | A_t = Σ γ^k r_{t+k} + γ^n V - V(s_t) | Tunable bias-variance |

### GAE Formula

```
δ_t = r_t + γ * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
A_t = δ_t + γ * λ * (1 - done_{t+1}) * A_{t+1}
Returns = A_t + V(s_t)
```

### AdvantageConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `normalize_advantages` | True | Standardize advantages |
| `advantage_type` | "gae" | gae, monte_carlo, n_step |

---

## 5. Environment Runner

**File:** `equi_diffpo/env_runner/robomimic_rl_runner.py`

Handles parallel environment execution and data collection.

### Key Features
- Vectorized environment execution (n_envs parallel)
- Action chunking support
- Reward aggregation across action chunks
- Episode termination handling
- Video recording (optional)

### RL Data Output Format

```python
{
    'observations': Dict[str, np.ndarray],  # [n_steps, batch, obs_horizon, ...]
    'actions': np.ndarray,      # [n_steps, batch, action_chunk, action_dim]
    'rewards': np.ndarray,      # [n_steps, batch, action_chunk]
    'dones': np.ndarray,        # [n_steps, batch, 1]
    'prev_logprobs': np.ndarray,  # [n_steps, batch, action_chunk, action_dim]
    'prev_values': np.ndarray,    # [n_steps, batch, 1]
    'chains': np.ndarray,         # [n_steps, batch, N+1, horizon, action_dim]
    'denoise_inds': np.ndarray,   # [n_steps, batch, N]
}
```

---

## 6. Key Concepts

### Action Chunking

ManiFlow predicts action trajectories, not single actions:

```
horizon = 16        # Total predicted trajectory length
n_action_steps = 8  # Steps actually executed per prediction
n_obs_steps = 2     # Observation history length
```

### Flow Matching for RL

The policy uses flow matching (not standard diffusion):
- Time goes from t=1 (noise) to t=0 (clean actions)
- SDE-style sampling adds fresh noise at each step
- Enables log probability computation for PPO

### Chain-based Training

Stores full denoising chains for training:
```
chains: [B, N+1, horizon, action_dim]
denoise_inds: [B, N]  # Which steps to use for gradient
```

---

## 7. Training Pipeline Summary

```
┌──────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │   COLLECT   │ ──► │  CALCULATE  │ ──► │    TRAIN    │    │
│  │  ROLLOUTS   │     │  ADVANTAGES │     │     PPO     │    │
│  └─────────────┘     └─────────────┘     └─────────────┘    │
│        │                    │                   │            │
│        ▼                    ▼                   ▼            │
│  • Run policy in    • GAE/MC/N-step      • Compute loss     │
│    parallel envs    • Normalize          • Clip gradients   │
│  • Store chains     • Compute returns    • Update weights   │
│  • Track rewards                         • Early stop (KL)  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. Usage Example

```python
# Load config and create trainer
cfg = OmegaConf.load("config/train_maniflow_pointcloud_rl.yaml")

trainer = create_maniflow_rl_trainer_from_config(
    cfg=cfg,
    pretrained_policy_path="checkpoints/pretrained.ckpt",
    device="cuda"
)

# Run training
trainer.train()
```

**Command Line:**
```bash
python train_maniflow_rl.py --config-name=train_maniflow_pointcloud_rl

# Override parameters
python train_maniflow_rl.py training.learning_rate=1e-4 training.num_envs=16
```

---

## 9. File Structure

```
maniflow_mimicgen/
├── train_maniflow_rl.py                    # Main entry point
├── config/
│   └── train_maniflow_pointcloud_rl.yaml   # Hydra config
└── equi_diffpo/
    ├── policy/
    │   └── maniflow/
    │       └── maniflow_pointcloud_rl_policy.py  # RL Policy
    ├── rl_training/
    │   ├── create_maniflow_rl_trainer.py   # Factory function
    │   ├── maniflow_ppo_workspace.py       # PPO Trainer
    │   ├── maniflow_rollout_collector.py   # Rollout collection
    │   ├── maniflow_advantage_calculator.py # GAE computation
    │   └── rl_utils.py                     # Helper functions
    └── env_runner/
        └── robomimic_rl_runner.py          # Environment runner
```

---

## 10. References

This implementation follows patterns from:
- **RLinf/OpenPI**: Chain-based PPO for flow policies
- **Pi0.5**: SDE-style flow sampling with exploration noise
- **CleanRL**: Standard PPO implementation practices
