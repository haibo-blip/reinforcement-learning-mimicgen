#!/usr/bin/env python3
"""
ManiFlow PPO Training Workspace
Complete PPO training pipeline for ManiFlow RL policy following RLinf pattern.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import wandb
from pathlib import Path
import json

from .maniflow_rollout_collector import ManiFlowRolloutCollector, ManiFlowRolloutBatch
from .maniflow_advantage_calculator import ManiFlowAdvantageCalculator, AdvantageConfig
from .rl_utils import masked_mean
from equi_diffpo.policy.maniflow.maniflow_pointcloud_rl_policy import ManiFlowRLPointcloudPolicy
from equi_diffpo.model.common.normalizer import LinearNormalizer


@dataclass
class PPOConfig:
    """PPO training configuration following RLinf pattern."""

    # Training
    total_timesteps: int = 1000000
    num_envs: int = 8
    num_steps_per_rollout: int = 256  # Steps per rollout per environment
    batch_size: int = 512             # Minibatch size for training
    num_epochs: int = 4               # PPO epochs per rollout
    max_grad_norm: float = 0.5        # Gradient clipping

    # PPO parameters
    clip_range: float = 0.2           # PPO clip parameter
    entropy_coef: float = 0        # Entropy loss coefficient
    value_coef: float = 0.5           # Value loss coefficient
    target_kl: float = 0.01           # Target KL divergence for early stopping

    # Critic warmup
    critic_warmup_rollouts: int = 0   # Number of rollouts to warmup critic before training actor

    # Learning rates
    learning_rate: float = 3e-4       # Adam learning rate
    lr_schedule: str = "linear"       # linear, constant, cosine
    warmup_steps: int = 10000         # LR warmup steps

    # Environment
    max_episode_length: int = 1000
    action_chunk_size: int = 8
    obs_chunk_size: int = 2

    # Logging
    log_interval: int = 10            # Log every N rollouts
    save_interval: int = 100          # Save every N rollouts
    eval_interval: int = 3           # Eval every N rollouts
    wandb_project: str = "maniflow_rl"
    wandb_run_name: str = "ppo_training"

    # Paths
    save_path: str = "checkpoints/maniflow_ppo"
    log_path: str = "logs/maniflow_ppo"

    def __post_init__(self):
        # Create directories
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        Path(self.log_path).mkdir(parents=True, exist_ok=True)

        print(f"🚀 PPO Config:")
        print(f"  - Total timesteps: {self.total_timesteps:,}")
        print(f"  - Num envs: {self.num_envs}")
        print(f"  - Steps per rollout: {self.num_steps_per_rollout}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Learning rate: {self.learning_rate}")
        print(f"  - Clip range: {self.clip_range}")


class ManiFlowPPOTrainer:
    """
    PPO trainer for ManiFlow RL policy following RLinf pattern.
    """

    def __init__(self,
                 policy: ManiFlowRLPointcloudPolicy,
                 env_runner,
                 config: PPOConfig,
                 advantage_config: AdvantageConfig,
                 device: str = "cuda",
                 use_wandb: bool = True):

        self.policy = policy
        self.env_runner = env_runner
        self.config = config
        self.device = torch.device(device)
        self.use_wandb = use_wandb

        # Move policy to device
        self.policy.to(self.device)

        # Create components
        self.rollout_collector = ManiFlowRolloutCollector(
            policy=self.policy,
            env_runner=self.env_runner,
            action_chunk_size=self.config.action_chunk_size,
            obs_chunk_size=self.config.obs_chunk_size,
            device=self.device
        )

        self.advantage_calculator = ManiFlowAdvantageCalculator(advantage_config)

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5
        )

        # Learning rate scheduler
        if self.config.lr_schedule == "linear":
            self.lr_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.config.total_timesteps // (self.config.num_envs * self.config.num_steps_per_rollout)
            )
        else:
            self.lr_scheduler = None

        # Training state
        self.global_step = 0
        self.rollout_count = 0
        self.total_episodes = 0

        # Metrics tracking
        self.training_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': [],
            'explained_variance': [],
            'learning_rate': [],
        }

        self.rollout_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'rollout_rewards': [],
        }

        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__
            )

        print(f"🚀 ManiFlow PPO Trainer initialized")
        print(f"  - Device: {self.device}")
        print(f"  - Parameters: {sum(p.numel() for p in self.policy.parameters()):,}")

    def train(self) -> None:
        """Main training loop following RLinf pattern."""
        print(f"🎯 Starting PPO training for {self.config.total_timesteps:,} timesteps")
        if self.config.critic_warmup_rollouts > 0:
            print(f"🔥 Critic warmup: {self.config.critic_warmup_rollouts} rollouts before training actor")

        start_time = time.time()
        steps_per_rollout = self.config.num_envs * self.config.num_steps_per_rollout

        while self.global_step < self.config.total_timesteps:
            if self.rollout_count % self.config.eval_interval == 0:
                self._run_evaluation()
            rollout_start_time = time.time()
            # Stage 1: Collect rollouts
            print(f"\n📊 Rollout {self.rollout_count + 1} (Step {self.global_step:,}/{self.config.total_timesteps:,})")
            rollout_batch = self.rollout_collector.collect_rollouts(
                num_episodes=None,  # Collect by steps, not episodes
                num_envs=self.config.num_envs
            )

            # Stage 2: Calculate advantages and returns
            rollout_batch = self.advantage_calculator.calculate_advantages_and_returns(rollout_batch)

            # Stage 3: PPO training (critic-only during warmup phase)
            critic_only = self.rollout_count < self.config.critic_warmup_rollouts
            training_stats = self.run_ppo_training(rollout_batch, critic_only=critic_only)

            # Update global step
            self.global_step += steps_per_rollout
            self.rollout_count += 1

            # Metrics and logging
            rollout_time = time.time() - rollout_start_time
            self._log_training_metrics(training_stats, rollout_batch, rollout_time)
            # Save checkpoint
            if self.rollout_count % self.config.save_interval == 0:
                self._save_checkpoint()
        total_time = time.time() - start_time
        print(f"🎉 Training completed in {total_time:.2f}s ({total_time/3600:.2f}h)")

        # Final save
        self._save_checkpoint(final=True)

        if self.use_wandb:
            wandb.finish()

    def run_ppo_training(self, rollout_batch: ManiFlowRolloutBatch, critic_only: bool = False) -> Dict[str, float]:
        """
        Run PPO training on rollout batch following RLinf pattern.

        Args:
            rollout_batch: Collected rollout data with advantages and returns
            critic_only: If True, only train critic (value head), skip actor updates

        Returns:
            Training statistics
        """
        if critic_only:
            print("🔥 Running critic warmup (value head only)...")
        else:
            print("🔥 Running PPO training...")

        # Convert to torch
        torch_batch = rollout_batch.to_torch(self.device)

        # Flatten batch dimensions for training
        flat_data = self._flatten_batch_data(torch_batch)
        total_samples = flat_data['advantages'].shape[0]

        # Training statistics
        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'kl_divergence': 0.0,
            'clip_fraction': 0.0,
            'explained_variance': 0.0,
        }

        num_updates = 0
        early_stop = False

        # Following RLinf pattern: Set policy to training mode only for gradient updates
        self.policy.train()

        for epoch in range(self.config.num_epochs):
            if early_stop:
                break

            # Shuffle data
            indices = torch.randperm(total_samples)

            for start in range(0, total_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, total_samples)
                batch_indices = indices[start:end]

                # Extract mini-batch (handle observation dict specially)
                mini_batch = {}
                for key, value in flat_data.items():
                    if key == 'observation':
                        # value is a dict of tensors
                        mini_batch[key] = {k: v[batch_indices] for k, v in value.items()}
                    else:
                        mini_batch[key] = value[batch_indices]

                # Forward pass through policy
                policy_outputs = self.policy.default_forward(
                    data={
                        'observation': mini_batch['observation'],
                        'chains': mini_batch['chains'],
                        'denoise_inds': mini_batch['denoise_inds'],
                        'prev_logprobs':mini_batch['prev_logprobs']
                    },
                    compute_values=True
                )

                # Compute losses
                loss_dict = self._compute_ppo_loss(mini_batch, policy_outputs, critic_only=critic_only)

                # Backward pass
                self.optimizer.zero_grad()
                loss_dict['total_loss'].backward()

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm
                )

                self.optimizer.step()

                # Update statistics
                for key in stats.keys():
                    if key in loss_dict:
                        stats[key] += loss_dict[key].item()

                num_updates += 1

                # Early stopping based on KL divergence
                if loss_dict['kl_divergence'].item() > 1.5 * self.config.target_kl:
                    print(f"  Early stopping at epoch {epoch}, KL={loss_dict['kl_divergence'].item():.4f}")
                    early_stop = True
                    break

        # Average statistics
        for key in stats.keys():
            stats[key] /= max(num_updates, 1)

        # Update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Update policy global step for noise annealing
        self.policy.set_global_step(self.global_step)

        # Following RLinf pattern: Restore policy to eval mode after training
        self.policy.eval()

        print(f"  ✅ PPO training completed: {num_updates} updates across {epoch + 1} epochs")
        print(f"    - Policy loss: {stats['policy_loss']:.6f}")
        print(f"    - Value loss: {stats['value_loss']:.6f}")
        print(f"    - KL divergence: {stats['kl_divergence']:.6f}")

        return stats

    def _compute_ppo_loss(self,
                         mini_batch: Dict[str, torch.Tensor],
                         policy_outputs: Dict[str, torch.Tensor],
                         critic_only: bool = False) -> Dict[str, torch.Tensor]:
        """Compute PPO loss following RLinf pattern with loss masking support.

        Args:
            mini_batch: Mini-batch of training data
            policy_outputs: Policy forward outputs
            critic_only: If True, only compute value loss (for critic warmup)
        """

        # Extract data
        old_logprobs = mini_batch['prev_logprobs']  # [batch, N, action_chunk, action_dim]
        advantages = mini_batch['advantages']        # [batch, action_chunk]
        returns = mini_batch['returns']             # [batch, 1]
        loss_mask = mini_batch.get('loss_mask')     # [batch, action_chunk] - mask for valid steps

        new_logprobs = policy_outputs['logprobs']    # [batch, action_chunk, action_dim]
        values = policy_outputs['values']            # [batch]
        entropy = policy_outputs['entropy']          # [batch, 1]

        # Value loss (always computed)
        values_expanded = values.unsqueeze(1)  # [batch, 1]
        value_loss_unmasked = (values_expanded - returns) ** 2

        # Apply loss mask for value loss (only for return dimension)
        if loss_mask is not None:
            # For value loss, we typically mask based on the first action chunk
            value_mask = loss_mask[:, 0:1].bool()  # [batch, 1]
            value_loss = masked_mean(value_loss_unmasked, value_mask)
        else:
            value_loss = value_loss_unmasked.mean()

        # If critic_only mode, skip policy loss computation
        if critic_only:
            total_loss = self.config.value_coef * value_loss
            return {
                'policy_loss': torch.tensor(0.0, device=values.device),
                'value_loss': value_loss,
                'entropy_loss': torch.tensor(0.0, device=values.device),
                'total_loss': total_loss,
                'kl_divergence': torch.tensor(0.0, device=values.device),
                'clip_fraction': torch.tensor(0.0, device=values.device),
                'explained_variance': torch.tensor(0.0, device=values.device),
            }

        # Average old_logprobs over N (denoising steps) to match new_logprobs shape
        # old_logprobs: [batch, N, action_chunk, action_dim] -> [batch, action_chunk, action_dim]
        old_logprobs = old_logprobs.mean(dim=1)
        new_logprobs = new_logprobs.mean(dim=1)
        # Sum over action_dim to get joint log probability
        old_logprobs_flat = old_logprobs.mean(dim=-1)  # [batch, action_chunk]
        new_logprobs_flat = new_logprobs.mean(dim=-1)  # [batch, action_chunk]
        # Importance sampling ratio
        log_ratio = new_logprobs_flat - old_logprobs_flat
        ratio = torch.exp(log_ratio)

        # Clipped policy loss (PPO objective)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range)
        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * clipped_ratio
        policy_loss_unmasked = torch.max(policy_loss_1, policy_loss_2)

        # Apply loss mask if available
        if loss_mask is not None:
            policy_loss = masked_mean(policy_loss_unmasked, loss_mask.bool())
        else:
            policy_loss = policy_loss_unmasked.mean()

        # Entropy loss (encourage exploration)
        entropy_unmasked = entropy
        if loss_mask is not None:
            # Use first action chunk mask for entropy
            entropy_mask = loss_mask[:, 0:1].bool()  # [batch, 1]
            entropy_loss = -masked_mean(entropy_unmasked, entropy_mask)
        else:
            entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (policy_loss +
                     self.config.value_coef * value_loss +
                     self.config.entropy_coef * entropy_loss)

        # Additional metrics (with masking)
        with torch.no_grad():
            kl_unmasked = (old_logprobs_flat - new_logprobs_flat) ** 2
            if loss_mask is not None:
                kl_divergence = masked_mean(kl_unmasked, loss_mask.bool())
                clip_fraction = masked_mean(
                    (torch.abs(ratio - 1.0) > self.config.clip_range).float(),
                    loss_mask.bool()
                )
            else:
                kl_divergence = kl_unmasked.mean()
                clip_fraction = (torch.abs(ratio - 1.0) > self.config.clip_range).float().mean()

            # Explained variance (masked)
            y_pred = values_expanded.flatten()
            y_true = returns.flatten()
            if loss_mask is not None:
                value_mask_flat = loss_mask[:, 0].bool()  # [batch]
                y_pred_masked = y_pred[value_mask_flat]
                y_true_masked = y_true[value_mask_flat]
                if len(y_true_masked) > 0:
                    var_y = torch.var(y_true_masked)
                    explained_var = 1 - torch.var(y_true_masked - y_pred_masked) / (var_y + 1e-8)
                else:
                    explained_var = torch.tensor(0.0)
            else:
                var_y = torch.var(y_true)
                explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)

        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'total_loss': total_loss,
            'kl_divergence': kl_divergence,
            'clip_fraction': clip_fraction,
            'explained_variance': explained_var,
        }

    def _flatten_batch_data(self, torch_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Flatten rollout batch for minibatch training."""
        flat_data = {}

        # Flatten observations
        obs_dict = {}
        for key, value in torch_batch['observation'].items():
            # [n_steps, batch, ...] -> [n_steps * batch, ...]
            obs_dict[key] = value.flatten(0, 1)
        flat_data['observation'] = obs_dict

        # Flatten other tensors
        for key in ['chains', 'denoise_inds', 'prev_logprobs', 'advantages', 'returns', 'loss_mask','x_stds','x_means']:
            if key in torch_batch and torch_batch[key] is not None:
                flat_data[key] = torch_batch[key].flatten(0, 1)

        return flat_data

    def _log_training_metrics(self,
                            training_stats: Dict[str, float],
                            rollout_batch: ManiFlowRolloutBatch,
                            rollout_time: float) -> None:
        """Log training metrics."""

        # Calculate rollout statistics
        mean_reward = rollout_batch.rewards.mean()
        total_reward = rollout_batch.rewards.sum()
        mean_episode_length = rollout_batch.rewards.shape[0]  # Approximate

        # Update metrics
        for key, value in training_stats.items():
            self.training_metrics[key].append(value)

        self.rollout_metrics['rollout_rewards'].append(float(total_reward))

        # Current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.training_metrics['learning_rate'].append(current_lr)

        # Calculate FPS (needed for both console and wandb logging)
        fps = (self.config.num_envs * self.config.num_steps_per_rollout) / rollout_time

        # Console logging
        if self.rollout_count % self.config.log_interval == 0:
            print(f"📊 Metrics (Rollout {self.rollout_count}):")
            print(f"  - Mean reward: {mean_reward:.3f}")
            print(f"  - Total reward: {total_reward:.1f}")
            print(f"  - Policy loss: {training_stats['policy_loss']:.6f}")
            print(f"  - Value loss: {training_stats['value_loss']:.6f}")
            print(f"  - KL divergence: {training_stats['kl_divergence']:.6f}")
            print(f"  - Learning rate: {current_lr:.2e}")
            print(f"  - FPS: {fps:.1f}")
            print(f"  - Time: {rollout_time:.2f}s")

        # Wandb logging
        if self.use_wandb:
            wandb.log({
                'rollout/mean_reward': mean_reward,
                'rollout/total_reward': total_reward,
                'rollout/episode_length': mean_episode_length,
                'train/policy_loss': training_stats['policy_loss'],
                'train/value_loss': training_stats['value_loss'],
                'train/entropy_loss': training_stats['entropy_loss'],
                'train/total_loss': training_stats['total_loss'],
                'train/kl_divergence': training_stats['kl_divergence'],
                'train/clip_fraction': training_stats['clip_fraction'],
                'train/explained_variance': training_stats['explained_variance'],
                'train/learning_rate': current_lr,
                'train/global_step': self.global_step,
                'train/rollout_count': self.rollout_count,
                'perf/fps': fps,
                'perf/rollout_time': rollout_time,
            }, step=self.global_step)

    def _save_checkpoint(self, final: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'rollout_count': self.rollout_count,
            'config': self.config.__dict__,
            'training_metrics': self.training_metrics,
            'rollout_metrics': self.rollout_metrics,
        }

        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        # Save checkpoint
        checkpoint_name = "final_checkpoint.pt" if final else f"checkpoint_{self.rollout_count}.pt"
        checkpoint_path = Path(self.config.save_path) / checkpoint_name
        torch.save(checkpoint, checkpoint_path)

        print(f"💾 Checkpoint saved: {checkpoint_path}")

    def _run_evaluation(self) -> None:
        """Run evaluation episodes following SFT workspace pattern."""
        print("🔍 Running evaluation...")

        # Set policy to evaluation mode
        self.policy.eval()

        eval_metrics = {}

        with torch.no_grad():
            # Run evaluation using the same env_runner as training but in eval mode
            # Use eval_mode=True to disable exploration noise
            eval_runner_log = self.env_runner.run(self.policy, eval_mode=True)

            # Add eval prefix to all metrics from runner
            for key, value in eval_runner_log.items():
                eval_metrics[f'eval/{key}'] = value

            # Log evaluation metrics
            print(f"📊 Evaluation Results:")
            for key, value in eval_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  - {key}: {value:.3f}")

            # Wandb logging
            if self.use_wandb:
                eval_metrics['eval/rollout_count'] = self.rollout_count
                eval_metrics['eval/global_step'] = self.global_step
                wandb.log(eval_metrics, step=self.global_step)

        # Set policy back to training mode
        self.policy.train()


def create_maniflow_ppo_trainer(
    policy: ManiFlowRLPointcloudPolicy,
    env_runner,
    config: Optional[PPOConfig] = None,
    advantage_config: Optional[AdvantageConfig] = None,
    device: str = "cuda",
    use_wandb: bool = True
) -> ManiFlowPPOTrainer:
    """
    Factory function to create a ManiFlowPPOTrainer.

    Args:
        policy: ManiFlow RL policy
        env_runner: Environment runner for collecting rollouts
        config: PPO training configuration (uses defaults if None)
        advantage_config: Advantage calculation configuration (uses defaults if None)
        device: Device to use for training
        use_wandb: Whether to use wandb logging

    Returns:
        Configured ManiFlowPPOTrainer instance
    """
    if config is None:
        config = PPOConfig()
    if advantage_config is None:
        advantage_config = AdvantageConfig()

    return ManiFlowPPOTrainer(
        policy=policy,
        env_runner=env_runner,
        config=config,
        advantage_config=advantage_config,
        device=device,
        use_wandb=use_wandb
    )


