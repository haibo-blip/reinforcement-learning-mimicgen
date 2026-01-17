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
    batch_size: int = 512             # Minibatch size for training
    max_grad_norm: float = 0.5        # Gradient clipping

    # PPO parameters
    clip_range: float = 0.2           # PPO clip parameter
    entropy_coef: float = 0        # Entropy loss coefficient
    value_coef: float = 0.5           # Value loss coefficient
    target_kl: float = 0.01           # Target KL divergence for early stopping

    # Critic warmup
    critic_warmup_rollouts: int = 0   # Number of rollouts to warmup critic before training actor

    # Learning rates
    learning_rate: float = 3e-4       # Actor learning rate
    value_lr: float = 1e-4            # Critic learning rate (typically higher than actor)
    lr_schedule: str = "linear"       # linear, constant, cosine
    warmup_steps: int = 10000         # LR warmup steps

    # Environment
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
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Actor LR: {self.learning_rate}, Critic LR: {self.value_lr}")
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

        # Setup optimizer with separate learning rates for actor and critic (like RLinf)
        self.optimizer = self._build_optimizer()

        # Learning rate scheduler
        # Estimate total rollouts: ~1600 samples per rollout (50 episodes * ~32 steps)
        estimated_samples_per_rollout = 1600
        estimated_total_rollouts = self.config.total_timesteps // estimated_samples_per_rollout
        if self.config.lr_schedule == "linear":
            self.lr_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=estimated_total_rollouts
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

        # 🔍 检查 visual encoder 是否被冻结
        self._check_visual_encoder_frozen()

    def _build_optimizer(self) -> optim.Optimizer:
        """
        Build optimizer with separate learning rates for actor and critic (following RLinf).

        Actor parameters use config.learning_rate (lower for stability)
        Critic/value_head parameters use config.value_lr (higher for faster value learning)
        """
        params_actor = []
        params_critic = []

        for name, param in self.policy.named_parameters():
            if param.requires_grad:
                # Check if this is a value head parameter
                if "value_head" in name or "value_mlp" in name:
                    params_critic.append(param)
                else:
                    params_actor.append(param)

        # Build parameter groups with different learning rates
        param_groups = []

        if len(params_actor) > 0:
            param_groups.append({
                "params": params_actor,
                "lr": self.config.learning_rate,
                "name": "actor"
            })

        if len(params_critic) > 0:
            param_groups.append({
                "params": params_critic,
                "lr": self.config.value_lr,
                "name": "critic"
            })

        optimizer = optim.AdamW(
            param_groups,
            eps=1e-5,
            weight_decay=0.01
        )

        # Log parameter counts
        actor_params = sum(p.numel() for p in params_actor)
        critic_params = sum(p.numel() for p in params_critic)
        print(f"\n📊 Optimizer setup (separate LRs like RLinf):")
        print(f"  - Actor params: {actor_params:,} (lr={self.config.learning_rate})")
        print(f"  - Critic params: {critic_params:,} (lr={self.config.value_lr})")

        return optimizer

    def _check_visual_encoder_frozen(self):
        """检查 visual encoder 的冻结状态"""
        print(f"\n🔍 Visual Encoder 冻结状态检查:")

        if hasattr(self.policy, 'obs_encoder'):
            encoder = self.policy.obs_encoder
            total_params = 0
            frozen_params = 0
            trainable_params = 0

            for name, param in encoder.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                else:
                    frozen_params += param.numel()

            frozen_ratio = frozen_params / total_params if total_params > 0 else 0

            print(f"  - Total encoder params: {total_params:,}")
            print(f"  - Frozen params: {frozen_params:,} ({frozen_ratio:.1%})")
            print(f"  - Trainable params: {trainable_params:,} ({1-frozen_ratio:.1%})")

            if frozen_ratio > 0.99:
                print(f"  ✅ Visual encoder 已冻结")
            elif frozen_ratio < 0.01:
                print(f"  ⚠️ Visual encoder 未冻结 (全部可训练)")
            else:
                print(f"  ⚠️ Visual encoder 部分冻结")
                # 打印前几个可训练的参数
                print(f"  可训练参数示例:")
                count = 0
                for name, param in encoder.named_parameters():
                    if param.requires_grad and count < 5:
                        print(f"    - {name}: {param.shape}")
                        count += 1
        else:
            print(f"  ⚠️ 未找到 obs_encoder")

        # 检查整体模型的参数分布
        print(f"\n🔍 整体模型参数分布:")
        total_all = sum(p.numel() for p in self.policy.parameters())
        trainable_all = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        print(f"  - Total: {total_all:,}")
        print(f"  - Trainable: {trainable_all:,} ({trainable_all/total_all:.1%})")
        print(f"  - Frozen: {total_all - trainable_all:,} ({(total_all-trainable_all)/total_all:.1%})")

    def train(self) -> None:
        """Main training loop following RLinf pattern."""
        print(f"🎯 Starting PPO training for {self.config.total_timesteps:,} timesteps")
        if self.config.critic_warmup_rollouts > 0:
            print(f"🔥 Critic warmup: {self.config.critic_warmup_rollouts} rollouts before training actor")

        start_time = time.time()

        while self.global_step < self.config.total_timesteps:
            rollout_start_time = time.time()
            # Stage 1: Collect rollouts
            print(f"\n📊 Rollout {self.rollout_count + 1} (Step {self.global_step:,}/{self.config.total_timesteps:,})")
            rollout_batch = self.rollout_collector.collect_rollouts(
                num_episodes=None,  # Collect by steps, not episodes
                num_envs=self.config.num_envs
            )

            # Get actual samples collected
            actual_samples = rollout_batch.n_chunk_steps * rollout_batch.batch_size

            # Stage 2: Calculate advantages and returns
            rollout_batch = self.advantage_calculator.calculate_advantages_and_returns(rollout_batch)

            # Stage 3: PPO training (critic-only during warmup phase)
            critic_only = self.rollout_count < self.config.critic_warmup_rollouts
            training_stats = self.run_ppo_training(rollout_batch, critic_only=critic_only)

            # Update global step with actual samples collected
            self.global_step += actual_samples
            self.rollout_count += 1

            # Metrics and logging
            rollout_time = time.time() - rollout_start_time
            self._log_training_metrics(training_stats, rollout_batch, rollout_time)
            # Save checkpoint
            if self.rollout_count % self.config.eval_interval == 0:
                self._run_evaluation()
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
        Run PPO training on rollout batch following Pi0.5/RLinf pattern.

        Pi0.5 style: 1 epoch per rollout, no early stopping.

        Args:
            rollout_batch: Collected rollout data with advantages and returns
            critic_only: If True, only train critic (value head), skip actor updates

        Returns:
            Training statistics
        """
        if critic_only:
            print("🔥 Running critic warmup (value head only)...")
        else:
            print("🔥 Running PPO training (Pi0.5 style: 1 epoch, no early stop)...")

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

        # Following RLinf pattern: Set policy to training mode only for gradient updates
        self.policy.train()

        # Pi0.5 style: only 1 epoch, no early stopping
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
                    'prev_logprobs': mini_batch['prev_logprobs']
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

        # Average statistics
        for key in stats.keys():
            stats[key] /= max(num_updates, 1)

        # Update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Update policy global step for noise annealing
        self.policy.set_global_step(self.rollout_count)

        # Following RLinf pattern: Restore policy to eval mode after training
        self.policy.eval()

        print(f"  ✅ PPO training completed: {num_updates} updates (1 epoch)")
        print(f"    - Policy loss: {stats['policy_loss']:.6f}")
        print(f"    - Value loss: {stats['value_loss']:.6f}")
        print(f"    - Clip fraction: {stats['clip_fraction']:.2%}")

        # Evaluate logprob changes for positive advantage samples
        if not critic_only:
            self._evaluate_positive_advantage_samples(flat_data)

        return stats

    def _evaluate_positive_advantage_samples(self, flat_data: Dict[str, torch.Tensor],
                                              max_samples: int = 50) -> None:
        """
        Evaluate logprob changes for samples with positive advantages.

        If PPO is working correctly, positive advantage samples should have
        INCREASED logprob after training (we want to increase probability of good actions).

        Args:
            flat_data: Flattened training data
            max_samples: Maximum number of positive samples to evaluate
        """
        print(f"\n  🔍 Evaluating positive advantage samples...")

        with torch.no_grad():
            # Get advantages and find positive ones
            advantages = flat_data['advantages']  # [total_samples, action_chunk]
            mean_advantages = advantages.mean(dim=-1)  # [total_samples]

            # Find indices of positive advantage samples
            positive_mask = mean_advantages > 0
            positive_indices = torch.where(positive_mask)[0]

            if len(positive_indices) == 0:
                print(f"    ⚠️ No positive advantage samples found!")
                return

            # Sample a subset for evaluation
            n_positive = len(positive_indices)
            n_eval = min(max_samples, n_positive)
            eval_indices = positive_indices[torch.randperm(n_positive)[:n_eval]]

            # Extract data for evaluation
            eval_batch = {}
            for key, value in flat_data.items():
                if key == 'observation':
                    eval_batch[key] = {k: v[eval_indices] for k, v in value.items()}
                else:
                    eval_batch[key] = value[eval_indices]

            # Get old logprobs
            old_logprobs = eval_batch['prev_logprobs']  # [n_eval, N, action_chunk, action_dim]
            old_logprobs_avg = old_logprobs.mean(dim=1)  # [n_eval, action_chunk, action_dim]
            old_logprobs_sum = old_logprobs_avg.sum(dim=-1)  # [n_eval, action_chunk]
            old_logprobs_mean = old_logprobs_sum.mean(dim=-1)  # [n_eval]

            # Forward pass to get new logprobs
            policy_outputs = self.policy.default_forward(
                data={
                    'observation': eval_batch['observation'],
                    'chains': eval_batch['chains'],
                    'denoise_inds': eval_batch['denoise_inds'],
                    'prev_logprobs': eval_batch['prev_logprobs']
                },
                compute_values=True
            )

            new_logprobs = policy_outputs['logprobs']  # [n_eval, action_chunk, action_dim]
            new_logprobs_avg = new_logprobs.mean(dim=1)  # Average over denoising if needed
            new_logprobs_sum = new_logprobs.sum(dim=-1)  # [n_eval, action_chunk]
            new_logprobs_mean = new_logprobs_sum.mean(dim=-1)  # [n_eval]

            # Calculate changes
            logprob_diff = new_logprobs_mean - old_logprobs_mean  # [n_eval]
            advantages_eval = mean_advantages[eval_indices]  # [n_eval]

            # Statistics
            increased = (logprob_diff > 0).float().mean().item()
            mean_diff = logprob_diff.mean().item()
            mean_adv = advantages_eval.mean().item()

            # Ratio (importance sampling)
            ratio = torch.exp(logprob_diff)
            mean_ratio = ratio.mean().item()

            print(f"    📊 Positive advantage samples ({n_eval}/{n_positive} evaluated):")
            print(f"       - Mean advantage: {mean_adv:.4f}")
            print(f"       - Old logprob mean: {old_logprobs_mean.mean().item():.4f}")
            print(f"       - New logprob mean: {new_logprobs_mean.mean().item():.4f}")
            print(f"       - Logprob diff (new - old): {mean_diff:.6f}")
            print(f"       - % samples with increased logprob: {increased*100:.1f}%")
            print(f"       - Mean ratio (π_new/π_old): {mean_ratio:.4f}")

            # Detailed breakdown by advantage magnitude
            high_adv_mask = advantages_eval > 1.0
            if high_adv_mask.any():
                high_adv_diff = logprob_diff[high_adv_mask].mean().item()
                high_adv_increased = (logprob_diff[high_adv_mask] > 0).float().mean().item()
                print(f"    📊 High advantage samples (adv > 1.0, n={high_adv_mask.sum().item()}):")
                print(f"       - Logprob diff: {high_adv_diff:.6f}")
                print(f"       - % increased: {high_adv_increased*100:.1f}%")

            # Check if PPO is working as expected
            if increased < 0.5:
                print(f"    ⚠️ WARNING: Less than 50% of positive advantage samples have increased logprob!")
                print(f"       This suggests PPO may not be learning correctly.")
            elif increased > 0.7:
                print(f"    ✅ Good: {increased*100:.1f}% of positive advantage samples have increased logprob.")

            # Also check negative advantage samples
            negative_mask = mean_advantages < 0
            negative_indices = torch.where(negative_mask)[0]

            if len(negative_indices) > 0:
                n_negative = len(negative_indices)
                n_eval_neg = min(max_samples, n_negative)
                eval_indices_neg = negative_indices[torch.randperm(n_negative)[:n_eval_neg]]

                # Extract negative samples
                eval_batch_neg = {}
                for key, value in flat_data.items():
                    if key == 'observation':
                        eval_batch_neg[key] = {k: v[eval_indices_neg] for k, v in value.items()}
                    else:
                        eval_batch_neg[key] = value[eval_indices_neg]

                old_logprobs_neg = eval_batch_neg['prev_logprobs'].mean(dim=1).sum(dim=-1).mean(dim=-1)

                policy_outputs_neg = self.policy.default_forward(
                    data={
                        'observation': eval_batch_neg['observation'],
                        'chains': eval_batch_neg['chains'],
                        'denoise_inds': eval_batch_neg['denoise_inds'],
                        'prev_logprobs': eval_batch_neg['prev_logprobs']
                    },
                    compute_values=True
                )

                new_logprobs_neg = policy_outputs_neg['logprobs'].sum(dim=-1).mean(dim=-1)
                logprob_diff_neg = new_logprobs_neg - old_logprobs_neg
                decreased = (logprob_diff_neg < 0).float().mean().item()
                mean_diff_neg = logprob_diff_neg.mean().item()
                mean_adv_neg = mean_advantages[eval_indices_neg].mean().item()

                print(f"    📊 Negative advantage samples ({n_eval_neg}/{n_negative} evaluated):")
                print(f"       - Mean advantage: {mean_adv_neg:.4f}")
                print(f"       - Logprob diff (new - old): {mean_diff_neg:.6f}")
                print(f"       - % samples with decreased logprob: {decreased*100:.1f}%")

                if decreased < 0.5:
                    print(f"    ⚠️ WARNING: Less than 50% of negative advantage samples have decreased logprob!")

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
        old_values = mini_batch['prev_values']      # [batch, 1] - for value clipping
        loss_mask = mini_batch.get('loss_mask')     # [batch, action_chunk] - mask for valid steps

        new_logprobs = policy_outputs['logprobs']    # [batch, action_chunk, action_dim]
        values = policy_outputs['values']            # [batch]
        entropy = policy_outputs['entropy']          # [batch, 1]

        # Value loss with clipping (like PPO policy clipping, following RLinf)
        values_expanded = values.unsqueeze(1)  # [batch, 1]

        # Unclipped value loss
        value_loss_unclipped = (values_expanded - returns) ** 2

        # Clipped value prediction: clip to be within clip_range of old value
        values_clipped = old_values + torch.clamp(
            values_expanded - old_values,
            -self.config.clip_range,
            self.config.clip_range
        )
        value_loss_clipped = (values_clipped - returns) ** 2

        # Take the maximum (more pessimistic) of clipped and unclipped loss
        value_loss_unmasked = torch.max(value_loss_unclipped, value_loss_clipped)

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
        # Sum over action_dim to get joint log probability (following RLinf pattern)
        old_logprobs_flat = old_logprobs.sum(dim=-1)  # [batch, action_chunk]
        new_logprobs_flat = new_logprobs.sum(dim=-1)  # [batch, action_chunk]
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
        for key in ['chains', 'denoise_inds', 'prev_logprobs', 'prev_values', 'advantages', 'returns', 'loss_mask','x_stds','x_means']:
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

        # Current learning rates (separate for actor and critic)
        actor_lr = self.optimizer.param_groups[0]['lr']
        critic_lr = self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) > 1 else actor_lr
        self.training_metrics['learning_rate'].append(actor_lr)

        # Calculate FPS using actual samples collected
        actual_samples = rollout_batch.n_chunk_steps * rollout_batch.batch_size
        fps = actual_samples / rollout_time

        # Console logging
        if self.rollout_count % self.config.log_interval == 0:
            print(f"📊 Metrics (Rollout {self.rollout_count}):")
            print(f"  - Mean reward: {mean_reward:.3f}")
            print(f"  - Total reward: {total_reward:.1f}")
            print(f"  - Policy loss: {training_stats['policy_loss']:.6f}")
            print(f"  - Value loss: {training_stats['value_loss']:.6f}")
            print(f"  - KL divergence: {training_stats['kl_divergence']:.6f}")
            print(f"  - Actor LR: {actor_lr:.2e}, Critic LR: {critic_lr:.2e}")
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
                'train/actor_lr': actor_lr,
                'train/critic_lr': critic_lr,
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


