#!/usr/bin/env python3
"""
ManiFlow NFT (Noise-Free Training) Workspace
Replaces PPO actor loss with NFT's x0-prediction contrastive loss.
Keeps GAE advantage estimation and value head (critic) training from PPO.

Ported from DiffusionNFT/scripts/train_nft_sd3.py
"""

import os
import copy
import time
import math
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
class NFTConfig:
    """NFT training configuration."""

    # Training
    total_timesteps: int = 1000000
    num_envs: int = 8
    batch_size: int = 32
    gradient_accumulate_every: int = 64
    max_grad_norm: float = 0.5

    # NFT-specific
    beta: float = 1.0              # NFT beta (pos/neg blend strength)
    beta_kl: float = 0.01          # KL regularization coefficient
    adv_clip_max: float = 5.0      # Advantage clipping range
    num_train_timesteps: int = 1   # Number of timesteps to sample per training step
    old_model_decay: float = 0.5   # Max EMA decay for old_model
    old_model_decay_rate: float = 0.001  # Decay ramp-up rate

    # Critic
    value_coef: float = 0.5
    clip_range: float = 0.2        # Value clipping range
    critic_warmup_rollouts: int = 4
    critic_warmup_epochs: int = 3

    # Learning rates
    learning_rate: float = 1e-5    # Actor LR
    value_lr: float = 1e-4         # Critic LR
    lr_schedule: str = "linear"
    warmup_steps: int = 10000

    # Environment
    action_chunk_size: int = 8
    obs_chunk_size: int = 2

    # Logging
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 3
    wandb_project: str = "maniflow_rl"
    wandb_run_name: str = "nft_training"

    # Rollout video logging
    rollout_video_interval: int = 20
    n_rollout_videos: int = 4

    # Episode counts
    train_n_episodes: int = 50
    eval_n_episodes: int = 50

    # Paths
    save_path: str = "checkpoints/maniflow_nft"
    log_path: str = "logs/maniflow_nft"

    def __post_init__(self):
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        Path(self.log_path).mkdir(parents=True, exist_ok=True)

        print(f"NFT Config:")
        print(f"  - Total timesteps: {self.total_timesteps:,}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Gradient accumulate: {self.gradient_accumulate_every}")
        print(f"  - Actor LR: {self.learning_rate}, Critic LR: {self.value_lr}")
        print(f"  - NFT beta: {self.beta}, beta_kl: {self.beta_kl}")
        print(f"  - adv_clip_max: {self.adv_clip_max}")
        print(f"  - old_model_decay: {self.old_model_decay}, rate: {self.old_model_decay_rate}")


class ManiFlowNFTTrainer:
    """
    NFT trainer for ManiFlow RL policy.
    Replaces PPO actor loss with NFT's contrastive x0-prediction loss.
    Keeps GAE advantage estimation and critic training unchanged.
    """

    def __init__(self,
                 policy: ManiFlowRLPointcloudPolicy,
                 env_runner,
                 config: NFTConfig,
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

        # Create old_model and ref_model as frozen deep copies of policy.model (DiTX)
        self.old_model = copy.deepcopy(policy.model)
        self.old_model.requires_grad_(False)
        self.old_model.eval()
        self.old_model.to(self.device)

        self.ref_model = copy.deepcopy(policy.model)
        self.ref_model.requires_grad_(False)
        self.ref_model.eval()
        self.ref_model.to(self.device)

        # Create components
        self.rollout_collector = ManiFlowRolloutCollector(
            policy=self.policy,
            env_runner=self.env_runner,
            action_chunk_size=self.config.action_chunk_size,
            obs_chunk_size=self.config.obs_chunk_size,
            device=self.device
        )

        self.advantage_calculator = ManiFlowAdvantageCalculator(advantage_config)

        # Setup optimizer with separate learning rates for actor and critic
        self.optimizer = self._build_optimizer()

        # Learning rate scheduler
        horizon = self.policy.horizon
        max_steps = self.env_runner.max_steps
        train_n_episodes = self.config.train_n_episodes
        n_chunk_steps = math.ceil(max_steps / horizon)
        samples_per_rollout = n_chunk_steps * train_n_episodes
        total_rollouts = self.config.total_timesteps // samples_per_rollout

        if self.config.lr_schedule == "linear":
            self.lr_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_rollouts
            )
        else:
            self.lr_scheduler = None

        # Training state
        self.global_step = 0
        self.rollout_count = 0

        # Metrics tracking
        self.training_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'kl_loss': [],
            'total_loss': [],
            'learning_rate': [],
        }

        self.rollout_metrics = {
            'rollout_rewards': [],
        }

        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__
            )

        print(f"ManiFlow NFT Trainer initialized")
        print(f"  - Device: {self.device}")
        print(f"  - Policy params: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"  - Old model params (frozen): {sum(p.numel() for p in self.old_model.parameters()):,}")
        print(f"  - Ref model params (frozen): {sum(p.numel() for p in self.ref_model.parameters()):,}")

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer with separate learning rates for actor and critic."""
        params_actor = []
        params_critic = []

        for name, param in self.policy.named_parameters():
            if param.requires_grad:
                if "value_head" in name or "value_mlp" in name or "attention_pool" in name:
                    params_critic.append(param)
                else:
                    params_actor.append(param)

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

        optimizer = optim.AdamW(param_groups, eps=1e-5, weight_decay=0.01)

        actor_params = sum(p.numel() for p in params_actor)
        critic_params = sum(p.numel() for p in params_critic)
        print(f"  - Actor params: {actor_params:,} (lr={self.config.learning_rate})")
        print(f"  - Critic params: {critic_params:,} (lr={self.config.value_lr})")

        return optimizer

    def train(self) -> None:
        """Main training loop."""
        print(f"Starting NFT training for {self.config.total_timesteps:,} timesteps")
        if self.config.critic_warmup_rollouts > 0:
            print(f"Critic warmup: {self.config.critic_warmup_rollouts} rollouts")

        start_time = time.time()

        while self.global_step < self.config.total_timesteps:
            rollout_start_time = time.time()

            # Stage 1: Collect rollouts
            print(f"\nRollout {self.rollout_count + 1} (Step {self.global_step:,}/{self.config.total_timesteps:,})")
            rollout_batch, video_paths = self.rollout_collector.collect_rollouts(
                num_episodes=self.config.train_n_episodes,
                num_envs=self.config.num_envs
            )

            actual_samples = rollout_batch.n_chunk_steps * rollout_batch.batch_size

            # Stage 2: Calculate advantages and returns (GAE, unchanged)
            rollout_batch = self.advantage_calculator.calculate_advantages_and_returns(rollout_batch)

            # Stage 3: NFT training (critic-only during warmup)
            critic_only = self.rollout_count < self.config.critic_warmup_rollouts
            training_stats = self.run_nft_training(rollout_batch, critic_only=critic_only)

            # Stage 4: Update old_model with EMA decay
            if not critic_only:
                self._update_old_model()

            # Update global step
            self.global_step += actual_samples
            self.rollout_count += 1

            # Metrics and logging
            rollout_time = time.time() - rollout_start_time
            self._log_training_metrics(training_stats, rollout_batch, rollout_time, video_paths)

            if self.rollout_count % self.config.eval_interval == 0:
                self._run_evaluation()
            if self.rollout_count % self.config.save_interval == 0:
                self._save_checkpoint()

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s ({total_time/3600:.2f}h)")
        self._save_checkpoint(final=True)

        if self.use_wandb:
            wandb.finish()

    def run_nft_training(self, rollout_batch: ManiFlowRolloutBatch, critic_only: bool = False) -> Dict[str, float]:
        """
        Run NFT training on rollout batch.

        Args:
            rollout_batch: Collected rollout data with advantages and returns
            critic_only: If True, only train critic (value head)

        Returns:
            Training statistics
        """
        if critic_only:
            n_epochs = self.config.critic_warmup_epochs
            print(f"Running critic warmup ({n_epochs} epochs)...")
        else:
            n_epochs = 1
            print("Running NFT training (1 epoch)...")

        # Keep data on CPU, move minibatches to GPU
        torch_batch = rollout_batch.to_torch(torch.device('cpu'))
        flat_data = self._flatten_batch_data(torch_batch)
        total_samples = flat_data['advantages'].shape[0]

        accumulate_steps = self.config.gradient_accumulate_every
        print(f"  Training samples: {total_samples}, batch: {self.config.batch_size}, "
              f"accumulate: {accumulate_steps}")

        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'kl_loss': 0.0,
            'total_loss': 0.0,
        }

        num_updates = 0
        num_minibatches = 0

        self.policy.train()
        self.optimizer.zero_grad()
        accumulated_count = 0

        for epoch in range(n_epochs):
            indices = torch.randperm(total_samples)

            for start in range(0, total_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, total_samples)
                batch_indices = indices[start:end]
                actual_batch_size = batch_indices.shape[0]

                mini_batch = self._extract_minibatch_to_device(flat_data, batch_indices)

                # Compute NFT loss
                loss_dict = self._compute_nft_loss(mini_batch, critic_only=critic_only)

                # Scale loss for gradient accumulation
                weight = actual_batch_size / total_samples
                scaled_loss = loss_dict['total_loss'] * weight
                scaled_loss.backward()

                accumulated_count += 1
                num_minibatches += 1

                for key in stats.keys():
                    if key in loss_dict:
                        stats[key] += loss_dict[key].item()

                if accumulated_count >= accumulate_steps:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    num_updates += 1
                    accumulated_count = 0

            if critic_only and n_epochs > 1:
                epoch_value_loss = stats['value_loss'] / num_minibatches if num_minibatches > 0 else 0
                print(f"    Epoch {epoch + 1}/{n_epochs}: value_loss={epoch_value_loss:.6f}")

        # Handle remaining gradients
        if accumulated_count > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            num_updates += 1

        # Average statistics
        for key in stats.keys():
            stats[key] /= max(num_minibatches, 1)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.policy.set_global_step(self.rollout_count)
        self.policy.eval()

        print(f"  NFT training completed: {num_updates} optimizer steps, {num_minibatches} minibatches")
        print(f"    - Policy loss: {stats['policy_loss']:.6f}")
        print(f"    - Value loss: {stats['value_loss']:.6f}")
        print(f"    - KL loss: {stats['kl_loss']:.6f}")

        return stats

    def _compute_nft_loss(self,
                          mini_batch: Dict[str, torch.Tensor],
                          critic_only: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute NFT loss.

        Core NFT logic:
        - Three forward passes: v_old (frozen old policy), v_fwd (current, has grad), v_ref (frozen pretrained)
        - Positive prediction: beta*v_fwd + (1-beta)*v_old  (push toward x0 for good samples)
        - Negative prediction: (1+beta)*v_old - beta*v_fwd  (push away from x0 for bad samples)
        - Advantage r in [0,1] blends positive and negative losses
        - Adaptive weight converts L2 to L1-like behavior
        """
        obs_dict = mini_batch['observation']
        returns = mini_batch['returns']           # [B, 1]
        advantages = mini_batch['advantages']      # [B, action_chunk]
        old_values = mini_batch['prev_values']     # [B, 1]
        loss_mask = mini_batch.get('loss_mask')    # [B, action_chunk]
        x0 = mini_batch['actions_clean']           # [B, horizon, action_dim]

        B = x0.shape[0]
        device = x0.device

        # 1. Encode observations (shared encoder, with grad for actor)
        obs_features = self.policy.encode_observations(obs_dict)

        # 2. Compute value (always, for critic training)
        values = self.policy.compute_value(obs_features)  # [B]
        values_expanded = values.unsqueeze(1)  # [B, 1]

        # Value loss with clipping (same as PPO)
        value_loss_unclipped = (values_expanded - returns) ** 2
        values_clipped = old_values + torch.clamp(
            values_expanded - old_values,
            -self.config.clip_range,
            self.config.clip_range
        )
        value_loss_clipped = (values_clipped - returns) ** 2
        value_loss_unmasked = torch.max(value_loss_unclipped, value_loss_clipped)

        if loss_mask is not None:
            value_mask = loss_mask[:, 0:1].bool()
            value_loss = masked_mean(value_loss_unmasked, value_mask)
        else:
            value_loss = value_loss_unmasked.mean()

        if critic_only:
            total_loss = self.config.value_coef * value_loss
            return {
                'policy_loss': torch.tensor(0.0, device=device),
                'value_loss': value_loss,
                'kl_loss': torch.tensor(0.0, device=device),
                'total_loss': total_loss,
            }

        # 3. NFT actor loss
        nft_loss_accum = torch.zeros(B, device=device)
        kl_loss_accum = torch.zeros(B, device=device)

        # Discrete timestep schedule (same as inference, excludes t=0)
        N = self.policy.num_inference_steps
        timestep_schedule = torch.linspace(1, 0, N + 1, device=device)[:-1]  # [N]

        beta = self.config.beta
        adv_clip_max = self.config.adv_clip_max

        # Normalize advantage to r in [0, 1]
        adv = advantages.mean(dim=-1)  # [B]
        adv_clipped = adv.clamp(-adv_clip_max, adv_clip_max)
        r = (adv_clipped / adv_clip_max) / 2.0 + 0.5  # [B]
        r = r.clamp(0, 1)

        for j in range(N):
            # Iterate over all discrete timesteps (same as DiffusionNFT)
            t = timestep_schedule[j].expand(B)  # [B]
            t_exp = t.view(B, 1, 1)  # [B, 1, 1]

            # Synthesize noisy input: xt = (1-t)*x0 + t*noise
            noise = torch.randn_like(x0)
            xt = (1 - t_exp) * x0 + t_exp * noise

            # Detach obs_features for frozen models to avoid unnecessary graph
            obs_features_detached = obs_features.detach()

            # v_old: frozen old policy
            with torch.no_grad():
                v_old = self.old_model(
                    sample=xt, timestep=t,
                    target_t=torch.zeros_like(t),
                    vis_cond=obs_features_detached, lang_cond=None
                )

            # v_fwd: current policy (has gradient)
            v_fwd = self.policy.predict_v(obs_features, xt, t)

            # v_ref: frozen pretrained base
            with torch.no_grad():
                v_ref = self.ref_model(
                    sample=xt, timestep=t,
                    target_t=torch.zeros_like(t),
                    vis_cond=obs_features_detached, lang_cond=None
                )

            # Positive/negative predictions
            pos_pred = beta * v_fwd + (1 - beta) * v_old
            neg_pred = (1 + beta) * v_old - beta * v_fwd

            # x0 predictions
            x0_pos = xt - t_exp * pos_pred
            x0_neg = xt - t_exp * neg_pred

            # Adaptive weight (detached, converts L2 to L1-like)
            with torch.no_grad():
                w_pos = (x0_pos - x0).abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-5)
                w_neg = (x0_neg - x0).abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-5)

            # Weighted L2 losses -> [B]
            pos_loss = ((x0_pos - x0) ** 2 / w_pos).mean(dim=(1, 2))
            neg_loss = ((x0_neg - x0) ** 2 / w_neg).mean(dim=(1, 2))

            # NFT combined loss: r * pos_loss / beta + (1 - r) * neg_loss / beta
            nft_loss = r * pos_loss / beta + (1 - r) * neg_loss / beta
            nft_loss_accum += nft_loss * adv_clip_max  # scale back (from DiffusionNFT)

            # KL regularization: ||v_fwd - v_ref||^2
            kl_loss = ((v_fwd - v_ref) ** 2).mean(dim=(1, 2))  # [B]
            kl_loss_accum += kl_loss

        # Average over timesteps
        nft_loss_accum /= N
        kl_loss_accum /= N

        # Apply loss mask
        if loss_mask is not None:
            policy_mask = loss_mask[:, 0].bool()  # [B]
            policy_loss = masked_mean(nft_loss_accum, policy_mask)
            kl_loss_final = masked_mean(kl_loss_accum, policy_mask)
        else:
            policy_loss = nft_loss_accum.mean()
            kl_loss_final = kl_loss_accum.mean()

        total_loss = policy_loss + self.config.beta_kl * kl_loss_final + self.config.value_coef * value_loss

        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'kl_loss': kl_loss_final,
            'total_loss': total_loss,
        }

    def _update_old_model(self):
        """EMA update of old_model toward current policy.model."""
        decay = min(self.rollout_count * self.config.old_model_decay_rate,
                    self.config.old_model_decay)
        with torch.no_grad():
            for old_p, new_p in zip(self.old_model.parameters(),
                                    self.policy.model.parameters()):
                old_p.data.copy_(decay * old_p.data + (1 - decay) * new_p.data)
        print(f"  Old model updated (decay={decay:.4f})")

    def _flatten_batch_data(self, torch_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Flatten rollout batch for minibatch training. Data stays on CPU."""
        flat_data = {}

        obs_dict = {}
        for key, value in torch_batch['observation'].items():
            obs_dict[key] = value.flatten(0, 1)
        flat_data['observation'] = obs_dict

        for key in ['chains', 'denoise_inds', 'prev_logprobs', 'prev_values',
                    'advantages', 'returns', 'actions_clean',
                    'loss_mask', 'x_stds', 'x_means']:
            if key in torch_batch and torch_batch[key] is not None:
                flat_data[key] = torch_batch[key].flatten(0, 1)

        return flat_data

    def _extract_minibatch_to_device(self, flat_data: Dict[str, torch.Tensor],
                                      batch_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract minibatch from CPU data and move to GPU."""
        mini_batch = {}

        for key, value in flat_data.items():
            if key == 'observation':
                mini_batch[key] = {
                    k: v[batch_indices].to(self.device, non_blocking=True)
                    for k, v in value.items()
                }
            else:
                mini_batch[key] = value[batch_indices].to(self.device, non_blocking=True)

        return mini_batch

    def _log_training_metrics(self,
                            training_stats: Dict[str, float],
                            rollout_batch: ManiFlowRolloutBatch,
                            rollout_time: float,
                            video_paths: List[str] = None) -> None:
        """Log training metrics."""
        mean_reward = rollout_batch.rewards.mean()
        total_reward = rollout_batch.rewards.sum()

        for key, value in training_stats.items():
            if key in self.training_metrics:
                self.training_metrics[key].append(value)

        self.rollout_metrics['rollout_rewards'].append(float(total_reward))

        actor_lr = self.optimizer.param_groups[0]['lr']
        critic_lr = self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) > 1 else actor_lr

        actual_samples = rollout_batch.n_chunk_steps * rollout_batch.batch_size
        fps = actual_samples / rollout_time

        if self.rollout_count % self.config.log_interval == 0:
            print(f"Metrics (Rollout {self.rollout_count}):")
            print(f"  - Mean reward: {mean_reward:.3f}")
            print(f"  - Policy loss: {training_stats['policy_loss']:.6f}")
            print(f"  - Value loss: {training_stats['value_loss']:.6f}")
            print(f"  - KL loss: {training_stats['kl_loss']:.6f}")
            print(f"  - Actor LR: {actor_lr:.2e}, Critic LR: {critic_lr:.2e}")
            print(f"  - FPS: {fps:.1f}")

        if self.use_wandb:
            log_dict = {
                'rollout/mean_reward': mean_reward,
                'rollout/total_reward': total_reward,
                'train/policy_loss': training_stats['policy_loss'],
                'train/value_loss': training_stats['value_loss'],
                'train/kl_loss': training_stats['kl_loss'],
                'train/total_loss': training_stats['total_loss'],
                'train/actor_lr': actor_lr,
                'train/critic_lr': critic_lr,
                'train/global_step': self.global_step,
                'train/rollout_count': self.rollout_count,
                'train/old_model_decay': min(self.rollout_count * self.config.old_model_decay_rate,
                                              self.config.old_model_decay),
                'perf/fps': fps,
                'perf/rollout_time': rollout_time,
            }

            if (video_paths is not None and
                self.rollout_count % self.config.rollout_video_interval == 0):
                n_videos = min(self.config.n_rollout_videos, len(video_paths))
                for i in range(n_videos):
                    if video_paths[i] is not None:
                        log_dict[f'rollout/video_{i}'] = wandb.Video(video_paths[i])

            wandb.log(log_dict, step=self.global_step)

    def _save_checkpoint(self, final: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'old_model_state_dict': self.old_model.state_dict(),
            'ref_model_state_dict': self.ref_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'rollout_count': self.rollout_count,
            'config': self.config.__dict__,
            'training_metrics': self.training_metrics,
            'rollout_metrics': self.rollout_metrics,
        }

        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        checkpoint_name = "final_checkpoint.pt" if final else f"checkpoint_{self.rollout_count}.pt"
        checkpoint_path = Path(self.config.save_path) / checkpoint_name
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def _run_evaluation(self) -> None:
        """Run evaluation episodes."""
        print(f"Running evaluation ({self.config.eval_n_episodes} episodes)...")
        self.policy.eval()

        eval_metrics = {}
        with torch.no_grad():
            eval_runner_log = self.env_runner.run(
                self.policy,
                eval_mode=True,
                n_episodes=self.config.eval_n_episodes
            )

            for key, value in eval_runner_log.items():
                eval_metrics[f'eval/{key}'] = value

            print(f"Evaluation Results:")
            for key, value in eval_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  - {key}: {value:.3f}")

            if self.use_wandb:
                eval_metrics['eval/rollout_count'] = self.rollout_count
                eval_metrics['eval/global_step'] = self.global_step
                wandb.log(eval_metrics, step=self.global_step)

        self.policy.train()


def create_maniflow_nft_trainer(
    policy: ManiFlowRLPointcloudPolicy,
    env_runner,
    config: Optional[NFTConfig] = None,
    advantage_config: Optional[AdvantageConfig] = None,
    device: str = "cuda",
    use_wandb: bool = True
) -> ManiFlowNFTTrainer:
    """Factory function to create a ManiFlowNFTTrainer."""
    if config is None:
        config = NFTConfig()
    if advantage_config is None:
        advantage_config = AdvantageConfig()

    return ManiFlowNFTTrainer(
        policy=policy,
        env_runner=env_runner,
        config=config,
        advantage_config=advantage_config,
        device=device,
        use_wandb=use_wandb
    )
