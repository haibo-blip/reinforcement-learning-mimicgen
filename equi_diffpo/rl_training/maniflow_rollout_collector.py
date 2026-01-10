#!/usr/bin/env python3
"""
ManiFlow RL Rollout Collector
Collects rollout data for PPO training following RLinf pattern.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import copy

from equi_diffpo.policy.maniflow.maniflow_pointcloud_rl_policy import ManiFlowRLPointcloudPolicy
from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.env_runner.robomimic_image_runner import RobomimicImageRunner
from equi_diffpo.rl_training.rl_utils import compute_loss_mask


@dataclass
class ManiFlowRolloutStep:
    """Single step in rollout following RLinf EmbodiedRolloutResult pattern."""

    # Environment data
    observations: Dict[str, np.ndarray]
    actions: np.ndarray  # [action_chunk, action_dim] - sent to env
    rewards: np.ndarray  # [action_chunk] - per action chunk
    dones: np.ndarray    # [1] - episode termination
    truncations: np.ndarray  # [1] - episode truncation

    # Policy outputs (from sample_actions)
    prev_logprobs: np.ndarray   # [action_chunk, action_dim] - log π_old(a|s)
    prev_values: np.ndarray     # [1] - V_old(s)

    # Forward inputs for training (stored chains and denoise_inds)
    forward_inputs: Dict[str, Any]  # Contains chains, denoise_inds, obs data


@dataclass
class ManiFlowRolloutBatch:
    """Batch of rollout data following RLinf pattern."""

    # Environment data [n_chunk_steps, batch_size, ...]
    observations: Dict[str, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    truncations: np.ndarray

    # Policy data [n_chunk_steps, batch_size, ...]
    prev_logprobs: np.ndarray
    prev_values: np.ndarray

    # Loss masking [n_chunk_steps, batch_size, action_chunk]
    loss_mask: Optional[np.ndarray] = None
    loss_mask_sum: Optional[np.ndarray] = None

    # Forward inputs [n_chunk_steps, batch_size, ...]
    chains: np.ndarray
    denoise_inds: np.ndarray

    # Computed during advantage calculation
    advantages: Optional[np.ndarray] = None  # [n_chunk_steps, batch_size, action_chunk]
    returns: Optional[np.ndarray] = None     # [n_chunk_steps, batch_size, 1]

    @property
    def batch_size(self) -> int:
        return self.rewards.shape[1]

    @property
    def n_chunk_steps(self) -> int:
        return self.rewards.shape[0]

    def to_torch(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Convert to torch tensors for training."""
        result = {}

        # Convert observations
        obs_dict = {}
        for key, value in self.observations.items():
            obs_dict[key] = torch.from_numpy(value).to(device)
        result['observation'] = obs_dict

        # Convert other arrays
        for field_name in ['actions', 'rewards', 'dones', 'prev_logprobs', 'prev_values',
                          'chains', 'denoise_inds', 'loss_mask', 'loss_mask_sum']:
            if hasattr(self, field_name) and getattr(self, field_name) is not None:
                result[field_name] = torch.from_numpy(getattr(self, field_name)).to(device)

        # Convert advantages and returns if computed
        if self.advantages is not None:
            result['advantages'] = torch.from_numpy(self.advantages).to(device)
        if self.returns is not None:
            result['returns'] = torch.from_numpy(self.returns).to(device)

        return result


class ManiFlowRolloutCollector:
    """
    Rollout collector for ManiFlow RL policy following RLinf pattern.
    """

    def __init__(self,
                 policy: ManiFlowRLPointcloudPolicy,
                 env_runner: RobomimicImageRunner,  # Use actual RobomimicImageRunner
                 max_steps_per_episode: int = 1000,
                 action_chunk_size: int = 8,
                 obs_chunk_size: int = 2,
                 device: str = "cuda"):

        self.policy = policy
        self.env_runner = env_runner
        self.max_steps_per_episode = max_steps_per_episode
        self.action_chunk_size = action_chunk_size
        self.obs_chunk_size = obs_chunk_size
        self.device = torch.device(device)

        # Set policy to eval mode for rollouts
        self.policy.eval()

        print(f"🎲 ManiFlow Rollout Collector initialized")
        print(f"  - Action chunk size: {action_chunk_size}")
        print(f"  - Obs chunk size: {obs_chunk_size}")
        print(f"  - Max steps per episode: {max_steps_per_episode}")

    def predict_action_batch(self, env_obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict actions for a batch of observations following RLinf pattern.

        Args:
            env_obs: Environment observations dict

        Returns:
            actions: Actions to send to environment [B, action_chunk, action_dim]
            result: Dict with prev_logprobs, prev_values, forward_inputs
        """
        with torch.no_grad():
            # Convert to torch and move to device
            obs_dict = {}
            for key, value in env_obs.items():
                if isinstance(value, np.ndarray):
                    obs_dict[key] = torch.from_numpy(value).to(self.device)
                else:
                    obs_dict[key] = value

            # Sample actions with chains (like RLinf sample_actions)
            sample_result = self.policy.sample_actions(
                obs_dict,
                mode="eval",  # Use eval mode for rollouts
                compute_values=True
            )

            # Extract actions for environment
            actions = sample_result['actions'].cpu().numpy()  # [B, action_chunk, action_dim]

            # Prepare result following RLinf pattern
            result = {
                'prev_logprobs': sample_result['prev_logprobs'].cpu().numpy(),  # [B, action_chunk, action_dim]
                'prev_values': sample_result['prev_values'].cpu().numpy(),      # [B, 1]
                'forward_inputs': {
                    'chains': sample_result['chains'].cpu().numpy(),            # [B, N+1, horizon, action_dim]
                    'denoise_inds': sample_result['denoise_inds'].cpu().numpy(), # [B, N]
                    # Include observation data for training
                    **{key: value.cpu().numpy() if torch.is_tensor(value) else value
                       for key, value in obs_dict.items()}
                }
            }

            return actions, result

    def collect_rollouts_from_runner_results(self, runner_results: Dict) -> ManiFlowRolloutBatch:
        """
        Convert RobomimicRLRunner results to ManiFlowRolloutBatch format.

        Args:
            runner_results: Results from RobomimicRLRunner.run_rl(policy)
                          Contains 'rl_data' with step-by-step information

        Returns:
            ManiFlowRolloutBatch: Collected rollout data in RL training format
        """
        print(f"🎲 Converting runner results to rollout batch format...")

        # Check if we have RL data
        if 'rl_data' not in runner_results:
            raise ValueError("Runner results missing 'rl_data' - use RobomimicRLRunner.run_rl()")

        rl_data = runner_results['rl_data']

        # Extract step-by-step data (already in correct format from RobomimicRLRunner)
        observations = rl_data['observations']        # Dict[str, np.ndarray] [n_steps, batch_size, ...]
        actions = rl_data['actions']                  # [n_steps, batch_size, action_chunk, action_dim]
        rewards = rl_data['rewards']                  # [n_steps, batch_size, action_chunk]
        dones = rl_data['dones']                      # [n_steps, batch_size, 1]
        prev_logprobs = rl_data['prev_logprobs']      # [n_steps, batch_size, action_chunk, action_dim]
        prev_values = rl_data['prev_values']          # [n_steps, batch_size, 1]
        chains = rl_data['chains']                    # [n_steps, batch_size, N+1, horizon, action_dim]
        denoise_inds = rl_data['denoise_inds']        # [n_steps, batch_size, N]

        n_steps = rl_data['total_steps']
        batch_size = rl_data['total_envs']

        print(f"✅ Converted runner results to rollout batch:")
        print(f"  - Steps: {n_steps}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Actions shape: {actions.shape}")
        print(f"  - Chains shape: {chains.shape}")
        print(f"  - Denoise inds shape: {denoise_inds.shape}")

        # Create truncations (not tracked by default)
        truncations = np.zeros_like(dones)

        # Compute loss mask following RLinf pattern
        # Convert to torch tensors for computation, then back to numpy
        dones_torch = torch.from_numpy(dones)

        # Add bootstrap step for loss mask computation (all zeros for last step)
        # Shape: [n_steps+1, batch_size, action_chunk]
        action_chunk_size = rewards.shape[2] if len(rewards.shape) > 2 else 1
        bootstrap_done = torch.zeros(1, batch_size, action_chunk_size, dtype=dones_torch.dtype)
        dones_with_bootstrap = torch.cat([dones_torch, bootstrap_done], dim=0)

        # Compute loss mask
        loss_mask, loss_mask_sum = compute_loss_mask(dones_with_bootstrap)

        # Convert back to numpy
        loss_mask_np = loss_mask.numpy()
        loss_mask_sum_np = loss_mask_sum.numpy()

        print(f"📊 Loss mask computed: {loss_mask_np.sum()} / {loss_mask_np.size} valid steps")

        # Validate shapes
        expected_shapes = {
            'actions': (n_steps, batch_size, actions.shape[2], actions.shape[3]),
            'rewards': (n_steps, batch_size, rewards.shape[2]),
            'dones': (n_steps, batch_size, 1),
            'prev_logprobs': (n_steps, batch_size, prev_logprobs.shape[2], prev_logprobs.shape[3]),
            'prev_values': (n_steps, batch_size, 1),
            'chains': (n_steps, batch_size, chains.shape[2], chains.shape[3], chains.shape[4]),
            'denoise_inds': (n_steps, batch_size, denoise_inds.shape[2]),
        }

        for key, expected_shape in expected_shapes.items():
            actual_shape = rl_data[key].shape
            if actual_shape != expected_shape:
                print(f"⚠️  Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}")

        return ManiFlowRolloutBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            truncations=truncations,
            prev_logprobs=prev_logprobs,
            prev_values=prev_values,
            loss_mask=loss_mask_np,
            loss_mask_sum=loss_mask_sum_np,
            chains=chains,
            denoise_inds=denoise_inds
        )

    def collect_rollouts_with_runner(self) -> ManiFlowRolloutBatch:
        """
        Use RobomimicRLRunner to collect rollouts with RL data.

        Returns:
            ManiFlowRolloutBatch: Collected rollout data
        """
        print(f"🎲 Using RobomimicRLRunner for rollout collection...")

        # Check if we have the RL-compatible runner
        if hasattr(self.env_runner, 'run_rl'):
            # Use RL-specific method
            runner_results = self.env_runner.run_rl(self.policy)
        else:
            # Fall back to regular run method
            print("⚠️  Using regular runner - RL data may not be available")
            runner_results = self.env_runner.run(self.policy)

        # Convert results to our batch format
        return self.collect_rollouts_from_runner_results(runner_results)

    def _convert_to_batch(self, steps: List[ManiFlowRolloutStep]) -> ManiFlowRolloutBatch:
        """Convert list of steps to batch format following RLinf pattern."""
        if not steps:
            raise ValueError("No steps to convert")

        n_steps = len(steps)
        batch_size = steps[0].actions.shape[0]
        action_chunk = steps[0].actions.shape[1]
        action_dim = steps[0].actions.shape[2]

        # Stack environment data [n_chunk_steps, batch_size, ...]
        observations = {}
        for key in steps[0].observations.keys():
            observations[key] = np.stack([step.observations[key] for step in steps], axis=0)

        actions = np.stack([step.actions for step in steps], axis=0)
        rewards = np.stack([step.rewards for step in steps], axis=0)
        dones = np.stack([step.dones for step in steps], axis=0)
        truncations = np.stack([step.truncations for step in steps], axis=0)

        # Stack policy data [n_chunk_steps, batch_size, ...]
        prev_logprobs = np.stack([step.prev_logprobs for step in steps], axis=0)
        prev_values = np.stack([step.prev_values for step in steps], axis=0)

        # Stack forward inputs [n_chunk_steps, batch_size, ...]
        chains = np.stack([step.forward_inputs['chains'] for step in steps], axis=0)
        denoise_inds = np.stack([step.forward_inputs['denoise_inds'] for step in steps], axis=0)

        print(f"📊 Batch shapes:")
        print(f"  - observations: {observations[list(observations.keys())[0]].shape}")
        print(f"  - actions: {actions.shape}")
        print(f"  - rewards: {rewards.shape}")
        print(f"  - prev_logprobs: {prev_logprobs.shape}")
        print(f"  - chains: {chains.shape}")
        print(f"  - denoise_inds: {denoise_inds.shape}")

        return ManiFlowRolloutBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            truncations=truncations,
            prev_logprobs=prev_logprobs,
            prev_values=prev_values,
            chains=chains,
            denoise_inds=denoise_inds
        )


