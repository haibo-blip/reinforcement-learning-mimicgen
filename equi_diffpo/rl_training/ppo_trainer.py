import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional
import wandb
from tqdm import tqdm

from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.common.pytorch_util import dict_apply


class SimpleValueNetwork(nn.Module):
    """Simple MLP value network for critic."""

    def __init__(self, obs_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        layers = []
        input_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of value network.

        Args:
            obs_dict: Dictionary of observations

        Returns:
            values: Value estimates [batch_size, 1]
        """
        # Flatten all observations
        obs_list = []
        for key, value in obs_dict.items():
            if key == 'point_cloud':
                # Flatten point cloud: [batch_size, n_points, features] -> [batch_size, n_points * features]
                value = value.flatten(start_dim=1)
            elif len(value.shape) > 2:
                # Flatten other high-dimensional observations
                value = value.flatten(start_dim=1)
            obs_list.append(value)

        # Concatenate all observations
        obs = torch.cat(obs_list, dim=-1)
        return self.network(obs)


class PPOTrainer:
    """PPO trainer for ManiFlow policies with GAE advantage computation."""

    def __init__(
        self,
        policy,
        critic: Optional[nn.Module] = None,
        policy_optimizer: Optional[torch.optim.Optimizer] = None,
        critic_optimizer: Optional[torch.optim.Optimizer] = None,
        normalizer: Optional[LinearNormalizer] = None,
        device: torch.device = torch.device("cuda"),
        # PPO hyperparameters
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        # GAE parameters
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        # Training parameters
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        policy_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        # Flow SDE parameters
        noise_level: float = 0.5,
        num_inference_steps: int = 10,
    ):
        self.policy = policy
        self.normalizer = normalizer
        self.device = device

        # Create critic if not provided
        if critic is None:
            # Estimate observation dimension
            obs_dim = self._estimate_obs_dim()
            self.critic = SimpleValueNetwork(obs_dim).to(device)
        else:
            self.critic = critic.to(device)

        # Create optimizers if not provided
        if policy_optimizer is None:
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        else:
            self.policy_optimizer = policy_optimizer

        if critic_optimizer is None:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        else:
            self.critic_optimizer = critic_optimizer

        # PPO hyperparameters
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # GAE parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Training parameters
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        # Flow SDE parameters
        self.noise_level = noise_level
        self.num_inference_steps = num_inference_steps

        # Metrics tracking
        self.train_step = 0

    def _estimate_obs_dim(self) -> int:
        """Estimate total observation dimension for value network."""
        # This is a rough estimate - adjust based on your observation space
        # Point cloud: 1024 * 6 = 6144
        # Low-dim obs: typically ~10-20 dimensions
        return 6144 + 20

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    dones: torch.Tensor, episode_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Rewards tensor [batch_size, max_episode_length]
            values: Value estimates [batch_size, max_episode_length]
            dones: Done flags [batch_size, max_episode_length]
            episode_lengths: Actual length of each episode [batch_size]

        Returns:
            advantages: GAE advantages [batch_size, max_episode_length]
            returns: Value targets [batch_size, max_episode_length]
        """
        batch_size, max_length = rewards.shape
        advantages = torch.zeros_like(rewards)

        for batch_idx in range(batch_size):
            ep_length = int(episode_lengths[batch_idx].item())

            # Compute advantages for this episode
            gae = 0
            for t in reversed(range(ep_length)):
                if t == ep_length - 1:
                    # Last step: no next value, treat as terminal
                    delta = rewards[batch_idx, t] - values[batch_idx, t]
                else:
                    # Use next step value
                    next_non_terminal = 1.0 - dones[batch_idx, t].float()
                    delta = rewards[batch_idx, t] + self.gamma * values[batch_idx, t + 1] * next_non_terminal - values[batch_idx, t]

                next_non_terminal = 1.0 - dones[batch_idx, t].float()
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                advantages[batch_idx, t] = gae

        # Compute returns as advantages + values
        returns = advantages + values
        return advantages, returns

    def sample_actions_with_fixed_noise(self, obs_dict: Dict[str, torch.Tensor],
                                       fixed_noise: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Sample actions using ManiFlow policy with fixed noise for flow SDE.
        This follows the approach from RLinf OpenPI pi0.5.

        Args:
            obs_dict: Observation dictionary
            fixed_noise: Fixed noise for deterministic sampling in training

        Returns:
            action_dict: Dictionary containing actions, chains, and flow info
        """
        batch_size = list(obs_dict.values())[0].shape[0]
        device = list(obs_dict.values())[0].device

        # Generate fixed noise if not provided
        if fixed_noise is None:
            action_shape = (batch_size, self.policy.horizon, self.policy.n_action_steps)
            fixed_noise = torch.randn(action_shape, device=device)

        # Use ManiFlow policy's prediction with fixed noise
        # This will need to be adapted based on your specific ManiFlow implementation
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict, noise=fixed_noise)

        # Store the fixed noise and flow trajectory for log probability computation
        action_dict['fixed_noise'] = fixed_noise

        return action_dict

    def compute_flow_sde_log_prob(self, obs_dict: Dict[str, torch.Tensor],
                                actions: torch.Tensor, fixed_noise: torch.Tensor) -> torch.Tensor:
        """
        Compute log probabilities for actions sampled via flow SDE.
        Based on RLinf OpenPI flow_sde implementation.

        Args:
            obs_dict: Observation dictionary
            actions: Final actions from flow SDE
            fixed_noise: The fixed noise used during sampling

        Returns:
            log_probs: Log probabilities for each action
        """
        batch_size = actions.shape[0]
        device = actions.device

        # Create timesteps following RLinf pattern
        timesteps = torch.linspace(1, 1 / self.num_inference_steps, self.num_inference_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])

        # Sample a random timestep for each batch element (following RLinf training pattern)
        sampled_indices = torch.randint(0, self.num_inference_steps, (batch_size,), device=device)

        # Get the timestep and delta for each sample
        t_input = timesteps[sampled_indices]
        delta = timesteps[sampled_indices] - timesteps[sampled_indices + 1]

        # Compute flow SDE weights following RLinf implementation
        # sigmas = noise_level * sqrt(t / (1 - t))
        sigmas = self.noise_level * torch.sqrt(
            timesteps / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
        )[:-1]

        sigma_i = sigmas[sampled_indices]

        # Flow SDE weights (from RLinf sample_mean_var_val)
        x_t_std = torch.sqrt(delta) * sigma_i

        # Approximate the velocity prediction using the policy
        # This is simplified - in practice you'd need to reconstruct the full flow
        with torch.no_grad():
            # Sample intermediate points along the flow trajectory
            t_expanded = t_input[:, None, None].expand_as(actions)
            intermediate_actions = fixed_noise * t_expanded + actions * (1 - t_expanded)

            # Get velocity prediction (approximate)
            velocity_dict = self.policy.predict_action(obs_dict)
            velocity = velocity_dict['action']

        # Compute log probability assuming Gaussian distribution
        # log p(action | obs) = log N(action; mean, std)
        std_expanded = x_t_std[:, None, None].expand_as(actions)
        log_probs = -0.5 * ((actions - (actions - velocity * t_expanded)) / (std_expanded + 1e-8)).pow(2).sum(dim=[1, 2])
        log_probs = log_probs - 0.5 * np.log(2 * np.pi) * actions.numel() // batch_size
        log_probs = log_probs - torch.log(std_expanded + 1e-8).sum(dim=[1, 2])

        return log_probs

    def compute_policy_loss(self, obs_dict: Dict[str, torch.Tensor], actions: torch.Tensor,
                           old_log_probs: torch.Tensor, advantages: torch.Tensor,
                           fixed_noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PPO policy loss with flow SDE log probabilities.

        Args:
            obs_dict: Observation dictionary
            actions: Actions tensor
            old_log_probs: Old log probabilities
            advantages: GAE advantages
            fixed_noise: Fixed noise used during action sampling

        Returns:
            policy_loss: PPO clipped policy loss
            info_dict: Training metrics
        """
        # Compute new log probabilities using flow SDE
        if fixed_noise is not None:
            log_probs = self.compute_flow_sde_log_prob(obs_dict, actions, fixed_noise)
        else:
            # Fallback to simple approximation
            action_dict = self.policy.predict_action(obs_dict)
            predicted_actions = action_dict['action']

            # Simple Gaussian approximation
            action_std = 0.1
            log_probs = -0.5 * ((actions - predicted_actions) / action_std).pow(2).sum(dim=-1)
            log_probs = log_probs - 0.5 * np.log(2 * np.pi) * actions.shape[-1] - np.log(action_std) * actions.shape[-1]

        # Compute ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy computation (simplified for flow models)
        entropy = torch.tensor(0.0, device=advantages.device)

        # Total policy loss
        total_policy_loss = policy_loss

        # Compute metrics
        clip_fraction = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()
        approx_kl = (old_log_probs - log_probs).mean()

        info_dict = {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'clip_fraction': clip_fraction.item(),
            'approx_kl': approx_kl.item(),
        }

        return total_policy_loss, info_dict

    def compute_value_loss(self, obs_dict: Dict[str, torch.Tensor],
                          returns: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute value function loss.

        Args:
            obs_dict: Observation dictionary
            returns: Value targets

        Returns:
            value_loss: MSE value loss
            info_dict: Training metrics
        """
        # Get value predictions
        values = self.critic(obs_dict).squeeze(-1)

        # Compute MSE loss
        value_loss = F.mse_loss(values, returns)

        info_dict = {
            'value_loss': value_loss.item(),
            'value_mean': values.mean().item(),
            'returns_mean': returns.mean().item(),
        }

        return value_loss, info_dict

    def update_policy_and_critic(self, batch_data: Dict[str, torch.Tensor]) -> Dict:
        """
        Update policy and critic using PPO with collected batch data.

        Args:
            batch_data: Dictionary containing rollout data

        Returns:
            metrics: Training metrics dictionary
        """
        # Extract data from batch
        obs_dict = batch_data['obs']
        actions = batch_data['actions']
        fixed_noise = batch_data.get('fixed_noise', None)
        rewards = batch_data['rewards']
        dones = batch_data['dones']
        episode_lengths = batch_data['episode_lengths']

        # Move to device
        obs_dict = dict_apply(obs_dict, lambda x: x.to(self.device))
        actions = actions.to(self.device)
        if fixed_noise is not None:
            fixed_noise = fixed_noise.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        episode_lengths = episode_lengths.to(self.device)

        # Normalize observations if normalizer is provided
        if self.normalizer is not None:
            obs_dict = self.normalizer.normalize(obs_dict)

        # Compute values for GAE
        with torch.no_grad():
            # Reshape observations for value network
            batch_size, max_length = rewards.shape
            flat_obs_dict = {}
            for key, value in obs_dict.items():
                flat_obs_dict[key] = value.reshape(batch_size * max_length, *value.shape[2:])

            flat_values = self.critic(flat_obs_dict).squeeze(-1)
            values = flat_values.reshape(batch_size, max_length)

            # Compute old log probabilities using flow SDE if fixed noise is available
            if fixed_noise is not None:
                flat_fixed_noise = fixed_noise.reshape(batch_size * max_length, *fixed_noise.shape[2:])
                flat_old_log_probs = self.compute_flow_sde_log_prob(
                    flat_obs_dict, flat_actions.reshape(*flat_actions.shape[:-1], -1), flat_fixed_noise)
                old_log_probs = flat_old_log_probs.reshape(batch_size, max_length)
            else:
                # Fallback to simple approximation
                flat_action_dict = self.policy.predict_action(flat_obs_dict)
                flat_predicted_actions = flat_action_dict['action']

                # Simple Gaussian log probability
                action_std = 0.1
                flat_old_log_probs = -0.5 * ((flat_actions - flat_predicted_actions) / action_std).pow(2).sum(dim=-1) - \
                                    0.5 * np.log(2 * np.pi) * flat_actions.shape[-1] - np.log(action_std) * flat_actions.shape[-1]
                old_log_probs = flat_old_log_probs.reshape(batch_size, max_length)

        # Compute GAE advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, episode_lengths)

        # Create mask for valid timesteps
        max_length = rewards.shape[1]
        timestep_indices = torch.arange(max_length, device=self.device).unsqueeze(0).expand(batch_size, -1)
        valid_mask = timestep_indices < episode_lengths.unsqueeze(1)

        # Flatten data for mini-batch training, but only include valid timesteps
        valid_indices = valid_mask.flatten()

        flat_obs_dict = {}
        for key, value in obs_dict.items():
            flat_obs_dict[key] = value.reshape(batch_size * max_length, *value.shape[2:])[valid_indices]

        flat_actions = actions.reshape(batch_size * max_length, -1)[valid_indices]
        flat_fixed_noise = None
        if fixed_noise is not None:
            flat_fixed_noise = fixed_noise.reshape(batch_size * max_length, *fixed_noise.shape[2:])[valid_indices]
        flat_old_log_probs = old_log_probs.flatten()[valid_indices]
        flat_advantages = advantages.flatten()[valid_indices]
        flat_returns = returns.flatten()[valid_indices]

        total_valid_steps = valid_indices.sum().item()

        # Training metrics
        all_metrics = []

        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(total_valid_steps, device=self.device)

            # Mini-batch training
            for start_idx in range(0, total_valid_steps, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, total_valid_steps)
                mb_indices = indices[start_idx:end_idx]

                # Extract mini-batch
                mb_obs_dict = {key: value[mb_indices] for key, value in flat_obs_dict.items()}
                mb_actions = flat_actions[mb_indices]
                mb_fixed_noise = None
                if flat_fixed_noise is not None:
                    mb_fixed_noise = flat_fixed_noise[mb_indices]
                mb_old_log_probs = flat_old_log_probs[mb_indices]
                mb_advantages = flat_advantages[mb_indices]
                mb_returns = flat_returns[mb_indices]

                # Update policy
                self.policy_optimizer.zero_grad()
                policy_loss, policy_metrics = self.compute_policy_loss(
                    mb_obs_dict, mb_actions, mb_old_log_probs, mb_advantages, mb_fixed_noise)
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                value_loss, value_metrics = self.compute_value_loss(mb_obs_dict, mb_returns)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # Collect metrics
                epoch_metrics = {**policy_metrics, **value_metrics}
                epoch_metrics['epoch'] = epoch
                all_metrics.append(epoch_metrics)

        # Average metrics across all mini-batches
        final_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                if key != 'epoch':
                    final_metrics[key] = np.mean([m[key] for m in all_metrics])

        # Add additional metrics
        final_metrics.update({
            'advantages_mean': advantages[valid_mask].mean().item(),
            'advantages_std': advantages[valid_mask].std().item(),
            'returns_mean': returns[valid_mask].mean().item(),
            'returns_std': returns[valid_mask].std().item(),
            'rewards_mean': rewards[valid_mask].mean().item(),
            'valid_steps': total_valid_steps,
        })

        self.train_step += 1
        return final_metrics