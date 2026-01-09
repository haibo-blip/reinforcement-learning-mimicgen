"""
Return computation and advantage estimation methods for RL training.

This module follows RLinf's approach for computing advantages and returns,
specifically designed to work with pi0.5 and other flow-based policies.

Based on RLinf's algorithms/advantages.py implementation.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List


def safe_normalize(tensor: torch.Tensor, loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Safely normalize a tensor using mean and standard deviation.
    Based on RLinf's algorithms/utils.py implementation.

    Args:
        tensor: Tensor to normalize
        loss_mask: Optional mask for valid entries

    Returns:
        Normalized tensor
    """
    if loss_mask is not None:
        # Masked normalization (for sequence data)
        valid_entries = tensor[loss_mask]
        if len(valid_entries) == 0:
            return tensor
        mean = valid_entries.mean()
        std = valid_entries.std()
    else:
        mean = tensor.mean()
        std = tensor.std()

    std = std.clamp(min=1e-8)
    return (tensor - mean) / std


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute mean over masked entries.
    Based on RLinf's utils/utils.py implementation.
    """
    return (tensor * mask).sum() / mask.sum().clamp(min=1)


def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    Based on RLinf's GAE implementation.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Rewards per timestep. Shape: [seq_len, bsz] or [bsz, seq_len].
        values (torch.Tensor): Value function estimates. Shape: [seq_len, bsz] or [bsz, seq_len].
        dones (torch.Tensor): Done flags (1 if episode ended, else 0). Shape: [seq_len, bsz] or [bsz, seq_len].
        gamma (float, optional): Discount factor. Defaults to 1.0.
        gae_lambda (float, optional): GAE smoothing factor. Defaults to 1.0.
        normalize_advantages (bool, optional): Whether to normalize advantages. Defaults to True.
        normalize_returns (bool, optional): Whether to normalize returns. Defaults to False.
        loss_mask (torch.Tensor, optional): Mask for valid entries. Same shape as rewards.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    # Handle different input shapes - convert to [seq_len, bsz] if needed
    if rewards.dim() == 2 and rewards.shape[0] > rewards.shape[1]:
        # Likely [bsz, seq_len] -> transpose to [seq_len, bsz]
        rewards = rewards.transpose(0, 1)
        if values is not None:
            values = values.transpose(0, 1)
        if dones is not None:
            dones = dones.transpose(0, 1)
        if loss_mask is not None:
            loss_mask = loss_mask.transpose(0, 1)
        transposed = True
    else:
        transposed = False

    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    critic_free = values is None
    if critic_free:
        gae_lambda = 1
        gamma = 1

    for step in reversed(range(T)):
        if critic_free:
            delta = rewards[step]
        else:
            if step == T - 1:
                # Last timestep - no next value
                delta = rewards[step] - values[step]
            else:
                delta = (
                    rewards[step]
                    + gamma * values[step + 1] * (~dones[step + 1] if dones is not None else 1.0)
                    - values[step]
                )

        if step == T - 1:
            gae = delta
        else:
            next_non_terminal = (~dones[step + 1] if dones is not None else 1.0)
            gae = delta + gamma * gae_lambda * next_non_terminal * gae

        returns[step] = gae if critic_free else gae + values[step]

    advantages = returns - values if not critic_free else returns

    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    if normalize_returns:
        returns = safe_normalize(returns, loss_mask=loss_mask)

    # Transpose back if we transposed input
    if transposed:
        advantages = advantages.transpose(0, 1)
        returns = returns.transpose(0, 1)

    return advantages, returns


def compute_grpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute GRPO (Group Relative Policy Optimization) advantages.
    Based on RLinf's GRPO implementation.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        group_size (int): Group size for advantage computation.

    Returns:
        torch.Tensor: advantages
        None: returns (GRPO doesn't use returns)
    """
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )

    advantages = grouped_rewards - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask

    return advantages, None


def compute_advantages_and_returns(
    method: str,
    rewards: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    group_size: int = 1,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Unified interface for computing advantages and returns using different methods.
    Based on RLinf's advantage computation registry.

    Args:
        method: Advantage computation method ("gae", "grpo")
        rewards: Reward tensor
        values: Value estimates (required for GAE)
        dones: Done flags (required for GAE)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        normalize_advantages: Whether to normalize advantages
        normalize_returns: Whether to normalize returns
        loss_mask: Mask for valid entries
        group_size: Group size for GRPO

    Returns:
        advantages: Computed advantages
        returns: Computed returns (None for some methods)
    """
    if method == "gae":
        return compute_gae_advantages_and_returns(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_advantages=normalize_advantages,
            normalize_returns=normalize_returns,
            loss_mask=loss_mask,
            **kwargs
        )
    elif method == "grpo":
        if loss_mask is None:
            loss_mask = torch.ones_like(rewards)
        return compute_grpo_advantages(
            rewards=rewards,
            loss_mask=loss_mask,
            group_size=group_size,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown advantage computation method: {method}")


def convert_batch_to_sequence_format(
    batch_data: Dict[str, torch.Tensor],
    episode_lengths: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Convert batch format data to sequence format for RLinf-style computation.

    Converts from [batch_size, max_length, ...] to [max_length, batch_size, ...]
    and creates appropriate loss masks.

    Args:
        batch_data: Dictionary containing batch format data
        episode_lengths: Length of each episode [batch_size]

    Returns:
        Dictionary with sequence format data and loss masks
    """
    batch_size, max_length = batch_data['rewards'].shape[:2]

    # Create loss mask for valid timesteps
    loss_mask = torch.zeros(max_length, batch_size, dtype=torch.bool)
    for i, length in enumerate(episode_lengths):
        loss_mask[:length, i] = True

    # Convert batch data to sequence format
    seq_data = {}
    for key, tensor in batch_data.items():
        if key in ['rewards', 'values', 'dones']:
            seq_data[key] = tensor.transpose(0, 1)  # [max_length, batch_size]
        else:
            seq_data[key] = tensor

    seq_data['loss_mask'] = loss_mask
    return seq_data


# Future TODOs and integration notes:
"""
Integration with RLinf-style RL Training:

1. Main Advantage/Return Functions:
   - compute_gae_advantages_and_returns(): Standard GAE for PPO
   - compute_grpo_advantages(): Group-based advantages for GRPO
   - compute_advantages_and_returns(): Unified interface

2. Utility Functions:
   - safe_normalize(): Normalize tensors with proper masking
   - masked_mean(): Compute means over masked entries
   - convert_batch_to_sequence_format(): Convert data formats for RLinf compatibility

3. Usage Pattern:
   ```python
   # Convert collected batch data to sequence format
   seq_data = convert_batch_to_sequence_format(batch_data, episode_lengths)

   # Compute advantages and returns
   advantages, returns = compute_advantages_and_returns(
       method="gae",
       rewards=seq_data['rewards'],
       values=seq_data['values'],
       dones=seq_data['dones'],
       loss_mask=seq_data['loss_mask'],
       gamma=0.99,
       gae_lambda=0.95
   )
   ```

4. Future Extensions (TODO):
   - Flow-aware advantage estimation using action_stds from pi0.5
   - Uncertainty-weighted advantages based on policy confidence
   - Multi-horizon returns for flow denoising processes
   - Adaptive discount factors for manipulation tasks
   - Reward shaping for specific robotic tasks

5. Key Differences from Standard Implementations:
   - Follows RLinf's [seq_len, batch_size] tensor format
   - Uses loss masks for proper sequence handling
   - Supports both GAE and GRPO methods
   - Designed specifically for flow-based policies like pi0.5

6. Integration with PPO Trainer:
   - Replace existing GAE computation with compute_gae_advantages_and_returns()
   - Use convert_batch_to_sequence_format() for data preparation
   - Apply safe_normalize() for advantage normalization
"""


