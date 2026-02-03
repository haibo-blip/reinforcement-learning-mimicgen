#!/usr/bin/env python3
"""
RL Training Utilities for ManiFlow
Includes loss masking and other RL-specific utilities following RLinf patterns.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def compute_loss_mask(dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute loss mask to exclude training on steps after environment termination.

    Uses cumulative sum along step dimension to detect first termination,
    then masks out all subsequent steps. The step where done=1 is still valid
    (the action caused the termination), only steps AFTER done are invalid.

    Args:
        dones: [n_steps+1, batch_size, action_chunk]
               Done flags where 1 indicates environment termination.
               Includes bootstrap step at the end.

    Returns:
        loss_mask: [n_steps, batch_size, action_chunk]
                  True for steps that should be included in loss calculation
        loss_mask_sum: [n_steps, batch_size, action_chunk]
                      Sum of valid steps per batch element (broadcasted)

    Logic:
        - cumsum along step dimension (dim=0) for each batch independently
        - Use SHIFTED cumsum: step i checks if done occurred BEFORE step i
        - step where done=1 is still valid, only steps after are invalid

    Example:
        dones:          [0, 0, 1, 0]  (step 2 done)
        cumsum:         [0, 0, 1, 1]
        shifted_cumsum: [0, 0, 0, 1]  (prepend 0, remove last)
        mask:           [T, T, T, F]  (step 2 still valid!)
    """
    n_steps_plus_one, batch_size, action_chunk = dones.shape
    n_steps = n_steps_plus_one - 1

    # Take first action_chunk position (all positions have same done value)
    # [n_steps+1, batch_size, 1]
    dones_squeezed = dones[:, :, 0:1]

    # Use only actual steps (exclude bootstrap) for cumsum
    # [n_steps, batch_size, 1]
    dones_actual = dones_squeezed[:-1]

    # Cumsum along step dimension (dim=0), independently for each batch
    # [n_steps, batch_size, 1]
    cumsum = dones_actual.cumsum(dim=0)

    # Shift cumsum right: prepend 0, remove last
    # This way step i checks if done occurred BEFORE step i (not at step i)
    # [n_steps, batch_size, 1]
    zero_pad = torch.zeros(1, batch_size, 1, dtype=cumsum.dtype, device=cumsum.device)
    shifted_cumsum = torch.cat([zero_pad, cumsum[:-1]], dim=0)

    # shifted_cumsum == 0 means no done has occurred before this step
    # [n_steps, batch_size, 1]
    loss_mask = (shifted_cumsum == 0)

    # Broadcast back to full action_chunk dimension
    # [n_steps, batch_size, action_chunk]
    loss_mask = loss_mask.expand(-1, -1, action_chunk)

    # Compute sum of valid steps per batch element
    loss_mask_sum = loss_mask.sum(dim=(0, 2), keepdim=True).expand_as(loss_mask)

    return loss_mask, loss_mask_sum


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute masked mean, avoiding division by zero.

    Args:
        tensor: Values to average
        mask: Boolean mask (True = include)
        eps: Small value to avoid division by zero

    Returns:
        Masked mean value
    """
    masked_sum = (tensor * mask).sum()
    mask_sum = mask.sum()
    return masked_sum / (mask_sum + eps)


def compute_advantages_and_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    loss_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and returns with optional loss masking.

    Args:
        rewards: [n_steps, batch_size, action_chunk]
        values: [n_steps+1, batch_size, 1] (includes bootstrap value)
        dones: [n_steps+1, batch_size, action_chunk]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        loss_mask: Optional mask for valid steps

    Returns:
        advantages: [n_steps, batch_size, action_chunk]
        returns: [n_steps, batch_size, action_chunk]
    """
    n_steps, batch_size, action_chunk = rewards.shape
    device = rewards.device

    # Initialize storage
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # Get value estimates
    current_values = values[:-1]  # [n_steps, batch_size, 1]
    next_values = values[1:]      # [n_steps, batch_size, 1]

    # Broadcast values to match action chunk dimension
    current_values = current_values.expand(-1, -1, action_chunk)
    next_values = next_values.expand(-1, -1, action_chunk)

    # Compute temporal differences
    deltas = rewards + gamma * next_values * (1 - dones[1:]) - current_values

    # Compute GAE advantages (backward pass)
    gae = 0
    for t in reversed(range(n_steps)):
        gae = deltas[t] + gamma * gae_lambda * gae * (1 - dones[t+1])
        advantages[t] = gae

    # Compute returns
    returns = advantages + current_values

    # Apply loss mask if provided
    if loss_mask is not None:
        advantages = advantages * loss_mask
        returns = returns * loss_mask

    return advantages, returns
