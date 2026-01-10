#!/usr/bin/env python3
"""
RL Training Utilities for ManiFlow
Includes loss masking and other RL-specific utilities following RLinf patterns.
"""

import torch
import numpy as np
from typing import Tuple


def compute_loss_mask(dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute loss mask to exclude training on steps after environment termination.

    Follows RLinf pattern: create a mask based on cumulative done states to exclude
    steps after an environment is finished from loss calculation.

    Args:
        dones: [n_chunk_steps+1, batch_size, num_action_chunks] or [n_steps+1, batch_size, action_chunk]
               Done flags where True indicates environment termination

    Returns:
        loss_mask: [n_chunk_steps, batch_size, num_action_chunks]
                  True for steps that should be included in loss calculation
        loss_mask_sum: [n_chunk_steps, batch_size, num_action_chunks]
                      Sum of valid steps per batch element (broadcasted)

    Logic:
        - Use cumulative sum to find first termination
        - Mask out all steps after termination
        - Return inverted mask (True = include in loss)
    """
    # Extract dimensions
    # Input shape: [n_chunk_steps+1, batch_size, num_action_chunks]
    n_steps_plus_one, batch_size, num_action_chunks = dones.shape
    n_chunk_steps = n_steps_plus_one - 1

    # Flatten to [total_steps, batch_size] for easier processing
    flattened_dones = dones.transpose(1, 2).reshape(-1, batch_size)
    # Shape: [(n_chunk_steps+1) * num_action_chunks, batch_size]

    # Take the last n_steps + 1 for processing
    flattened_dones = flattened_dones[-(n_chunk_steps * num_action_chunks + 1):]
    # Shape: [n_steps+1, batch_size]

    # Compute cumulative sum to find first done
    # cumsum == 0 means environment is still active
    # cumsum > 0 means environment terminated in previous or current step
    flattened_loss_mask = (flattened_dones.cumsum(dim=0) == 0)[:-1]
    # Shape: [n_steps, batch_size] - exclude last step

    # Reshape back to [n_chunk_steps, num_action_chunks, batch_size]
    loss_mask = flattened_loss_mask.reshape(n_chunk_steps, num_action_chunks, batch_size)

    # Transpose to [n_chunk_steps, batch_size, num_action_chunks]
    loss_mask = loss_mask.transpose(1, 2)

    # Compute sum of valid steps per batch element
    loss_mask_sum = loss_mask.sum(dim=(0, 2), keepdim=True)  # [1, batch_size, 1]
    loss_mask_sum = loss_mask_sum.expand_as(loss_mask)      # Broadcast to full shape

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