#!/usr/bin/env python3
"""
Test Loss Mask Implementation
Verify that the loss mask correctly excludes steps after environment termination.
"""

import torch
import numpy as np
from equi_diffpo.rl_training.rl_utils import compute_loss_mask, masked_mean


def test_compute_loss_mask():
    """Test loss mask computation with different done patterns."""
    print("🧪 Testing compute_loss_mask function...")

    # Test case 1: Single environment, episode ends at step 3
    dones = torch.tensor([
        [[0]],  # Step 0: active
        [[0]],  # Step 1: active
        [[1]],  # Step 2: terminated
        [[0]],  # Step 3: bootstrap (should be masked out)
    ], dtype=torch.float32)  # [4, 1, 1] = [n_steps+1, batch_size, action_chunk]

    loss_mask, loss_mask_sum = compute_loss_mask(dones)

    print(f"Input dones shape: {dones.shape}")
    print(f"Input dones:\n{dones.squeeze()}")
    print(f"Loss mask shape: {loss_mask.shape}")
    print(f"Loss mask:\n{loss_mask.squeeze()}")

    # Expected: [True, True, False] - exclude step 2 onwards after termination
    expected_mask = torch.tensor([[True], [True], [False]], dtype=torch.bool)
    assert torch.equal(loss_mask.bool(), expected_mask), f"Expected {expected_mask}, got {loss_mask.bool()}"

    print("✅ Test case 1 passed: Single environment termination")

    # Test case 2: Multiple environments with different termination times
    dones = torch.tensor([
        [[0], [0]],  # Step 0: both active
        [[1], [0]],  # Step 1: env 0 terminates, env 1 active
        [[0], [1]],  # Step 2: env 0 done (masked), env 1 terminates
        [[0], [0]],  # Step 3: bootstrap
    ], dtype=torch.float32)  # [4, 2, 1]

    loss_mask, loss_mask_sum = compute_loss_mask(dones)

    print(f"\nTest case 2 - Multiple environments:")
    print(f"Input dones:\n{dones.squeeze()}")
    print(f"Loss mask:\n{loss_mask.squeeze()}")

    # Expected: env 0 stops at step 1, env 1 stops at step 2
    expected_mask = torch.tensor([
        [True, True],    # Step 0: both active
        [False, True],   # Step 1: env 0 done, env 1 active
        [False, False]   # Step 2: both done
    ], dtype=torch.bool)

    assert torch.equal(loss_mask.bool(), expected_mask), f"Expected {expected_mask}, got {loss_mask.bool()}"

    print("✅ Test case 2 passed: Multiple environments with different termination times")

    # Test case 3: Action chunking
    dones = torch.tensor([
        [[0, 0], [0, 0]],  # Step 0: both envs active for both action chunks
        [[1, 1], [0, 0]],  # Step 1: env 0 terminates, env 1 active
        [[0, 0], [0, 0]],  # Step 2: bootstrap
    ], dtype=torch.float32)  # [3, 2, 2] = action_chunk=2

    loss_mask, loss_mask_sum = compute_loss_mask(dones)

    print(f"\nTest case 3 - Action chunking:")
    print(f"Input dones shape: {dones.shape}")
    print(f"Loss mask shape: {loss_mask.shape}")
    print(f"Loss mask:\n{loss_mask}")

    # Expected: env 0 is masked for both action chunks after step 0
    expected_mask = torch.tensor([
        [[True, True], [True, True]],    # Step 0: both envs, both chunks active
        [[False, False], [True, True]]   # Step 1: env 0 masked, env 1 active
    ], dtype=torch.bool)

    assert torch.equal(loss_mask.bool(), expected_mask), f"Expected {expected_mask}, got {loss_mask.bool()}"

    print("✅ Test case 3 passed: Action chunking support")


def test_masked_mean():
    """Test masked mean computation."""
    print("\n🧪 Testing masked_mean function...")

    # Test data
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([True, True, False, False])

    result = masked_mean(values, mask)
    expected = (1.0 + 2.0) / 2  # Only first two values

    print(f"Values: {values}")
    print(f"Mask: {mask}")
    print(f"Masked mean: {result.item():.3f}")
    print(f"Expected: {expected:.3f}")

    assert abs(result.item() - expected) < 1e-6, f"Expected {expected}, got {result.item()}"

    print("✅ Masked mean test passed")


def test_loss_mask_integration():
    """Test loss mask integration with mock PPO loss computation."""
    print("\n🧪 Testing loss mask integration...")

    batch_size = 2
    action_chunk = 2
    n_steps = 3

    # Mock data
    advantages = torch.randn(batch_size, action_chunk)
    old_logprobs = torch.randn(batch_size, action_chunk)
    new_logprobs = torch.randn(batch_size, action_chunk)

    # Loss mask: env 0 terminates early, env 1 runs full episode
    loss_mask = torch.tensor([
        [True, False],   # env 0: only first action chunk valid
        [True, True]     # env 1: both action chunks valid
    ], dtype=torch.bool)

    # Compute ratio and policy loss
    ratio = torch.exp(new_logprobs - old_logprobs)
    policy_loss_unmasked = -advantages * ratio

    # Apply mask
    policy_loss_masked = masked_mean(policy_loss_unmasked, loss_mask)
    policy_loss_unmasked_mean = policy_loss_unmasked.mean()

    print(f"Policy loss (unmasked): {policy_loss_unmasked_mean.item():.6f}")
    print(f"Policy loss (masked): {policy_loss_masked.item():.6f}")
    print(f"Loss mask:\n{loss_mask}")

    # The masked loss should only consider valid steps
    valid_elements = policy_loss_unmasked[loss_mask]
    expected_masked_loss = valid_elements.mean()

    assert abs(policy_loss_masked.item() - expected_masked_loss.item()) < 1e-6

    print("✅ Loss mask integration test passed")


if __name__ == "__main__":
    print("🎯 Testing Loss Mask Implementation")
    print("=" * 50)

    test_compute_loss_mask()
    test_masked_mean()
    test_loss_mask_integration()

    print("\n🎉 All loss mask tests passed!")
    print("✅ Loss masking will correctly exclude steps after environment termination")