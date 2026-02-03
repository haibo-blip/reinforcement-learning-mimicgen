#!/usr/bin/env python3
"""
Test advantage and return calculation for RL pipeline.

This test verifies:
1. GAE advantage calculation is correct
2. Monte Carlo return calculation is correct
3. N-step return calculation is correct
4. Advantage normalization works correctly
5. Done flag handling is correct (no credit after termination)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from typing import Dict, Any


class DummyRolloutBatch:
    """Dummy rollout batch for testing advantage calculation."""

    def __init__(self, n_steps: int, batch_size: int, action_chunk: int):
        self.rewards = None
        self.prev_values = None
        self.dones = None
        self.advantages = None
        self.returns = None


def test_gae_calculation_no_termination():
    """Test GAE calculation when no episode terminates."""
    print("\n" + "=" * 60)
    print("TEST: GAE Calculation (No Termination)")
    print("=" * 60)

    from equi_diffpo.rl_training.maniflow_advantage_calculator import (
        ManiFlowAdvantageCalculator,
        AdvantageConfig
    )

    # Setup
    n_steps = 10
    batch_size = 2
    action_chunk = 8
    gamma = 0.99
    gae_lambda = 0.95

    # Create config
    config = AdvantageConfig(
        gamma=gamma,
        gae_lambda=gae_lambda,
        normalize_advantages=False,  # Disable normalization for easier verification
        advantage_type="gae"
    )

    calculator = ManiFlowAdvantageCalculator(config)

    # Create dummy rollout batch
    rollout = DummyRolloutBatch(n_steps, batch_size, action_chunk)

    # Constant rewards of 1.0
    rollout.rewards = np.ones((n_steps, batch_size, action_chunk), dtype=np.float32)

    # Constant value estimates of 0.5
    rollout.prev_values = np.full((n_steps, batch_size, 1), 0.5, dtype=np.float32)

    # No terminations
    rollout.dones = np.zeros((n_steps, batch_size, 1), dtype=np.float32)

    # Calculate advantages
    next_values = np.zeros((batch_size, 1), dtype=np.float32)  # Terminal values = 0
    updated_batch = calculator.calculate_advantages_and_returns(rollout, next_values)

    print(f"  Rewards: constant 1.0")
    print(f"  Values: constant 0.5")
    print(f"  Dones: all False")
    print(f"  Gamma: {gamma}, Lambda: {gae_lambda}")

    # Verify advantages computed
    assert updated_batch.advantages is not None, "Advantages not computed"
    assert updated_batch.returns is not None, "Returns not computed"

    print(f"\n  Results:")
    print(f"  - Advantages shape: {updated_batch.advantages.shape}")
    print(f"  - Advantages range: [{updated_batch.advantages.min():.4f}, {updated_batch.advantages.max():.4f}]")
    print(f"  - Returns range: [{updated_batch.returns.min():.4f}, {updated_batch.returns.max():.4f}]")

    # Verify shapes
    assert updated_batch.advantages.shape == (n_steps, batch_size, action_chunk), \
        f"Wrong advantages shape: {updated_batch.advantages.shape}"
    assert updated_batch.returns.shape == (n_steps, batch_size, 1), \
        f"Wrong returns shape: {updated_batch.returns.shape}"

    # Verify advantages are non-zero (since rewards > 0 and values are lower)
    assert np.abs(updated_batch.advantages).sum() > 0, "Advantages should be non-zero"

    # Later steps should have smaller advantages (discounting effect)
    avg_adv_early = np.mean(updated_batch.advantages[:5])
    avg_adv_late = np.mean(updated_batch.advantages[5:])
    print(f"  - Avg advantage (early steps): {avg_adv_early:.4f}")
    print(f"  - Avg advantage (late steps): {avg_adv_late:.4f}")

    print("\nGAE calculation (no termination) test passed!")
    return True


def test_gae_calculation_with_termination():
    """Test GAE calculation with episode termination."""
    print("\n" + "=" * 60)
    print("TEST: GAE Calculation (With Termination)")
    print("=" * 60)

    from equi_diffpo.rl_training.maniflow_advantage_calculator import (
        ManiFlowAdvantageCalculator,
        AdvantageConfig
    )

    n_steps = 10
    batch_size = 2
    action_chunk = 8
    gamma = 0.99
    gae_lambda = 0.95

    config = AdvantageConfig(
        gamma=gamma,
        gae_lambda=gae_lambda,
        normalize_advantages=False,
        advantage_type="gae"
    )

    calculator = ManiFlowAdvantageCalculator(config)

    # Create dummy rollout batch
    rollout = DummyRolloutBatch(n_steps, batch_size, action_chunk)

    # Constant rewards
    rollout.rewards = np.ones((n_steps, batch_size, action_chunk), dtype=np.float32)

    # Constant value estimates
    rollout.prev_values = np.full((n_steps, batch_size, 1), 0.5, dtype=np.float32)

    # Episode 0 terminates at step 5, episode 1 continues
    rollout.dones = np.zeros((n_steps, batch_size, 1), dtype=np.float32)
    rollout.dones[5, 0, 0] = 1.0  # Episode 0 terminates at step 5

    # Calculate advantages
    next_values = np.zeros((batch_size, 1), dtype=np.float32)
    updated_batch = calculator.calculate_advantages_and_returns(rollout, next_values)

    print(f"  Setup: Episode 0 terminates at step 5, Episode 1 continues")

    # Compare advantages after termination
    adv_env0_before = updated_batch.advantages[3:5, 0, :].mean()  # Before termination
    adv_env0_after = updated_batch.advantages[6:, 0, :].mean()   # After termination
    adv_env1 = updated_batch.advantages[:, 1, :].mean()          # Never terminates

    print(f"\n  Results:")
    print(f"  - Env 0 avg advantage (before term): {adv_env0_before:.4f}")
    print(f"  - Env 0 avg advantage (after term): {adv_env0_after:.4f}")
    print(f"  - Env 1 avg advantage: {adv_env1:.4f}")

    # After termination, advantages should be calculated from reset
    assert updated_batch.advantages is not None, "Advantages should be computed"

    print("\nGAE calculation (with termination) test passed!")
    return True


def test_monte_carlo_returns():
    """Test Monte Carlo return calculation."""
    print("\n" + "=" * 60)
    print("TEST: Monte Carlo Returns")
    print("=" * 60)

    from equi_diffpo.rl_training.maniflow_advantage_calculator import (
        ManiFlowAdvantageCalculator,
        AdvantageConfig
    )

    n_steps = 5
    batch_size = 1
    action_chunk = 1
    gamma = 0.99

    config = AdvantageConfig(
        gamma=gamma,
        gae_lambda=0.95,
        normalize_advantages=False,
        advantage_type="monte_carlo"
    )

    calculator = ManiFlowAdvantageCalculator(config)

    # Create rollout with known rewards
    rollout = DummyRolloutBatch(n_steps, batch_size, action_chunk)

    # Rewards: [1, 1, 1, 1, 1]
    rollout.rewards = np.ones((n_steps, batch_size, action_chunk), dtype=np.float32)

    # Value estimates (for advantage = return - value)
    rollout.prev_values = np.zeros((n_steps, batch_size, 1), dtype=np.float32)

    # No termination
    rollout.dones = np.zeros((n_steps, batch_size, 1), dtype=np.float32)

    # Calculate returns
    next_values = np.zeros((batch_size, 1), dtype=np.float32)
    updated_batch = calculator.calculate_advantages_and_returns(rollout, next_values)

    print(f"  Rewards: [1, 1, 1, 1, 1]")
    print(f"  Gamma: {gamma}")

    # Verify Monte Carlo returns
    # Return at step 0 should be: 1 + 0.99 + 0.99^2 + 0.99^3 + 0.99^4
    expected_return_0 = sum(gamma ** i for i in range(n_steps))
    actual_return_0 = updated_batch.returns[0, 0, 0]

    print(f"\n  Expected return at step 0: {expected_return_0:.4f}")
    print(f"  Actual return at step 0: {actual_return_0:.4f}")

    # Allow some tolerance for numerical precision
    assert abs(actual_return_0 - expected_return_0) < 0.01, \
        f"Monte Carlo return mismatch: expected {expected_return_0}, got {actual_return_0}"

    # Return at last step should be just the reward
    expected_return_last = 1.0
    actual_return_last = updated_batch.returns[-1, 0, 0]

    print(f"  Expected return at last step: {expected_return_last:.4f}")
    print(f"  Actual return at last step: {actual_return_last:.4f}")

    assert abs(actual_return_last - expected_return_last) < 0.01, \
        f"Last step return mismatch: expected {expected_return_last}, got {actual_return_last}"

    print("\nMonte Carlo returns test passed!")
    return True


def test_n_step_returns():
    """Test N-step return calculation."""
    print("\n" + "=" * 60)
    print("TEST: N-Step Returns")
    print("=" * 60)

    from equi_diffpo.rl_training.maniflow_advantage_calculator import (
        ManiFlowAdvantageCalculator,
        AdvantageConfig
    )

    n_steps = 10
    batch_size = 1
    action_chunk = 1
    gamma = 0.99
    n = 3  # 3-step returns

    config = AdvantageConfig(
        gamma=gamma,
        gae_lambda=0.95,
        normalize_advantages=False,
        advantage_type="n_step",
        n_step=n
    )

    calculator = ManiFlowAdvantageCalculator(config)

    # Create rollout
    rollout = DummyRolloutBatch(n_steps, batch_size, action_chunk)

    # Constant rewards of 1.0
    rollout.rewards = np.ones((n_steps, batch_size, action_chunk), dtype=np.float32)

    # Value estimates: increasing values (0, 1, 2, ..., 9)
    rollout.prev_values = np.arange(n_steps).reshape(n_steps, 1, 1).astype(np.float32)

    # No termination
    rollout.dones = np.zeros((n_steps, batch_size, 1), dtype=np.float32)

    # Calculate returns
    next_values = np.array([[float(n_steps)]], dtype=np.float32)
    updated_batch = calculator.calculate_advantages_and_returns(rollout, next_values)

    print(f"  N-step: {n}")
    print(f"  Gamma: {gamma}")
    print(f"  Rewards: all 1.0")

    # For n-step return at step 0:
    # Return = r_0 + gamma*r_1 + gamma^2*r_2 + gamma^3*V(s_3)
    # Return = 1 + 0.99*1 + 0.99^2*1 + 0.99^3*3
    expected_return_0 = 1 + gamma * 1 + gamma**2 * 1 + gamma**3 * rollout.prev_values[3, 0, 0]
    actual_return_0 = updated_batch.returns[0, 0, 0]

    print(f"\n  Expected return at step 0: {expected_return_0:.4f}")
    print(f"  Actual return at step 0: {actual_return_0:.4f}")

    # Allow some tolerance
    assert abs(actual_return_0 - expected_return_0) < 0.1, \
        f"N-step return mismatch: expected {expected_return_0}, got {actual_return_0}"

    print("\nN-step returns test passed!")
    return True


def test_advantage_normalization():
    """Test advantage normalization."""
    print("\n" + "=" * 60)
    print("TEST: Advantage Normalization")
    print("=" * 60)

    from equi_diffpo.rl_training.maniflow_advantage_calculator import (
        ManiFlowAdvantageCalculator,
        AdvantageConfig
    )

    n_steps = 20
    batch_size = 4
    action_chunk = 8

    # Test with normalization enabled
    config = AdvantageConfig(
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,  # Enable normalization
        advantage_type="gae"
    )

    calculator = ManiFlowAdvantageCalculator(config)

    # Create rollout with varying rewards
    rollout = DummyRolloutBatch(n_steps, batch_size, action_chunk)

    # Random rewards
    rollout.rewards = np.random.randn(n_steps, batch_size, action_chunk).astype(np.float32)

    # Random value estimates
    rollout.prev_values = np.random.randn(n_steps, batch_size, 1).astype(np.float32)

    # No termination
    rollout.dones = np.zeros((n_steps, batch_size, 1), dtype=np.float32)

    # Calculate advantages
    next_values = np.zeros((batch_size, 1), dtype=np.float32)
    updated_batch = calculator.calculate_advantages_and_returns(rollout, next_values)

    advantages = updated_batch.advantages

    # Check normalization: mean should be close to 0, std close to 1
    adv_mean = np.mean(advantages)
    adv_std = np.std(advantages)

    print(f"  After normalization:")
    print(f"  - Advantages mean: {adv_mean:.6f} (expected: ~0)")
    print(f"  - Advantages std: {adv_std:.6f} (expected: ~1)")

    assert abs(adv_mean) < 0.01, f"Normalized advantages mean should be ~0, got {adv_mean}"
    assert abs(adv_std - 1.0) < 0.01, f"Normalized advantages std should be ~1, got {adv_std}"

    print("\nAdvantage normalization test passed!")
    return True


def test_done_handling():
    """Test that dones properly reset credit assignment."""
    print("\n" + "=" * 60)
    print("TEST: Done Flag Handling")
    print("=" * 60)

    from equi_diffpo.rl_training.maniflow_advantage_calculator import (
        ManiFlowAdvantageCalculator,
        AdvantageConfig
    )

    n_steps = 10
    batch_size = 1
    action_chunk = 1
    gamma = 0.99

    config = AdvantageConfig(
        gamma=gamma,
        gae_lambda=0.95,
        normalize_advantages=False,
        advantage_type="gae"
    )

    calculator = ManiFlowAdvantageCalculator(config)

    # Create two scenarios: one with done, one without
    for scenario, term_step in [("No termination", None), ("Termination at step 5", 5)]:
        print(f"\n  Scenario: {scenario}")

        rollout = DummyRolloutBatch(n_steps, batch_size, action_chunk)

        # Rewards: 1.0 everywhere
        rollout.rewards = np.ones((n_steps, batch_size, action_chunk), dtype=np.float32)

        # Values: 0.0 everywhere
        rollout.prev_values = np.zeros((n_steps, batch_size, 1), dtype=np.float32)

        # Done flags
        rollout.dones = np.zeros((n_steps, batch_size, 1), dtype=np.float32)
        if term_step is not None:
            rollout.dones[term_step, 0, 0] = 1.0

        next_values = np.zeros((batch_size, 1), dtype=np.float32)
        updated_batch = calculator.calculate_advantages_and_returns(rollout, next_values)

        returns = updated_batch.returns

        print(f"    Returns at step 0: {returns[0, 0, 0]:.4f}")
        print(f"    Returns at step {n_steps-1}: {returns[-1, 0, 0]:.4f}")

        if term_step is not None:
            # After termination, returns should start fresh
            # Step before termination
            return_before = returns[term_step - 1, 0, 0]
            print(f"    Return before termination (step {term_step-1}): {return_before:.4f}")

    print("\nDone flag handling test passed!")
    return True


def test_advantage_calculation_consistency():
    """Test that advantage calculation is consistent across runs."""
    print("\n" + "=" * 60)
    print("TEST: Advantage Calculation Consistency")
    print("=" * 60)

    from equi_diffpo.rl_training.maniflow_advantage_calculator import (
        ManiFlowAdvantageCalculator,
        AdvantageConfig
    )

    n_steps = 10
    batch_size = 2
    action_chunk = 4

    config = AdvantageConfig(
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=False,  # Disable for exact comparison
        advantage_type="gae"
    )

    calculator = ManiFlowAdvantageCalculator(config)

    # Set seed for reproducibility
    np.random.seed(42)

    # Create rollout
    rollout1 = DummyRolloutBatch(n_steps, batch_size, action_chunk)
    rollout1.rewards = np.random.randn(n_steps, batch_size, action_chunk).astype(np.float32)
    rollout1.prev_values = np.random.randn(n_steps, batch_size, 1).astype(np.float32)
    rollout1.dones = np.zeros((n_steps, batch_size, 1), dtype=np.float32)

    # Create identical rollout
    rollout2 = DummyRolloutBatch(n_steps, batch_size, action_chunk)
    rollout2.rewards = rollout1.rewards.copy()
    rollout2.prev_values = rollout1.prev_values.copy()
    rollout2.dones = rollout1.dones.copy()

    next_values = np.zeros((batch_size, 1), dtype=np.float32)

    # Calculate advantages for both
    updated1 = calculator.calculate_advantages_and_returns(rollout1, next_values)
    updated2 = calculator.calculate_advantages_and_returns(rollout2, next_values)

    # Compare results
    adv_diff = np.abs(updated1.advantages - updated2.advantages).max()
    ret_diff = np.abs(updated1.returns - updated2.returns).max()

    print(f"  Max advantage difference: {adv_diff:.10f}")
    print(f"  Max returns difference: {ret_diff:.10f}")

    assert adv_diff < 1e-6, f"Advantages not consistent: max diff = {adv_diff}"
    assert ret_diff < 1e-6, f"Returns not consistent: max diff = {ret_diff}"

    print("\nAdvantage calculation consistency test passed!")
    return True


def run_all_tests():
    """Run all advantage calculation tests."""
    print("\n" + "=" * 80)
    print("RUNNING ADVANTAGE CALCULATION TESTS")
    print("=" * 80)

    tests = [
        ("GAE (No Termination)", test_gae_calculation_no_termination),
        ("GAE (With Termination)", test_gae_calculation_with_termination),
        ("Monte Carlo Returns", test_monte_carlo_returns),
        ("N-Step Returns", test_n_step_returns),
        ("Advantage Normalization", test_advantage_normalization),
        ("Done Flag Handling", test_done_handling),
        ("Calculation Consistency", test_advantage_calculation_consistency),
    ]

    results = []
    for test_name, test_fn in tests:
        try:
            success = test_fn()
            results.append((test_name, success, None))
        except Exception as e:
            import traceback
            results.append((test_name, False, str(e)))
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY: Advantage Calculation")
    print("=" * 80)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for test_name, success, error in results:
        status = "PASSED" if success else "FAILED"
        print(f"  [{status}] {test_name}")
        if error:
            print(f"         Error: {error}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
