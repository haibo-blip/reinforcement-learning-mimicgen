#!/usr/bin/env python3
"""
ManiFlow Advantage and Return Calculator
Computes GAE advantages and returns for PPO training following RLinf pattern.
"""

import numpy as np
import torch
from typing import Optional, Literal
from dataclasses import dataclass

from .maniflow_rollout_collector import ManiFlowRolloutBatch


@dataclass
class AdvantageConfig:
    """Configuration for advantage calculation following RLinf pattern."""

    # GAE parameters
    gamma: float = 0.99          # Discount factor
    gae_lambda: float = 0.95     # GAE lambda parameter

    # Value function parameters
    value_coef: float = 0.5      # Value loss coefficient
    use_value_clipping: bool = True
    value_clip_range: float = 0.2

    # Advantage normalization
    normalize_advantages: bool = True
    advantage_eps: float = 1e-8

    # Return computation
    advantage_type: Literal["gae", "monte_carlo", "n_step"] = "gae"
    n_step: int = 5              # For n-step returns

    def __post_init__(self):
        print(f"📊 Advantage Config:")
        print(f"  - Advantage type: {self.advantage_type}")
        print(f"  - Gamma: {self.gamma}")
        print(f"  - GAE Lambda: {self.gae_lambda}")
        print(f"  - Normalize advantages: {self.normalize_advantages}")


class ManiFlowAdvantageCalculator:
    """
    Advantage and return calculator for ManiFlow RL policy following RLinf pattern.

    Implements GAE (Generalized Advantage Estimation) and other advantage methods.
    """

    def __init__(self, config: AdvantageConfig):
        self.config = config
        print(f"📊 ManiFlow Advantage Calculator initialized")

    def calculate_advantages_and_returns(self,
                                       rollout_batch: ManiFlowRolloutBatch,
                                       next_values: Optional[np.ndarray] = None) -> ManiFlowRolloutBatch:
        """
        Calculate advantages and returns following RLinf pattern.

        Args:
            rollout_batch: Rollout data
            next_values: Value estimates for states after the last step [batch_size, 1]
                        If None, assumes terminal states (value = 0)

        Returns:
            Updated rollout_batch with advantages and returns
        """
        if self.config.advantage_type == "gae":
            return self._calculate_gae_advantages(rollout_batch, next_values)
        elif self.config.advantage_type == "monte_carlo":
            return self._calculate_mc_advantages(rollout_batch, next_values)
        elif self.config.advantage_type == "n_step":
            return self._calculate_n_step_advantages(rollout_batch, next_values)
        else:
            raise ValueError(f"Unknown advantage type: {self.config.advantage_type}")

    def _calculate_gae_advantages(self,
                                rollout_batch: ManiFlowRolloutBatch,
                                next_values: Optional[np.ndarray] = None) -> ManiFlowRolloutBatch:
        """
        Calculate GAE advantages following RLinf OpenPI pattern.

        RLinf GAE formula:
        δ_t = r_t + γ * V(s_{t+1}) * (~done_{t+1}) - V(s_t)
        A_t = δ_t + γ * λ * (~done_{t+1}) * A_{t+1}

        Key difference from standard GAE: RLinf sums rewards across action chunks
        """
        n_steps, batch_size = rollout_batch.rewards.shape[:2]
        action_chunk = rollout_batch.rewards.shape[2]

        # Get value estimates [n_steps, batch_size, 1]
        values = rollout_batch.prev_values  # [n_steps, batch_size, 1]

        # Handle next values (terminal or bootstrap)
        if next_values is None:
            next_values = np.zeros((batch_size, 1), dtype=np.float32)  # Terminal states

        # Following RLinf: Sum rewards across action chunks (not average)
        # This matches RLinf's chunk_level reward aggregation
        chunk_rewards = rollout_batch.rewards.sum(axis=2, keepdims=True)  # [n_steps, batch_size, 1]

        # Prepare outputs
        advantages = np.zeros((n_steps, batch_size, action_chunk), dtype=np.float32)
        returns = np.zeros((n_steps, batch_size, 1), dtype=np.float32)

        # Create full value sequence with bootstrap [n_steps+1, batch_size, 1]
        values_with_bootstrap = np.concatenate([values, next_values.reshape(1, batch_size, 1)], axis=0)

        # GAE calculation: work backwards through time
        gae = np.zeros((batch_size, 1), dtype=np.float32)

        for step in reversed(range(n_steps)):
            # Current step data
            reward = chunk_rewards[step]            # [batch_size, 1] - sum across chunks
            done = rollout_batch.dones[step]        # [batch_size, 1]
            current_value = values_with_bootstrap[step]       # [batch_size, 1]
            next_value = values_with_bootstrap[step + 1]      # [batch_size, 1]

            # TD error following RLinf pattern
            delta = reward + self.config.gamma * next_value * (1.0 - done) - current_value

            # GAE advantage recursion following RLinf pattern
            gae = delta + self.config.gamma * self.config.gae_lambda * (1.0 - done) * gae

            # Store advantage for this step (broadcast to all action chunks)
            advantages[step] = np.tile(gae, (1, action_chunk))  # [batch_size, action_chunk]

            # Compute returns following RLinf: returns = advantages + values
            returns[step] = gae + current_value

        # Normalize advantages if requested
        if self.config.normalize_advantages:
            advantages_flat = advantages.flatten()
            advantages_mean = np.mean(advantages_flat)
            advantages_std = np.std(advantages_flat)
            advantages = (advantages - advantages_mean) / (advantages_std + self.config.advantage_eps)

        print(f"📊 GAE Calculation completed:")
        print(f"  - Advantages shape: {advantages.shape}")
        print(f"  - Returns shape: {returns.shape}")
        print(f"  - Advantage range: [{advantages.min():.3f}, {advantages.max():.3f}]")
        print(f"  - Return range: [{returns.min():.3f}, {returns.max():.3f}]")

        # Update rollout batch
        rollout_batch.advantages = advantages
        rollout_batch.returns = returns

        return rollout_batch

    def _calculate_mc_advantages(self,
                               rollout_batch: ManiFlowRolloutBatch,
                               next_values: Optional[np.ndarray] = None) -> ManiFlowRolloutBatch:
        """Calculate Monte Carlo advantages (simple discounted returns)."""
        n_steps, batch_size = rollout_batch.rewards.shape[:2]
        action_chunk = rollout_batch.rewards.shape[2]

        values = rollout_batch.prev_values  # [n_steps, batch_size, 1]

        # Compute discounted returns
        returns = np.zeros((n_steps, batch_size, 1), dtype=np.float32)
        advantages = np.zeros((n_steps, batch_size, action_chunk), dtype=np.float32)

        # Start from terminal value
        next_return = next_values if next_values is not None else np.zeros((batch_size, 1), dtype=np.float32)

        for t in reversed(range(n_steps)):
            rewards = rollout_batch.rewards[t]      # [batch_size, action_chunk]
            dones = rollout_batch.dones[t]          # [batch_size, 1]
            current_values = values[t]              # [batch_size, 1]

            # Average reward across chunks
            avg_reward = rewards.mean(axis=1, keepdims=True)

            # Monte Carlo return
            next_return = avg_reward + self.config.gamma * next_return * (1.0 - dones)
            returns[t] = next_return

            # Advantage = Return - Value
            advantage = next_return - current_values
            advantages[t] = np.tile(advantage, (1, action_chunk))

        # Normalize advantages
        if self.config.normalize_advantages:
            advantages_flat = advantages.flatten()
            advantages_mean = np.mean(advantages_flat)
            advantages_std = np.std(advantages_flat)
            advantages = (advantages - advantages_mean) / (advantages_std + self.config.advantage_eps)

        rollout_batch.advantages = advantages
        rollout_batch.returns = returns

        return rollout_batch

    def _calculate_n_step_advantages(self,
                                   rollout_batch: ManiFlowRolloutBatch,
                                   next_values: Optional[np.ndarray] = None) -> ManiFlowRolloutBatch:
        """Calculate n-step advantages."""
        n_steps, batch_size = rollout_batch.rewards.shape[:2]
        action_chunk = rollout_batch.rewards.shape[2]
        n = min(self.config.n_step, n_steps)

        values = rollout_batch.prev_values  # [n_steps, batch_size, 1]

        advantages = np.zeros((n_steps, batch_size, action_chunk), dtype=np.float32)
        returns = np.zeros((n_steps, batch_size, 1), dtype=np.float32)

        for t in range(n_steps):
            # Compute n-step return
            n_step_return = 0.0
            gamma_power = 1.0

            for k in range(min(n, n_steps - t)):
                rewards = rollout_batch.rewards[t + k]  # [batch_size, action_chunk]
                avg_reward = rewards.mean(axis=1, keepdims=True)
                n_step_return += gamma_power * avg_reward
                gamma_power *= self.config.gamma

            # Bootstrap with value if not terminal
            if t + n < n_steps:
                bootstrap_values = values[t + n]
                dones = rollout_batch.dones[t + n - 1]
                n_step_return += gamma_power * bootstrap_values * (1.0 - dones)
            elif next_values is not None:
                dones = rollout_batch.dones[-1]
                n_step_return += gamma_power * next_values * (1.0 - dones)

            returns[t] = n_step_return
            advantage = n_step_return - values[t]
            advantages[t] = np.tile(advantage, (1, action_chunk))

        # Normalize advantages
        if self.config.normalize_advantages:
            advantages_flat = advantages.flatten()
            advantages_mean = np.mean(advantages_flat)
            advantages_std = np.std(advantages_flat)
            advantages = (advantages - advantages_mean) / (advantages_std + self.config.advantage_eps)

        rollout_batch.advantages = advantages
        rollout_batch.returns = returns

        return rollout_batch


def test_advantage_calculator():
    """Test the advantage calculator with dummy data."""
    print("🧪 Testing ManiFlow Advantage Calculator")
    print("=" * 50)

    # Create dummy rollout batch
    n_steps, batch_size, action_chunk = 10, 4, 8

    # Simulate increasing rewards over time
    rewards = np.random.randn(n_steps, batch_size, action_chunk).astype(np.float32)
    for t in range(n_steps):
        rewards[t] += t * 0.1  # Increasing trend

    # Simulate value estimates (slightly underestimating rewards)
    prev_values = np.random.randn(n_steps, batch_size, 1).astype(np.float32)
    for t in range(n_steps):
        prev_values[t] += t * 0.05  # Slower increase than rewards

    # Simulate episode termination
    dones = np.zeros((n_steps, batch_size, 1), dtype=np.float32)
    dones[-2, 1, 0] = 1.0  # Episode 1 ends at step -2
    dones[-1, 3, 0] = 1.0  # Episode 3 ends at step -1

    # Create dummy rollout batch
    class DummyRolloutBatch:
        def __init__(self):
            self.rewards = rewards
            self.prev_values = prev_values
            self.dones = dones
            self.advantages = None
            self.returns = None

    rollout_batch = DummyRolloutBatch()

    # Test different advantage methods
    for adv_type in ["gae", "monte_carlo", "n_step"]:
        print(f"\n🔬 Testing {adv_type.upper()} advantages...")

        config = AdvantageConfig(advantage_type=adv_type, normalize_advantages=True)
        calculator = ManiFlowAdvantageCalculator(config)

        try:
            # Calculate advantages
            updated_batch = calculator.calculate_advantages_and_returns(rollout_batch)

            print(f"✅ {adv_type.upper()} calculation successful!")
            print(f"  - Advantages mean: {updated_batch.advantages.mean():.6f}")
            print(f"  - Advantages std: {updated_batch.advantages.std():.6f}")
            print(f"  - Returns mean: {updated_batch.returns.mean():.6f}")

            # Reset for next test
            rollout_batch.advantages = None
            rollout_batch.returns = None

        except Exception as e:
            print(f"❌ {adv_type.upper()} calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print(f"\n✅ All advantage calculation methods tested successfully!")
    return True


if __name__ == "__main__":
    test_advantage_calculator()