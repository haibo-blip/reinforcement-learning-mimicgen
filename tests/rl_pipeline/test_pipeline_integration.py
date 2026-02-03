#!/usr/bin/env python3
"""
End-to-end integration test for the RL pipeline.

This test verifies the full pipeline works together:
1. Policy initialization
2. Rollout collection
3. Advantage calculation
4. PPO loss computation
5. Gradient flow
6. Loading pretrained weights
7. Full training step simulation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any


# Standard shape meta for testing
SHAPE_META = {
    'obs': {
        'robot0_eye_in_hand_image': {'shape': [3, 84, 84], 'type': 'rgb'},
        'point_cloud': {'shape': [1024, 6], 'type': 'point_cloud'},
        'robot0_eef_pos': {'shape': [3]},
        'robot0_eef_quat': {'shape': [4]},
        'robot0_gripper_qpos': {'shape': [2]},
    },
    'action': {'shape': [10]}
}


class MockEnvRunner:
    """Mock environment runner that simulates RL data collection."""

    def __init__(self, num_envs=4, max_steps=20):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.action_dim = 10
        self.action_chunk = 8
        self.horizon = 16
        self.num_inference_steps = 10
        self.n_obs_steps = 2

    def run_rl(self, policy):
        """Generate mock RL data."""
        n_steps = self.max_steps
        batch_size = self.num_envs
        N = self.num_inference_steps

        observations = {
            'robot0_eye_in_hand_image': np.random.randn(n_steps, batch_size, self.n_obs_steps, 3, 84, 84).astype(np.float32),
            'point_cloud': np.random.randn(n_steps, batch_size, self.n_obs_steps, 1024, 6).astype(np.float32),
            'robot0_eef_pos': np.random.randn(n_steps, batch_size, self.n_obs_steps, 3).astype(np.float32),
            'robot0_eef_quat': np.random.randn(n_steps, batch_size, self.n_obs_steps, 4).astype(np.float32),
            'robot0_gripper_qpos': np.random.randn(n_steps, batch_size, self.n_obs_steps, 2).astype(np.float32),
        }

        rl_data = {
            'observations': observations,
            'actions': np.random.randn(n_steps, batch_size, self.action_chunk, self.action_dim).astype(np.float32),
            'rewards': np.random.randn(n_steps, batch_size, self.action_chunk).astype(np.float32) * 0.1 + 0.1,
            'dones': np.zeros((n_steps, batch_size, 1), dtype=np.float32),
            'prev_logprobs': np.random.randn(n_steps, batch_size, self.action_chunk, self.action_dim).astype(np.float32) * 0.1 - 1.0,
            'prev_values': np.random.randn(n_steps, batch_size, 1).astype(np.float32),
            'chains': np.random.randn(n_steps, batch_size, N + 1, self.horizon, self.action_dim).astype(np.float32),
            'denoise_inds': np.tile(np.arange(N), (n_steps, batch_size, 1)).astype(np.int64),
            'total_steps': n_steps,
            'total_envs': batch_size,
        }

        # Simulate some episode terminations
        for b in range(batch_size):
            term_step = np.random.randint(n_steps // 2, n_steps)
            rl_data['dones'][term_step:, b, 0] = 1.0

        return {'rl_data': rl_data, 'log_data': {}}


def create_test_policy(device='cpu'):
    """Create a test RL policy."""
    from equi_diffpo.policy.maniflow.maniflow_pointcloud_rl_policy import ManiFlowRLPointcloudPolicy
    from equi_diffpo.model.common.normalizer import LinearNormalizer
    from omegaconf import OmegaConf

    # Create pointcloud_encoder_cfg as OmegaConf object
    # visual_cond_len should equal num_points (DiTX internally multiplies by n_obs_steps)
    visual_cond_len = 64
    n_obs_steps = 2
    pointcloud_encoder_cfg = OmegaConf.create({
        'in_channels': 6,
        'out_channels': 32,
        'use_layernorm': True,
        'final_norm': 'layernorm',
        'normal_channel': False,
        'num_points': visual_cond_len,  # num_points = visual_cond_len
        'pointwise': True,
    })

    policy = ManiFlowRLPointcloudPolicy(
        shape_meta=SHAPE_META,
        horizon=16,
        n_action_steps=8,
        n_obs_steps=n_obs_steps,
        num_inference_steps=10,
        n_layer=2,
        n_head=4,
        n_emb=64,
        encoder_output_dim=32,
        visual_cond_len=visual_cond_len,  # Same as num_points (DiTX multiplies by n_obs_steps internally)
        add_value_head=True,
        noise_method="flow_sde",
        noise_level=0.5,
        pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        downsample_points=True,  # Enable point cloud downsampling to num_points
    )

    # Set up normalizer - use dict fitting pattern
    normalizer = LinearNormalizer()
    obs_data = {}
    for key, meta in SHAPE_META['obs'].items():
        obs_data[key] = torch.randn(100, 2, *meta['shape'])
    obs_data['action'] = torch.randn(100, SHAPE_META['action']['shape'][0])
    normalizer.fit(obs_data)
    policy.set_normalizer(normalizer)

    policy.to(device)
    return policy


def test_full_rollout_pipeline():
    """Test full rollout collection -> advantage calculation pipeline."""
    print("\n" + "=" * 60)
    print("TEST: Full Rollout Pipeline")
    print("=" * 60)

    from equi_diffpo.rl_training.maniflow_rollout_collector import ManiFlowRolloutCollector
    from equi_diffpo.rl_training.maniflow_advantage_calculator import (
        ManiFlowAdvantageCalculator,
        AdvantageConfig
    )

    # Create components
    policy = create_test_policy(device='cpu')
    env_runner = MockEnvRunner(num_envs=4, max_steps=20)

    collector = ManiFlowRolloutCollector(
        policy=policy,
        env_runner=env_runner,
        max_steps_per_episode=100,
        action_chunk_size=8,
        obs_chunk_size=2,
        device='cpu'
    )

    adv_config = AdvantageConfig(
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
        advantage_type="gae"
    )
    advantage_calculator = ManiFlowAdvantageCalculator(adv_config)

    print("  1. Collecting rollouts...")
    runner_results = env_runner.run_rl(policy)
    rollout_batch = collector.collect_rollouts_from_runner_results(runner_results)

    print(f"     - Rollout batch collected")
    print(f"     - Steps: {rollout_batch.n_chunk_steps}")
    print(f"     - Batch size: {rollout_batch.batch_size}")

    print("\n  2. Calculating advantages...")
    next_values = np.zeros((rollout_batch.batch_size, 1), dtype=np.float32)
    rollout_batch = advantage_calculator.calculate_advantages_and_returns(rollout_batch, next_values)

    print(f"     - Advantages computed")
    print(f"     - Advantages shape: {rollout_batch.advantages.shape}")
    print(f"     - Returns shape: {rollout_batch.returns.shape}")

    # Verify data integrity
    assert rollout_batch.advantages is not None, "Advantages should be computed"
    assert rollout_batch.returns is not None, "Returns should be computed"
    assert rollout_batch.loss_mask is not None, "Loss mask should be computed"

    # Verify shapes
    assert rollout_batch.advantages.shape[0] == rollout_batch.n_chunk_steps
    assert rollout_batch.advantages.shape[1] == rollout_batch.batch_size

    print("\nFull rollout pipeline test passed!")
    return True


def test_ppo_loss_computation():
    """Test PPO loss computation with mock data."""
    print("\n" + "=" * 60)
    print("TEST: PPO Loss Computation")
    print("=" * 60)

    from equi_diffpo.rl_training.rl_utils import masked_mean

    # Create test policy
    policy = create_test_policy(device='cpu')
    policy.train()

    # Create mock batch data
    batch_size = 8
    action_chunk = 8
    action_dim = 10
    N = 10
    horizon = 16

    # Prepare mock data
    old_logprobs = torch.randn(batch_size, action_chunk, action_dim) * 0.1 - 1.0
    advantages = torch.randn(batch_size, action_chunk)
    returns = torch.randn(batch_size, 1)
    loss_mask = torch.ones(batch_size, action_chunk)

    # Create observation dict for policy forward
    obs = {
        'robot0_eye_in_hand_image': torch.randn(batch_size, 2, 3, 84, 84),
        'point_cloud': torch.randn(batch_size, 2, 1024, 6),
        'robot0_eef_pos': torch.randn(batch_size, 2, 3),
        'robot0_eef_quat': torch.randn(batch_size, 2, 4),
        'robot0_gripper_qpos': torch.randn(batch_size, 2, 2),
    }

    chains = torch.randn(batch_size, N + 1, horizon, action_dim)
    denoise_inds = torch.zeros(batch_size, N, dtype=torch.long)
    for b in range(batch_size):
        denoise_inds[b] = torch.arange(N)

    print("  1. Running policy forward...")
    data = {
        'observation': obs,
        'chains': chains,
        'denoise_inds': denoise_inds,
    }

    policy_outputs = policy.default_forward(data, compute_values=True)

    new_logprobs = policy_outputs['logprobs']
    values = policy_outputs['values']
    entropy = policy_outputs['entropy']

    print(f"     - new_logprobs shape: {new_logprobs.shape}")
    print(f"     - values shape: {values.shape}")
    print(f"     - entropy shape: {entropy.shape}")

    print("\n  2. Computing PPO loss...")

    # Flatten logprobs
    old_logprobs_flat = old_logprobs.sum(dim=-1)  # [batch, action_chunk]
    new_logprobs_flat = new_logprobs.sum(dim=-1)  # [batch, action_chunk]

    # Importance sampling ratio
    log_ratio = new_logprobs_flat - old_logprobs_flat
    ratio = torch.exp(log_ratio)

    # PPO clip parameters
    clip_range = 0.2

    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

    # Policy loss
    policy_loss_1 = -advantages * ratio
    policy_loss_2 = -advantages * clipped_ratio
    policy_loss_unmasked = torch.max(policy_loss_1, policy_loss_2)
    policy_loss = masked_mean(policy_loss_unmasked, loss_mask.bool())

    # Value loss
    values_expanded = values.unsqueeze(1)
    value_loss = ((values_expanded - returns) ** 2).mean()

    # Entropy loss
    entropy_loss = -entropy.mean()

    # Total loss
    total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

    print(f"     - Policy loss: {policy_loss.item():.6f}")
    print(f"     - Value loss: {value_loss.item():.6f}")
    print(f"     - Entropy loss: {entropy_loss.item():.6f}")
    print(f"     - Total loss: {total_loss.item():.6f}")

    # Verify losses are finite
    assert torch.isfinite(policy_loss), "Policy loss is not finite"
    assert torch.isfinite(value_loss), "Value loss is not finite"
    assert torch.isfinite(entropy_loss), "Entropy loss is not finite"
    assert torch.isfinite(total_loss), "Total loss is not finite"

    print("\nPPO loss computation test passed!")
    return True


def test_gradient_flow():
    """Test that gradients flow through the entire pipeline."""
    print("\n" + "=" * 60)
    print("TEST: Gradient Flow")
    print("=" * 60)

    # Create test policy
    policy = create_test_policy(device='cpu')
    policy.train()

    # Create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    batch_size = 4
    action_chunk = 8
    action_dim = 10
    N = 10
    horizon = 16

    # Create mock data
    obs = {
        'robot0_eye_in_hand_image': torch.randn(batch_size, 2, 3, 84, 84),
        'point_cloud': torch.randn(batch_size, 2, 1024, 6),
        'robot0_eef_pos': torch.randn(batch_size, 2, 3),
        'robot0_eef_quat': torch.randn(batch_size, 2, 4),
        'robot0_gripper_qpos': torch.randn(batch_size, 2, 2),
    }

    chains = torch.randn(batch_size, N + 1, horizon, action_dim, requires_grad=True)
    denoise_inds = torch.zeros(batch_size, N, dtype=torch.long)
    for b in range(batch_size):
        denoise_inds[b] = torch.arange(N)

    advantages = torch.randn(batch_size, action_chunk)
    returns = torch.randn(batch_size, 1)

    print("  1. Computing forward pass...")
    data = {
        'observation': obs,
        'chains': chains,
        'denoise_inds': denoise_inds,
    }

    policy_outputs = policy.default_forward(data, compute_values=True)

    # Simple loss
    logprobs = policy_outputs['logprobs']
    values = policy_outputs['values']

    loss = -logprobs.mean() + (values - returns.squeeze()).pow(2).mean()

    print(f"     - Loss: {loss.item():.6f}")

    print("\n  2. Computing backward pass...")
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    has_grad = False
    total_grad_norm = 0.0
    for name, param in policy.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            has_grad = True

    total_grad_norm = total_grad_norm ** 0.5

    print(f"     - Total gradient norm: {total_grad_norm:.6f}")
    assert has_grad, "No gradients computed"
    assert total_grad_norm > 0, "Gradient norm is zero"

    print("\n  3. Performing optimizer step...")
    # Record initial weights - skip empty parameters like _dummy_variable
    initial_weight = None
    param_name = None
    for name, param in policy.named_parameters():
        if param.numel() > 0:
            initial_weight = param.clone()
            param_name = name
            break

    assert initial_weight is not None, "No non-empty parameters found"

    optimizer.step()

    # Check weights changed
    final_weight = dict(policy.named_parameters())[param_name]
    weight_diff = (final_weight - initial_weight).abs().max().item()

    print(f"     - Checked parameter: {param_name}")
    print(f"     - Max weight change: {weight_diff:.8f}")
    assert weight_diff > 0, "Weights did not change after optimizer step"

    print("\nGradient flow test passed!")
    return True


def test_full_training_step():
    """Test a complete training step simulation."""
    print("\n" + "=" * 60)
    print("TEST: Full Training Step Simulation")
    print("=" * 60)

    from equi_diffpo.rl_training.maniflow_rollout_collector import ManiFlowRolloutCollector
    from equi_diffpo.rl_training.maniflow_advantage_calculator import (
        ManiFlowAdvantageCalculator,
        AdvantageConfig
    )
    from equi_diffpo.rl_training.rl_utils import masked_mean

    # Create all components
    policy = create_test_policy(device='cpu')
    policy.train()

    env_runner = MockEnvRunner(num_envs=4, max_steps=20)

    collector = ManiFlowRolloutCollector(
        policy=policy,
        env_runner=env_runner,
        max_steps_per_episode=100,
        action_chunk_size=8,
        obs_chunk_size=2,
        device='cpu'
    )

    adv_config = AdvantageConfig(gamma=0.99, gae_lambda=0.95, normalize_advantages=True)
    advantage_calculator = ManiFlowAdvantageCalculator(adv_config)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    print("  Step 1: Collect rollouts")
    runner_results = env_runner.run_rl(policy)
    rollout_batch = collector.collect_rollouts_from_runner_results(runner_results)
    print(f"          Collected {rollout_batch.n_chunk_steps} steps")

    print("\n  Step 2: Calculate advantages")
    next_values = np.zeros((rollout_batch.batch_size, 1), dtype=np.float32)
    rollout_batch = advantage_calculator.calculate_advantages_and_returns(rollout_batch, next_values)
    print(f"          Advantages range: [{rollout_batch.advantages.min():.3f}, {rollout_batch.advantages.max():.3f}]")

    print("\n  Step 3: Convert to torch")
    torch_batch = rollout_batch.to_torch(torch.device('cpu'))

    print("\n  Step 4: Flatten batch for minibatch training")
    # Flatten [n_steps, batch_size, ...] -> [n_steps * batch_size, ...]
    flat_obs = {}
    for key, value in torch_batch['observation'].items():
        flat_obs[key] = value.flatten(0, 1)

    flat_chains = torch_batch['chains'].flatten(0, 1)
    flat_denoise_inds = torch_batch['denoise_inds'].flatten(0, 1)
    flat_advantages = torch_batch['advantages'].flatten(0, 1)
    flat_returns = torch_batch['returns'].flatten(0, 1)
    flat_prev_logprobs = torch_batch['prev_logprobs'].flatten(0, 1)
    flat_loss_mask = torch_batch['loss_mask'].flatten(0, 1) if 'loss_mask' in torch_batch else None

    total_samples = flat_chains.shape[0]
    print(f"          Total samples: {total_samples}")

    print("\n  Step 5: Run minibatch training")
    minibatch_size = min(32, total_samples)
    indices = torch.randperm(total_samples)[:minibatch_size]

    # Extract minibatch
    mini_obs = {key: value[indices] for key, value in flat_obs.items()}
    mini_chains = flat_chains[indices]
    mini_denoise_inds = flat_denoise_inds[indices]
    mini_advantages = flat_advantages[indices]
    mini_returns = flat_returns[indices]
    mini_prev_logprobs = flat_prev_logprobs[indices]
    mini_loss_mask = flat_loss_mask[indices] if flat_loss_mask is not None else torch.ones_like(mini_advantages)

    # Forward pass
    data = {
        'observation': mini_obs,
        'chains': mini_chains,
        'denoise_inds': mini_denoise_inds,
    }

    policy_outputs = policy.default_forward(data, compute_values=True)

    # Compute PPO loss
    new_logprobs = policy_outputs['logprobs']
    values = policy_outputs['values']
    entropy = policy_outputs['entropy']

    old_logprobs_flat = mini_prev_logprobs.sum(dim=-1)
    new_logprobs_flat = new_logprobs.sum(dim=-1)

    ratio = torch.exp(new_logprobs_flat - old_logprobs_flat)
    clipped_ratio = torch.clamp(ratio, 0.8, 1.2)

    policy_loss = -torch.min(mini_advantages * ratio, mini_advantages * clipped_ratio)
    policy_loss = masked_mean(policy_loss, mini_loss_mask.bool())

    value_loss = (values.unsqueeze(1) - mini_returns).pow(2).mean()
    entropy_loss = -entropy.mean()

    total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

    print(f"          Policy loss: {policy_loss.item():.6f}")
    print(f"          Value loss: {value_loss.item():.6f}")
    print(f"          Total loss: {total_loss.item():.6f}")

    print("\n  Step 6: Backward pass and optimizer step")
    optimizer.zero_grad()
    total_loss.backward()

    grad_norm = 0.0
    for param in policy.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)

    optimizer.step()

    print(f"          Gradient norm before clip: {grad_norm:.6f}")
    print(f"          Optimizer step completed!")

    # Verify training worked
    assert torch.isfinite(total_loss), "Loss is not finite"
    assert grad_norm > 0, "No gradients"

    print("\nFull training step simulation test passed!")
    return True


def test_checkpoint_loading():
    """Test loading pretrained weights into RL policy."""
    print("\n" + "=" * 60)
    print("TEST: Checkpoint Loading Simulation")
    print("=" * 60)

    import tempfile
    import os

    # Create two policies
    policy1 = create_test_policy(device='cpu')
    policy2 = create_test_policy(device='cpu')

    # Get initial weights
    with torch.no_grad():
        w1_before = {name: p.clone() for name, p in policy1.named_parameters()}

    # Modify policy1 weights
    with torch.no_grad():
        for param in policy1.parameters():
            param.add_(torch.randn_like(param) * 0.1)

    print("  1. Saving checkpoint...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_policy.ckpt")

        checkpoint = {
            'policy_state_dict': policy1.state_dict(),
            'global_step': 1000,
        }
        torch.save(checkpoint, ckpt_path)
        print(f"     Saved to: {ckpt_path}")

        print("\n  2. Loading checkpoint...")
        loaded_ckpt = torch.load(ckpt_path)
        policy2.load_state_dict(loaded_ckpt['policy_state_dict'])

        print(f"     Global step: {loaded_ckpt['global_step']}")

    # Verify weights match
    print("\n  3. Verifying weights match...")
    with torch.no_grad():
        max_diff = 0.0
        for (name1, p1), (name2, p2) in zip(policy1.named_parameters(), policy2.named_parameters()):
            # Skip empty parameters like _dummy_variable
            if p1.numel() == 0:
                continue
            diff = (p1 - p2).abs().max().item()
            max_diff = max(max_diff, diff)

    print(f"     Max weight difference: {max_diff:.10f}")
    assert max_diff < 1e-6, f"Weights do not match after loading: max diff = {max_diff}"

    print("\nCheckpoint loading test passed!")
    return True


def test_nut_assembly_config():
    """Test loading configuration for nut_assembly_d0 task."""
    print("\n" + "=" * 60)
    print("TEST: Nut Assembly D0 Configuration")
    print("=" * 60)

    # This is the actual shape meta for nut_assembly_d0
    nut_assembly_shape_meta = {
        'obs': {
            'robot0_eye_in_hand_image': {'shape': [3, 84, 84], 'type': 'rgb'},
            'point_cloud': {'shape': [1024, 6], 'type': 'point_cloud'},
            'robot0_eef_pos': {'shape': [3]},
            'robot0_eef_quat': {'shape': [4]},
            'robot0_gripper_qpos': {'shape': [2]},
        },
        'action': {'shape': [10]}
    }

    print("  Creating RL policy with nut_assembly_d0 config...")

    from equi_diffpo.policy.maniflow.maniflow_pointcloud_rl_policy import ManiFlowRLPointcloudPolicy
    from equi_diffpo.model.common.normalizer import LinearNormalizer
    from omegaconf import OmegaConf

    # Create pointcloud_encoder_cfg as OmegaConf object
    # visual_cond_len should equal num_points (DiTX internally multiplies by n_obs_steps)
    visual_cond_len = 256
    n_obs_steps = 2
    pointcloud_encoder_cfg = OmegaConf.create({
        'in_channels': 6,
        'out_channels': 128,
        'use_layernorm': True,
        'final_norm': 'layernorm',
        'normal_channel': False,
        'num_points': visual_cond_len,  # num_points = visual_cond_len
        'pointwise': True,
    })

    # Create policy with nut_assembly config matching the trained model
    policy = ManiFlowRLPointcloudPolicy(
        shape_meta=nut_assembly_shape_meta,
        horizon=16,
        n_action_steps=8,
        n_obs_steps=n_obs_steps,
        num_inference_steps=10,
        obs_as_global_cond=True,
        # Model architecture matching trained model
        n_layer=12,  # Matches trained model
        n_head=8,
        n_emb=768,
        visual_cond_len=visual_cond_len,  # Same as num_points (DiTX multiplies by n_obs_steps internally)
        encoder_output_dim=128,
        # RL-specific parameters
        add_value_head=True,
        noise_method="flow_sde",
        noise_level=0.5,
        noise_anneal=True,
        noise_params=[0.7, 0.3, 400],
        pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        downsample_points=True,  # Enable point cloud downsampling to num_points
    )

    print(f"  - Policy created successfully")
    print(f"  - Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  - Value head: {policy.value_head is not None}")
    print(f"  - Action dim: {policy.action_dim}")
    print(f"  - Horizon: {policy.horizon}")

    # Set up normalizer - use dict fitting pattern
    normalizer = LinearNormalizer()
    obs_data = {}
    for key, meta in nut_assembly_shape_meta['obs'].items():
        obs_data[key] = torch.randn(100, 2, *meta['shape'])
    obs_data['action'] = torch.randn(100, nut_assembly_shape_meta['action']['shape'][0])
    normalizer.fit(obs_data)
    policy.set_normalizer(normalizer)

    # Test forward pass
    batch_size = 2
    obs = {
        'robot0_eye_in_hand_image': torch.randn(batch_size, 2, 3, 84, 84),
        'point_cloud': torch.randn(batch_size, 2, 1024, 6),
        'robot0_eef_pos': torch.randn(batch_size, 2, 3),
        'robot0_eef_quat': torch.randn(batch_size, 2, 4),
        'robot0_gripper_qpos': torch.randn(batch_size, 2, 2),
    }

    policy.eval()
    with torch.no_grad():
        result = policy.predict_action(obs, return_chains=True)

    print(f"\n  Forward pass test:")
    print(f"  - Action shape: {result['action'].shape}")
    print(f"  - Chains shape: {result['chains'].shape}")

    print("\nNut assembly configuration test passed!")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 80)
    print("RUNNING PIPELINE INTEGRATION TESTS")
    print("=" * 80)

    tests = [
        ("Full Rollout Pipeline", test_full_rollout_pipeline),
        ("PPO Loss Computation", test_ppo_loss_computation),
        ("Gradient Flow", test_gradient_flow),
        ("Full Training Step", test_full_training_step),
        ("Checkpoint Loading", test_checkpoint_loading),
        ("Nut Assembly Config", test_nut_assembly_config),
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
    print("TEST SUMMARY: Pipeline Integration")
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
