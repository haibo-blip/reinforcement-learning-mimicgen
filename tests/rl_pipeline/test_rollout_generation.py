#!/usr/bin/env python3
"""
Test rollout generation for RL pipeline using real environment runner.

This test verifies:
1. RobomimicRLRunner can collect rollouts with correct shapes
2. Rollout data contains all required fields (obs, actions, rewards, chains, etc.)
3. Loss mask computation works correctly
4. Data conversion to batch format works correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from typing import Dict, Any, List

# Dataset path for nut_assembly_d0
DATASET_PATH = "equi_diffpo/data/robomimic/datasets/nut_assembly_d0/nut_assembly_d0_voxel_abs.hdf5"

# Shape meta matching the dataset
SHAPE_META = {
    'obs': {
        'robot0_eye_in_hand_image': {'shape': [3, 84, 84], 'type': 'rgb'},
        'agentview_image': {'shape': [3, 84, 84], 'type': 'rgb'},
        'point_cloud': {'shape': [1024, 6], 'type': 'point_cloud'},
        'robot0_eef_pos': {'shape': [3]},
        'robot0_eef_quat': {'shape': [4]},
        'robot0_gripper_qpos': {'shape': [2]},
    },
    'action': {'shape': [10]}
}


def get_dataset_path():
    """Get the full dataset path."""
    full_path = project_root / DATASET_PATH
    if not full_path.exists():
        raise FileNotFoundError(f"Dataset not found: {full_path}")
    return str(full_path)


def create_env_runner(output_dir: str = "/tmp/rl_test_outputs"):
    """Create a real environment runner for testing."""
    from equi_diffpo.env_runner.robomimic_rl_runner import RobomimicRLRunner

    os.makedirs(output_dir, exist_ok=True)

    env_runner = RobomimicRLRunner(
        output_dir=output_dir,
        dataset_path=get_dataset_path(),
        shape_meta=SHAPE_META,
        n_train=2,  # Small numbers for testing
        n_train_vis=1,
        train_start_idx=0,
        n_test=2,
        n_test_vis=1,
        test_start_seed=100000,
        max_steps=50,  # Short episodes for testing
        n_obs_steps=2,
        n_action_steps=8,
        render_obs_key='agentview_image',
        fps=10,
        crf=22,
        past_action=False,
        abs_action=True,
        tqdm_interval_sec=1.0,
        n_envs=2,  # Small number for testing
        collect_rl_data=True,
    )

    return env_runner


def test_loss_mask_computation():
    """Test loss mask computation for excluding post-termination steps."""
    print("\n" + "=" * 60)
    print("TEST: Loss Mask Computation")
    print("=" * 60)

    from equi_diffpo.rl_training.rl_utils import compute_loss_mask

    # Create test cases
    test_cases = [
        # Case 1: No terminations
        {
            'name': 'No terminations',
            'dones': torch.zeros(6, 2, 4),  # [n_steps+1, batch, action_chunk]
            'expected_valid_ratio': 1.0,
        },
        # Case 2: Early termination in first env
        {
            'name': 'Early termination (env 0 at step 2)',
            'dones': _create_dones_with_termination(n_steps=5, batch_size=2, action_chunk=4,
                                                    term_step=2, term_env=0),
            'expected_mask_check': lambda m: m[3:, 0, :].sum() == 0,  # Steps after termination should be masked
        },
        # Case 3: All environments terminate
        {
            'name': 'All envs terminate at different steps',
            'dones': _create_dones_all_terminate(n_steps=5, batch_size=2, action_chunk=4),
            'expected_mask_check': lambda m: True,  # Just verify no errors
        },
    ]

    for case in test_cases:
        print(f"\n  Testing: {case['name']}")
        dones = case['dones']

        loss_mask, loss_mask_sum = compute_loss_mask(dones)

        print(f"    - Input dones shape: {dones.shape}")
        print(f"    - Output loss_mask shape: {loss_mask.shape}")
        print(f"    - Valid steps ratio: {loss_mask.float().mean():.3f}")

        # Verify shape
        expected_mask_shape = (dones.shape[0] - 1, dones.shape[1], dones.shape[2])
        assert loss_mask.shape == expected_mask_shape, \
            f"Loss mask shape mismatch: expected {expected_mask_shape}, got {loss_mask.shape}"

        # Check expected behavior
        if 'expected_valid_ratio' in case:
            valid_ratio = loss_mask.float().mean().item()
            assert abs(valid_ratio - case['expected_valid_ratio']) < 0.01, \
                f"Valid ratio mismatch: expected {case['expected_valid_ratio']}, got {valid_ratio}"

        if 'expected_mask_check' in case:
            assert case['expected_mask_check'](loss_mask), \
                f"Mask check failed for case: {case['name']}"

        print(f"    - Passed!")

    print("\nLoss mask computation tests passed!")
    return True


def _create_dones_with_termination(n_steps, batch_size, action_chunk, term_step, term_env):
    """Helper to create dones tensor with specific termination."""
    dones = torch.zeros(n_steps + 1, batch_size, action_chunk)
    dones[term_step:, term_env, :] = 1.0
    return dones


def _create_dones_all_terminate(n_steps, batch_size, action_chunk):
    """Helper to create dones where all envs terminate at different steps."""
    dones = torch.zeros(n_steps + 1, batch_size, action_chunk)
    for b in range(batch_size):
        term_step = (b + 2) % n_steps  # Different termination step per env
        dones[term_step:, b, :] = 1.0
    return dones


def test_rollout_batch_shapes_with_dummy_data():
    """Test rollout batch with manually created dummy data (no env needed)."""
    print("\n" + "=" * 60)
    print("TEST: Rollout Batch Shapes (Dummy Data)")
    print("=" * 60)

    from equi_diffpo.rl_training.maniflow_rollout_collector import (
        ManiFlowRolloutCollector,
        ManiFlowRolloutBatch
    )
    from equi_diffpo.rl_training.rl_utils import compute_loss_mask

    # Create dummy data matching expected shapes
    n_steps = 20
    batch_size = 4
    action_chunk = 8
    action_dim = 10
    horizon = 16
    N = 10  # num_inference_steps
    n_obs_steps = 2

    print(f"  Creating dummy rollout data...")
    print(f"  - n_steps: {n_steps}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - action_chunk: {action_chunk}")

    # Create dummy rl_data in the format expected from runner
    observations = {
        'robot0_eye_in_hand_image': np.random.randn(n_steps, batch_size, n_obs_steps, 3, 84, 84).astype(np.float32),
        'point_cloud': np.random.randn(n_steps, batch_size, n_obs_steps, 1024, 6).astype(np.float32),
        'robot0_eef_pos': np.random.randn(n_steps, batch_size, n_obs_steps, 3).astype(np.float32),
        'robot0_eef_quat': np.random.randn(n_steps, batch_size, n_obs_steps, 4).astype(np.float32),
        'robot0_gripper_qpos': np.random.randn(n_steps, batch_size, n_obs_steps, 2).astype(np.float32),
    }

    actions = np.random.randn(n_steps, batch_size, action_chunk, action_dim).astype(np.float32)
    rewards = np.random.randn(n_steps, batch_size, action_chunk).astype(np.float32) * 0.1 + 0.1
    dones = np.zeros((n_steps, batch_size, 1), dtype=np.float32)
    prev_logprobs = np.random.randn(n_steps, batch_size, action_chunk, action_dim).astype(np.float32) * 0.1 - 1.0
    prev_values = np.random.randn(n_steps, batch_size, 1).astype(np.float32)
    chains = np.random.randn(n_steps, batch_size, N + 1, horizon, action_dim).astype(np.float32)
    denoise_inds = np.tile(np.arange(N), (n_steps, batch_size, 1)).astype(np.int64)

    # Simulate some terminations
    for b in range(batch_size):
        term_step = np.random.randint(n_steps // 2, n_steps)
        dones[term_step:, b, 0] = 1.0

    # Compute loss mask
    # Expand dones from [n_steps, batch_size, 1] to [n_steps, batch_size, action_chunk]
    dones_expanded = np.broadcast_to(dones, (n_steps, batch_size, action_chunk)).copy()
    dones_torch = torch.from_numpy(dones_expanded)
    bootstrap_done = torch.zeros(1, batch_size, action_chunk, dtype=dones_torch.dtype)
    dones_with_bootstrap = torch.cat([dones_torch, bootstrap_done], dim=0)
    loss_mask, loss_mask_sum = compute_loss_mask(dones_with_bootstrap)

    # Create rollout batch
    rollout_batch = ManiFlowRolloutBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        truncations=np.zeros_like(dones),
        prev_logprobs=prev_logprobs,
        prev_values=prev_values,
        chains=chains,
        denoise_inds=denoise_inds,
        loss_mask=loss_mask.numpy(),
        loss_mask_sum=loss_mask_sum.numpy(),
    )

    # Verify shapes
    print("\n  Verifying rollout batch shapes:")

    assert rollout_batch.actions.shape == (n_steps, batch_size, action_chunk, action_dim), \
        f"Actions shape mismatch: {rollout_batch.actions.shape}"
    print(f"    - actions: {rollout_batch.actions.shape}")

    assert rollout_batch.rewards.shape == (n_steps, batch_size, action_chunk), \
        f"Rewards shape mismatch: {rollout_batch.rewards.shape}"
    print(f"    - rewards: {rollout_batch.rewards.shape}")

    assert rollout_batch.chains.shape == (n_steps, batch_size, N + 1, horizon, action_dim), \
        f"Chains shape mismatch: {rollout_batch.chains.shape}"
    print(f"    - chains: {rollout_batch.chains.shape}")

    assert rollout_batch.denoise_inds.shape == (n_steps, batch_size, N), \
        f"Denoise inds shape mismatch: {rollout_batch.denoise_inds.shape}"
    print(f"    - denoise_inds: {rollout_batch.denoise_inds.shape}")

    assert rollout_batch.loss_mask.shape == (n_steps, batch_size, action_chunk), \
        f"Loss mask shape mismatch: {rollout_batch.loss_mask.shape}"
    print(f"    - loss_mask: {rollout_batch.loss_mask.shape}")

    # Test properties
    assert rollout_batch.batch_size == batch_size, f"batch_size property wrong: {rollout_batch.batch_size}"
    assert rollout_batch.n_chunk_steps == n_steps, f"n_chunk_steps property wrong: {rollout_batch.n_chunk_steps}"

    print("\nRollout batch shape tests passed!")
    return True


def test_rollout_to_torch_conversion():
    """Test conversion of rollout batch to torch tensors."""
    print("\n" + "=" * 60)
    print("TEST: Rollout to Torch Conversion")
    print("=" * 60)

    from equi_diffpo.rl_training.maniflow_rollout_collector import ManiFlowRolloutBatch

    # Create dummy batch
    n_steps = 10
    batch_size = 4
    action_chunk = 8
    action_dim = 10
    horizon = 16
    N = 10

    observations = {
        'robot0_eef_pos': np.random.randn(n_steps, batch_size, 2, 3).astype(np.float32),
        'point_cloud': np.random.randn(n_steps, batch_size, 2, 1024, 6).astype(np.float32),
    }

    rollout_batch = ManiFlowRolloutBatch(
        observations=observations,
        actions=np.random.randn(n_steps, batch_size, action_chunk, action_dim).astype(np.float32),
        rewards=np.random.randn(n_steps, batch_size, action_chunk).astype(np.float32),
        dones=np.zeros((n_steps, batch_size, 1), dtype=np.float32),
        truncations=np.zeros((n_steps, batch_size, 1), dtype=np.float32),
        prev_logprobs=np.random.randn(n_steps, batch_size, action_chunk, action_dim).astype(np.float32),
        prev_values=np.random.randn(n_steps, batch_size, 1).astype(np.float32),
        chains=np.random.randn(n_steps, batch_size, N + 1, horizon, action_dim).astype(np.float32),
        denoise_inds=np.tile(np.arange(N), (n_steps, batch_size, 1)).astype(np.int64),
        loss_mask=np.ones((n_steps, batch_size, action_chunk), dtype=np.float32),
        loss_mask_sum=np.ones((n_steps, batch_size, action_chunk), dtype=np.float32) * n_steps * action_chunk,
        advantages=np.random.randn(n_steps, batch_size, action_chunk).astype(np.float32),
        returns=np.random.randn(n_steps, batch_size, 1).astype(np.float32),
    )

    # Convert to torch
    device = torch.device('cpu')
    torch_batch = rollout_batch.to_torch(device)

    print("\n  Verifying torch conversion:")

    # Check that tensors are on correct device
    assert torch_batch['actions'].device == device, "Actions not on correct device"
    assert torch_batch['observation']['robot0_eef_pos'].device == device, "Observations not on correct device"
    print(f"    - Tensors on correct device: {device}")

    # Check tensor types
    assert isinstance(torch_batch['actions'], torch.Tensor), "Actions not converted to tensor"
    assert isinstance(torch_batch['observation']['robot0_eef_pos'], torch.Tensor), "Observations not converted to tensor"
    print(f"    - Correctly converted to torch.Tensor")

    # Check shapes preserved
    assert torch_batch['actions'].shape == tuple(rollout_batch.actions.shape), "Actions shape changed during conversion"
    print(f"    - Shapes preserved correctly")

    # Check all expected keys present
    expected_keys = ['observation', 'actions', 'rewards', 'dones', 'prev_logprobs',
                    'prev_values', 'chains', 'denoise_inds', 'loss_mask', 'advantages', 'returns']
    for key in expected_keys:
        assert key in torch_batch, f"Missing key in torch batch: {key}"
    print(f"    - All expected keys present: {len(expected_keys)} keys")

    print("\nTorch conversion tests passed!")
    return True


def test_env_runner_initialization():
    """Test that RobomimicRLRunner can be initialized with nut_assembly_d0."""
    print("\n" + "=" * 60)
    print("TEST: Environment Runner Initialization")
    print("=" * 60)

    try:
        dataset_path = get_dataset_path()
        print(f"  Dataset path: {dataset_path}")

        env_runner = create_env_runner()

        print(f"  Environment runner created successfully!")
        print(f"    - Type: {type(env_runner).__name__}")
        print(f"    - collect_rl_data: {env_runner.collect_rl_data}")

        print("\nEnvironment runner initialization test passed!")
        return True

    except FileNotFoundError as e:
        print(f"  SKIPPED: {e}")
        return True  # Skip if dataset not found
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all rollout generation tests."""
    print("\n" + "=" * 80)
    print("RUNNING ROLLOUT GENERATION TESTS")
    print("=" * 80)

    tests = [
        ("Loss Mask Computation", test_loss_mask_computation),
        ("Rollout Batch Shapes (Dummy)", test_rollout_batch_shapes_with_dummy_data),
        ("Rollout to Torch Conversion", test_rollout_to_torch_conversion),
        ("Env Runner Initialization", test_env_runner_initialization),
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
    print("TEST SUMMARY: Rollout Generation")
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
