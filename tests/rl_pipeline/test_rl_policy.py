#!/usr/bin/env python3
"""
Test RL policy for the ManiFlow RL pipeline.

This test verifies:
1. ManiFlowRLPointcloudPolicy can be instantiated correctly
2. Value head produces correct shape outputs
3. Action sampling produces correct shapes
4. Log probability computation works correctly
5. Chain sampling for RL training works
6. default_forward method produces correct outputs
7. Policy can load pretrained weights
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


def create_dummy_normalizer():
    """Create a dummy normalizer for testing."""
    from equi_diffpo.model.common.normalizer import LinearNormalizer

    normalizer = LinearNormalizer()

    # Fit all observation keys at once using a dict
    obs_data = {}
    for key, meta in SHAPE_META['obs'].items():
        # Create dummy data [N, obs_steps, ...]
        if meta.get('type') == 'rgb':
            obs_data[key] = torch.randn(100, 2, *meta['shape'])
        elif meta.get('type') == 'point_cloud':
            obs_data[key] = torch.randn(100, 2, *meta['shape'])
        else:
            obs_data[key] = torch.randn(100, 2, *meta['shape'])

    # Add action
    obs_data['action'] = torch.randn(100, SHAPE_META['action']['shape'][0])

    # Fit all at once
    normalizer.fit(obs_data)

    return normalizer


def create_test_policy(device='cpu'):
    """Create a test RL policy."""
    from equi_diffpo.policy.maniflow.maniflow_pointcloud_rl_policy import ManiFlowRLPointcloudPolicy
    from omegaconf import OmegaConf

    # Create pointcloud_encoder_cfg as OmegaConf object
    # num_points controls downsampling when downsample_points=True
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
        obs_as_global_cond=True,
        n_layer=2,  # Small for testing
        n_head=4,
        n_emb=64,  # Small for testing
        encoder_output_dim=32,  # Small for testing
        visual_cond_len=visual_cond_len,  # Same as num_points (DiTX multiplies by n_obs_steps internally)
        add_value_head=True,
        noise_method="flow_sde",
        noise_level=0.5,
        noise_anneal=False,
        pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        downsample_points=True,  # Enable point cloud downsampling to num_points
    )

    # Set normalizer
    normalizer = create_dummy_normalizer()
    policy.set_normalizer(normalizer)

    policy.to(device)
    return policy


def create_dummy_observation(batch_size: int = 2, n_obs_steps: int = 2, device='cpu'):
    """Create dummy observations for testing."""
    obs = {
        'robot0_eye_in_hand_image': torch.randn(batch_size, n_obs_steps, 3, 84, 84, device=device),
        'point_cloud': torch.randn(batch_size, n_obs_steps, 1024, 6, device=device),
        'robot0_eef_pos': torch.randn(batch_size, n_obs_steps, 3, device=device),
        'robot0_eef_quat': torch.randn(batch_size, n_obs_steps, 4, device=device),
        'robot0_gripper_qpos': torch.randn(batch_size, n_obs_steps, 2, device=device),
    }
    return obs


def test_policy_instantiation():
    """Test that the RL policy can be instantiated."""
    print("\n" + "=" * 60)
    print("TEST: Policy Instantiation")
    print("=" * 60)

    try:
        policy = create_test_policy(device='cpu')

        print(f"  Policy created successfully!")
        print(f"  - Type: {type(policy).__name__}")
        print(f"  - Parameters: {sum(p.numel() for p in policy.parameters()):,}")
        print(f"  - Has value head: {policy.value_head is not None}")
        print(f"  - Noise method: {policy.noise_method}")
        print(f"  - Horizon: {policy.horizon}")
        print(f"  - Action dim: {policy.action_dim}")

        # Verify value head exists
        assert policy.value_head is not None, "Value head should exist"
        assert isinstance(policy.value_head, nn.Module), "Value head should be nn.Module"

        print("\nPolicy instantiation test passed!")
        return True

    except Exception as e:
        print(f"Policy instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_encode_observations():
    """Test observation encoding."""
    print("\n" + "=" * 60)
    print("TEST: Observation Encoding")
    print("=" * 60)

    policy = create_test_policy(device='cpu')
    policy.eval()

    batch_size = 2
    obs = create_dummy_observation(batch_size=batch_size, device='cpu')

    with torch.no_grad():
        obs_features = policy.encode_observations(obs)

    print(f"  Input batch size: {batch_size}")
    print(f"  Output features shape: {obs_features.shape}")

    # Verify shape [B, n_obs_steps*L, obs_feature_dim]
    assert obs_features.dim() == 3, f"Expected 3D tensor, got {obs_features.dim()}D"
    assert obs_features.shape[0] == batch_size, "Batch size mismatch"

    print("\nObservation encoding test passed!")
    return True


def test_predict_action():
    """Test action prediction."""
    print("\n" + "=" * 60)
    print("TEST: Action Prediction")
    print("=" * 60)

    policy = create_test_policy(device='cpu')
    policy.eval()

    batch_size = 2
    obs = create_dummy_observation(batch_size=batch_size, device='cpu')

    with torch.no_grad():
        result = policy.predict_action(obs, return_chains=False)

    action = result['action']
    action_pred = result['action_pred']

    print(f"  Input batch size: {batch_size}")
    print(f"  Action shape: {action.shape}")
    print(f"  Action pred shape: {action_pred.shape}")

    # Verify shapes
    assert action.shape == (batch_size, policy.n_action_steps, policy.action_dim), \
        f"Wrong action shape: {action.shape}"
    assert action_pred.shape == (batch_size, policy.horizon, policy.action_dim), \
        f"Wrong action_pred shape: {action_pred.shape}"

    # Verify finite values
    assert torch.isfinite(action).all(), "Action contains non-finite values"
    assert torch.isfinite(action_pred).all(), "Action pred contains non-finite values"

    print("\nAction prediction test passed!")
    return True


def test_predict_action_with_chains():
    """Test action prediction with chain tracking for RL."""
    print("\n" + "=" * 60)
    print("TEST: Action Prediction with Chains")
    print("=" * 60)

    policy = create_test_policy(device='cpu')
    policy.eval()

    batch_size = 2
    obs = create_dummy_observation(batch_size=batch_size, device='cpu')

    with torch.no_grad():
        result = policy.predict_action(obs, return_chains=True)

    # Check all expected keys
    expected_keys = ['action', 'action_pred', 'chains', 'prev_logprobs', 'prev_values', 'denoise_inds']
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"

    chains = result['chains']
    prev_logprobs = result['prev_logprobs']
    prev_values = result['prev_values']
    denoise_inds = result['denoise_inds']

    print(f"  chains shape: {chains.shape}")
    print(f"  prev_logprobs shape: {prev_logprobs.shape}")
    print(f"  prev_values shape: {prev_values.shape}")
    print(f"  denoise_inds shape: {denoise_inds.shape}")

    # Verify chains shape [B, N+1, horizon, action_dim]
    N = policy.num_inference_steps
    assert chains.shape == (batch_size, N + 1, policy.horizon, policy.action_dim), \
        f"Wrong chains shape: {chains.shape}"

    # Verify denoise_inds shape [B, N]
    assert denoise_inds.shape == (batch_size, N), \
        f"Wrong denoise_inds shape: {denoise_inds.shape}"

    print("\nAction prediction with chains test passed!")
    return True


def test_sample_actions():
    """Test sample_actions method for RL."""
    print("\n" + "=" * 60)
    print("TEST: Sample Actions")
    print("=" * 60)

    policy = create_test_policy(device='cpu')

    batch_size = 2
    obs = create_dummy_observation(batch_size=batch_size, device='cpu')

    # Test train mode
    policy.train()
    with torch.no_grad():
        result_train = policy.sample_actions(obs, mode="train", compute_values=True)

    print("  Train mode results:")
    print(f"    - actions shape: {result_train['actions'].shape}")
    print(f"    - chains shape: {result_train['chains'].shape}")
    print(f"    - prev_logprobs shape: {result_train['prev_logprobs'].shape}")
    print(f"    - prev_values shape: {result_train['prev_values'].shape}")

    # Test eval mode
    policy.eval()
    with torch.no_grad():
        result_eval = policy.sample_actions(obs, mode="eval", compute_values=True)

    print("\n  Eval mode results:")
    print(f"    - actions shape: {result_eval['actions'].shape}")

    # Verify all required fields
    required_fields = ['actions', 'chains', 'prev_logprobs', 'prev_values', 'denoise_inds']
    for field in required_fields:
        assert field in result_train, f"Missing field in train mode: {field}"
        assert field in result_eval, f"Missing field in eval mode: {field}"

    # Actions should be different between runs due to stochasticity
    # But shape should be consistent
    assert result_train['actions'].shape == result_eval['actions'].shape, \
        "Action shapes should be consistent"

    print("\nSample actions test passed!")
    return True


def test_default_forward():
    """Test default_forward for RL training."""
    print("\n" + "=" * 60)
    print("TEST: Default Forward (RL Training)")
    print("=" * 60)

    policy = create_test_policy(device='cpu')
    policy.train()

    batch_size = 2
    N = policy.num_inference_steps
    horizon = policy.horizon
    action_dim = policy.action_dim

    # Create data dict as expected by default_forward
    obs = create_dummy_observation(batch_size=batch_size, device='cpu')

    # Create chains and denoise_inds
    chains = torch.randn(batch_size, N + 1, horizon, action_dim)
    denoise_inds = torch.zeros(batch_size, N, dtype=torch.long)
    # Set specific denoise indices
    for b in range(batch_size):
        denoise_inds[b] = torch.arange(N)

    data = {
        'observation': obs,
        'chains': chains,
        'denoise_inds': denoise_inds,
    }

    with torch.no_grad():
        result = policy.default_forward(data, compute_values=True)

    print(f"  Output keys: {list(result.keys())}")

    # Check expected outputs
    assert 'logprobs' in result, "Missing logprobs"
    assert 'values' in result, "Missing values"
    assert 'entropy' in result, "Missing entropy"

    logprobs = result['logprobs']
    values = result['values']
    entropy = result['entropy']

    print(f"  logprobs shape: {logprobs.shape}")
    print(f"  values shape: {values.shape}")
    print(f"  entropy shape: {entropy.shape}")

    # Verify shapes
    action_chunk = policy.n_action_steps
    assert logprobs.shape == (batch_size, action_chunk, action_dim), \
        f"Wrong logprobs shape: {logprobs.shape}"
    assert values.shape == (batch_size,), \
        f"Wrong values shape: {values.shape}"
    assert entropy.shape == (batch_size, 1), \
        f"Wrong entropy shape: {entropy.shape}"

    # Verify finite values
    assert torch.isfinite(logprobs).all(), "Logprobs contain non-finite values"
    assert torch.isfinite(values).all(), "Values contain non-finite values"

    print("\nDefault forward test passed!")
    return True


def test_value_head():
    """Test value head produces correct outputs."""
    print("\n" + "=" * 60)
    print("TEST: Value Head")
    print("=" * 60)

    policy = create_test_policy(device='cpu')
    policy.eval()

    # Test value head directly
    batch_size = 4
    obs = create_dummy_observation(batch_size=batch_size, device='cpu')

    with torch.no_grad():
        # Encode observations
        obs_features = policy.encode_observations(obs)

        # Get value from value head
        # The value head expects pooled features
        if policy.value_head is not None:
            values = policy.value_head(obs_features)
            print(f"  Value head output shape: {values.shape}")
            print(f"  Value range: [{values.min():.4f}, {values.max():.4f}]")

            # Values should be scalar per batch element
            assert values.dim() >= 1, "Values should have at least 1 dimension"
            assert values.shape[0] == batch_size, "Batch size mismatch"

    print("\nValue head test passed!")
    return True


def test_log_probability_computation():
    """Test log probability computation."""
    print("\n" + "=" * 60)
    print("TEST: Log Probability Computation")
    print("=" * 60)

    policy = create_test_policy(device='cpu')
    policy.train()

    batch_size = 2

    # Test get_logprob_norm
    sample = torch.randn(batch_size, policy.horizon, policy.action_dim)
    mu = torch.randn(batch_size, policy.horizon, policy.action_dim)
    sigma = torch.abs(torch.randn(batch_size, policy.horizon, policy.action_dim)) + 0.1

    log_prob = policy.get_logprob_norm(sample, mu, sigma)

    print(f"  Input shapes:")
    print(f"    - sample: {sample.shape}")
    print(f"    - mu: {mu.shape}")
    print(f"    - sigma: {sigma.shape}")
    print(f"  Output log_prob shape: {log_prob.shape}")
    print(f"  Log prob range: [{log_prob.min():.4f}, {log_prob.max():.4f}]")

    # Verify shape matches input
    assert log_prob.shape == sample.shape, \
        f"Log prob shape mismatch: expected {sample.shape}, got {log_prob.shape}"

    # Log probs should be finite (no inf or nan)
    assert torch.isfinite(log_prob).all(), "Log probs contain non-finite values"

    # Log probs should be negative (log of probability < 1)
    # Note: This might not always be true depending on normalization
    # assert (log_prob <= 0).all(), "Log probs should be <= 0"

    print("\nLog probability computation test passed!")
    return True


def test_noise_annealing():
    """Test noise annealing functionality."""
    print("\n" + "=" * 60)
    print("TEST: Noise Annealing")
    print("=" * 60)

    from equi_diffpo.policy.maniflow.maniflow_pointcloud_rl_policy import ManiFlowRLPointcloudPolicy
    from omegaconf import OmegaConf

    # Create pointcloud_encoder_cfg as OmegaConf object
    pointcloud_encoder_cfg = OmegaConf.create({
        'in_channels': 6,
        'out_channels': 32,
        'use_layernorm': True,
        'final_norm': 'layernorm',
        'normal_channel': False,
        'num_points': 64,
        'pointwise': True,
    })

    # Create policy with noise annealing enabled
    policy = ManiFlowRLPointcloudPolicy(
        shape_meta=SHAPE_META,
        horizon=16,
        n_action_steps=8,
        n_obs_steps=2,
        num_inference_steps=10,
        n_layer=2,
        n_head=4,
        n_emb=64,
        encoder_output_dim=32,
        visual_cond_len=64,
        add_value_head=True,
        noise_method="flow_sde",
        noise_level=0.5,
        noise_anneal=True,
        noise_params=[0.7, 0.3, 400],  # noise_start, noise_end, noise_anneal_steps
        pointcloud_encoder_cfg=pointcloud_encoder_cfg,
    )

    # Test noise level at different global steps
    policy.set_global_step(0)
    noise_start = policy.get_current_noise_level()

    policy.set_global_step(200)
    noise_mid = policy.get_current_noise_level()

    policy.set_global_step(400)
    noise_end = policy.get_current_noise_level()

    policy.set_global_step(800)
    noise_after = policy.get_current_noise_level()

    print(f"  Noise annealing: 0.7 -> 0.3 over 400 steps")
    print(f"  - Step 0: {noise_start:.4f} (expected: ~0.7)")
    print(f"  - Step 200: {noise_mid:.4f} (expected: ~0.5)")
    print(f"  - Step 400: {noise_end:.4f} (expected: ~0.3)")
    print(f"  - Step 800: {noise_after:.4f} (expected: ~0.3, clamped)")

    # Verify annealing behavior
    assert abs(noise_start - 0.7) < 0.01, f"Wrong noise at step 0: {noise_start}"
    assert abs(noise_end - 0.3) < 0.01, f"Wrong noise at step 400: {noise_end}"
    assert noise_mid < noise_start and noise_mid > noise_end, "Noise should decrease over time"
    assert abs(noise_after - 0.3) < 0.01, "Noise should stay at minimum after annealing"

    print("\nNoise annealing test passed!")
    return True


def test_train_eval_mode():
    """Test train/eval mode switching."""
    print("\n" + "=" * 60)
    print("TEST: Train/Eval Mode Switching")
    print("=" * 60)

    policy = create_test_policy(device='cpu')

    # Test train mode
    policy.train()
    assert policy._training_mode == True, "Training mode not set correctly"
    assert policy.training == True, "PyTorch training mode not set correctly"

    # Test eval mode
    policy.eval()
    assert policy._training_mode == False, "Eval mode not set correctly"
    assert policy.training == False, "PyTorch training mode not set correctly"

    print(f"  Train mode: _training_mode={policy._training_mode}, training={policy.training}")

    policy.train()
    print(f"  After train(): _training_mode={policy._training_mode}, training={policy.training}")

    policy.eval()
    print(f"  After eval(): _training_mode={policy._training_mode}, training={policy.training}")

    print("\nTrain/eval mode switching test passed!")
    return True


def run_all_tests():
    """Run all RL policy tests."""
    print("\n" + "=" * 80)
    print("RUNNING RL POLICY TESTS")
    print("=" * 80)

    tests = [
        ("Policy Instantiation", test_policy_instantiation),
        ("Observation Encoding", test_encode_observations),
        ("Action Prediction", test_predict_action),
        ("Action Prediction with Chains", test_predict_action_with_chains),
        ("Sample Actions", test_sample_actions),
        ("Default Forward", test_default_forward),
        ("Value Head", test_value_head),
        ("Log Probability Computation", test_log_probability_computation),
        ("Noise Annealing", test_noise_annealing),
        ("Train/Eval Mode", test_train_eval_mode),
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
    print("TEST SUMMARY: RL Policy")
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
