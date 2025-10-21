#!/usr/bin/env python3
"""
Quick test script to verify ManiFlow integration in equidiff.
Run this after setup_maniflow_policy.sh completes.
"""

import sys

def test_basic_imports():
    """Test basic imports"""
    print("=" * 60)
    print("Testing ManiFlow Integration")
    print("=" * 60)
    print()

    print("Step 1: Testing core imports...")
    print("-" * 60)

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        import timm
        print(f"✓ timm {timm.__version__}")
    except Exception as e:
        print(f"✗ timm import failed: {e}")
        return False

    try:
        from equi_diffpo.model.common.normalizer import LinearNormalizer
        print("✓ LinearNormalizer")
    except Exception as e:
        print(f"✗ LinearNormalizer import failed: {e}")
        return False

    try:
        from equi_diffpo.model.diffusion.ditx import DiTX
        print("✓ DiTX model")
    except Exception as e:
        print(f"✗ DiTX import failed: {e}")
        return False

    try:
        from equi_diffpo.policy.maniflow.maniflow_image_policy import ManiFlowTransformerImagePolicy
        print("✓ ManiFlowTransformerImagePolicy")
    except Exception as e:
        print(f"✗ ManiFlowTransformerImagePolicy import failed: {e}")
        return False

    try:
        from equi_diffpo.policy.maniflow.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy
        print("✓ ManiFlowTransformerPointcloudPolicy")
    except Exception as e:
        print(f"✗ ManiFlowTransformerPointcloudPolicy import failed: {e}")
        return False

    print()
    return True


def test_policy_creation():
    """Test creating policies"""
    print("Step 2: Testing policy creation...")
    print("-" * 60)

    try:
        from equi_diffpo.policy.maniflow.maniflow_image_policy import ManiFlowTransformerImagePolicy
        from equi_diffpo.model.vision_2d.timm_obs_encoder import TimmObsEncoder
        import torch

        # Define minimal shape_meta
        shape_meta = {
            'obs': {
                'agentview_image': {
                    'shape': [3, 96, 96],  # Must be divisible by downsample_ratio (32)
                    'type': 'rgb',
                    'horizon': 2  # n_obs_steps
                }
            },
            'action': {
                'shape': [7],
                'horizon': 16  # horizon
            }
        }

        # Create obs encoder
        obs_encoder = TimmObsEncoder(
            shape_meta=shape_meta,
            model_name='resnet34',
            pretrained=False,  # Don't download weights for test
            frozen=False,
            global_pool='',
            transforms=None,  # Required parameter
            downsample_ratio=32
        )
        print("✓ Created TimmObsEncoder")

        # Create policy
        policy = ManiFlowTransformerImagePolicy(
            shape_meta=shape_meta,
            obs_encoder=obs_encoder,
            horizon=16,
            n_action_steps=8,
            n_obs_steps=2,
            num_inference_steps=10,
            obs_as_global_cond=True,
            n_layer=3,  # Small for test
            n_head=4,
            n_emb=256,
            visual_cond_len=256
        )

        # Initialize normalizer with dummy data for testing
        from equi_diffpo.model.common.normalizer import SingleFieldLinearNormalizer

        dummy_obs_data = {
            'agentview_image': torch.randn(10, 2, 3, 96, 96)  # (N_samples, T, C, H, W)
        }
        dummy_action_data = torch.randn(10, 16, 7)  # (N_samples, horizon, action_dim)

        policy.normalizer.fit(dummy_obs_data, mode='limits', output_max=1.0, output_min=-1.0)

        # Create action normalizer using create_fit
        action_normalizer = SingleFieldLinearNormalizer.create_fit(
            dummy_action_data, mode='limits', output_max=1.0, output_min=-1.0
        )
        policy.normalizer['action'] = action_normalizer

        print("✓ Created ManiFlowTransformerImagePolicy")

        param_count = sum(p.numel() for p in policy.parameters())
        print(f"  └─ Parameters: {param_count:,}")

        return True

    except Exception as e:
        print(f"✗ Policy creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass with dummy data"""
    print()
    print("Step 3: Testing forward pass...")
    print("-" * 60)

    try:
        from equi_diffpo.policy.maniflow.maniflow_image_policy import ManiFlowTransformerImagePolicy
        from equi_diffpo.model.vision_2d.timm_obs_encoder import TimmObsEncoder
        import torch

        shape_meta = {
            'obs': {
                'agentview_image': {
                    'shape': [3, 96, 96],  # Must be divisible by downsample_ratio (32)
                    'type': 'rgb',
                    'horizon': 2  # n_obs_steps
                }
            },
            'action': {
                'shape': [7],
                'horizon': 16  # horizon
            }
        }

        obs_encoder = TimmObsEncoder(
            shape_meta=shape_meta,
            model_name='resnet34',
            pretrained=False,
            frozen=False,
            global_pool='',
            transforms=None,  # Required parameter
            downsample_ratio=32
        )

        policy = ManiFlowTransformerImagePolicy(
            shape_meta=shape_meta,
            obs_encoder=obs_encoder,
            horizon=16,
            n_action_steps=8,
            n_obs_steps=2,
            num_inference_steps=10,
            obs_as_global_cond=True,
            n_layer=3,
            n_head=4,
            n_emb=256,
            visual_cond_len=256
        )

        # Initialize normalizer with dummy data for testing
        from equi_diffpo.model.common.normalizer import SingleFieldLinearNormalizer

        dummy_obs_data = {
            'agentview_image': torch.randn(10, 2, 3, 96, 96)  # (N_samples, T, C, H, W)
        }
        dummy_action_data = torch.randn(10, 16, 7)  # (N_samples, horizon, action_dim)

        policy.normalizer.fit(dummy_obs_data, mode='limits', output_max=1.0, output_min=-1.0)

        # Create action normalizer using create_fit
        action_normalizer = SingleFieldLinearNormalizer.create_fit(
            dummy_action_data, mode='limits', output_max=1.0, output_min=-1.0
        )
        policy.normalizer['action'] = action_normalizer

        # Create dummy observation
        obs_dict = {
            'agentview_image': torch.randn(1, 2, 3, 96, 96)  # B, T, C, H, W
        }

        # Test inference
        policy.eval()
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)
            action = action_dict['action']

        print(f"✓ Forward pass successful")
        print(f"  └─ Input shape: {obs_dict['agentview_image'].shape}")
        print(f"  └─ Output shape: {action.shape}")
        print(f"  └─ Expected: (1, {policy.n_action_steps}, 7)")

        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print()

    results = []

    # Test 1: Imports
    results.append(("Basic Imports", test_basic_imports()))

    # Test 2: Policy Creation
    if results[0][1]:
        results.append(("Policy Creation", test_policy_creation()))
    else:
        print("\nSkipping policy creation test due to import failures")
        results.append(("Policy Creation", False))

    # Test 3: Forward Pass
    if results[1][1]:
        results.append(("Forward Pass", test_forward_pass()))
    else:
        print("\nSkipping forward pass test due to policy creation failure")
        results.append(("Forward Pass", False))

    # Summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    print()
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("🎉 All tests passed! ManiFlow integration is working.")
        print()
        print("Next steps:")
        print("1. Load a pre-trained ManiFlow checkpoint (if available)")
        print("2. Adapt robomimic_image_runner.py to use ManiFlow policies")
        print("3. Run evaluation on Robomimic tasks")
        return 0
    else:
        print("⚠️  Some tests failed. Check errors above.")
        print()
        print("Common issues:")
        print("- Missing packages: pip install timm einops")
        print("- Wrong environment: conda activate equidiff")
        print("- Import paths: check that files were copied correctly")
        return 1


if __name__ == "__main__":
    sys.exit(main())
