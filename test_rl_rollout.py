#!/usr/bin/env python3

import os
import sys
import hydra
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Register hydra resolvers (from train.py)
max_steps = {
    'stack_d1': 400,
    'stack_three_d1': 400,
    'square_d2': 400,
    'threading_d2': 400,
    'coffee_d2': 400,
    'three_piece_assembly_d2': 500,
    'hammer_cleanup_d1': 500,
    'mug_cleanup_d1': 500,
    'kitchen_d1': 800,
    'nut_assembly_d0': 500,
    'pick_place_d0': 1000,
    'coffee_preparation_d1': 800,
    'tool_hang': 700,
    'can': 400,
    'lift': 400,
    'square': 400,
}

def get_ws_x_center(task_name):
    if task_name.startswith('kitchen_') or task_name.startswith('hammer_cleanup_'):
        return -0.2
    else:
        return 0.

def get_ws_y_center(task_name):
    return 0.

OmegaConf.register_new_resolver("get_max_steps", lambda x: max_steps[x], replace=True)
OmegaConf.register_new_resolver("get_ws_x_center", get_ws_x_center, replace=True)
OmegaConf.register_new_resolver("get_ws_y_center", get_ws_y_center, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)

from equi_diffpo.rl_training.rl_workspace import RLTrainingWorkspace


@hydra.main(version_base=None, config_path="equi_diffpo/config", config_name="train_maniflow_pointcloud_rl")
def test_rl_rollout(cfg: DictConfig):
    """Test RL rollout collection with ManiFlow policy."""

    print("=" * 60)
    print("Testing RL Rollout Collection")
    print("=" * 60)

    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    try:
        # Create workspace
        print("\n1. Creating RL Training Workspace...")
        workspace = RLTrainingWorkspace(cfg)
        print("   ✓ Workspace created successfully")

        # Test rollout collection using RL runner
        print("\n2. Testing Rollout Collection...")
        rollout_metrics, episode_data = workspace.rl_runner.run_rl_collection(workspace.policy)

        print("   ✓ Rollout collection completed")
        print(f"   - Collected {len(episode_data['episode_lengths'])} episodes")
        print(f"   - Episode length range: {min(episode_data['episode_lengths'])}-{max(episode_data['episode_lengths'])}")

        mean_reward = rollout_metrics.get('train/mean_score', rollout_metrics.get('test/mean_score', 0.0))
        print(f"   - Mean episode reward: {mean_reward:.3f}")

        # Convert to batch format and print shapes
        print("\n3. Converting to Batch Format...")
        batch_data = workspace._convert_episode_data_to_batch(episode_data)

        print("   - Batch Data Shapes:")
        print(f"     Observations:")
        for key, value in batch_data['obs'].items():
            print(f"       {key}: {value.shape}")
        print(f"     Actions: {batch_data['actions'].shape}")
        print(f"     Fixed Noise: {batch_data['fixed_noise'].shape}")
        print(f"     Rewards: {batch_data['rewards'].shape}")
        print(f"     Dones: {batch_data['dones'].shape}")
        print(f"     Episode lengths: {batch_data['episode_lengths'].shape}")

        # Test PPO trainer (without actual training)
        print("\n4. Testing PPO Trainer Setup...")
        ppo_trainer = workspace.ppo_trainer
        print(f"   ✓ PPO trainer initialized")
        print(f"   - Policy parameters: {sum(p.numel() for p in ppo_trainer.policy.parameters()):,}")
        print(f"   - Critic parameters: {sum(p.numel() for p in ppo_trainer.critic.parameters()):,}")

        # Test GAE computation
        print("\n5. Testing GAE Computation...")
        with torch.no_grad():
            # Get values for GAE computation
            batch_size, max_length = batch_data['rewards'].shape
            obs_dict = batch_data['obs']

            # Move to device and reshape for value network
            obs_dict = {k: v.to(workspace.device) for k, v in obs_dict.items()}
            flat_obs_dict = {}
            for key, value in obs_dict.items():
                flat_obs_dict[key] = value.reshape(batch_size * max_length, *value.shape[2:])

            flat_values = ppo_trainer.critic(flat_obs_dict).squeeze(-1)
            values = flat_values.reshape(batch_size, max_length)

            # Compute GAE
            advantages, returns = ppo_trainer.compute_gae(
                batch_data['rewards'].to(workspace.device),
                values,
                batch_data['dones'].to(workspace.device),
                batch_data['episode_lengths'].to(workspace.device)
            )

            print(f"   ✓ GAE computation successful")
            print(f"   - Advantages shape: {advantages.shape}")
            print(f"   - Returns shape: {returns.shape}")
            print(f"   - Advantages mean/std: {advantages.mean().item():.3f} / {advantages.std().item():.3f}")
            print(f"   - Returns mean/std: {returns.mean().item():.3f} / {returns.std().item():.3f}")

        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)

        # Print collector stats
        stats = workspace.rollout_collector.get_stats()
        print(f"\nCollector Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


@hydra.main(version_base=None, config_path="equi_diffpo/config", config_name="train_maniflow_pointcloud_rl")
def test_rl_training(cfg: DictConfig):
    """Test full RL training loop (1 epoch)."""

    print("=" * 60)
    print("Testing RL Training Loop")
    print("=" * 60)

    try:
        # Modify config for quick test
        cfg.rl_training.num_epochs = 1
        cfg.rl_training.rollout_collector.n_episodes_per_batch = 4
        cfg.rl_training.rollout_collector.n_envs = 4
        cfg.rl_training.rollout_collector.render_episodes = 2
        cfg.rl_training.ppo_trainer.ppo_epochs = 1
        cfg.rl_training.ppo_trainer.mini_batch_size = 16
        cfg.logging.mode = "disabled"  # Disable wandb for test

        print("\nModified config for quick test:")
        print(f"  - num_epochs: {cfg.rl_training.num_epochs}")
        print(f"  - n_episodes_per_batch: {cfg.rl_training.rollout_collector.n_episodes_per_batch}")
        print(f"  - n_envs: {cfg.rl_training.rollout_collector.n_envs}")
        print(f"  - ppo_epochs: {cfg.rl_training.ppo_trainer.ppo_epochs}")

        # Create workspace
        print("\n1. Creating RL Training Workspace...")
        workspace = RLTrainingWorkspace(cfg)
        print("   ✓ Workspace created successfully")

        # Run training
        print("\n2. Running RL Training (1 epoch)...")
        workspace.run()

        print("\n" + "=" * 60)
        print("✅ RL Training test completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Training test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("Available tests:")
    print("1. test_rl_rollout - Test rollout collection only")
    print("2. test_rl_training - Test full training loop (1 epoch)")

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "training":
        # Remove the training argument before passing to hydra
        sys.argv = sys.argv[:1] + sys.argv[2:]
        test_rl_training()
    else:
        # Default to rollout test
        if len(sys.argv) > 1 and sys.argv[1] == "rollout":
            sys.argv = sys.argv[:1] + sys.argv[2:]
        test_rl_rollout()