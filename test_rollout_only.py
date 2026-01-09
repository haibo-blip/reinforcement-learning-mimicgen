#!/usr/bin/env python3

import os
import sys
import torch
from pathlib import Path
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from equi_diffpo.rl_training.minimal_dummy_policy import MinimalDummyPolicy


class MockEnvironment:
    """Minimal mock environment that only tests rollout collection logic."""

    def __init__(self, n_envs: int = 4, max_steps: int = 50):
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self):
        """Return random initial observations."""
        self.step_count = 0
        return {
            'point_cloud': np.random.randn(self.n_envs, 1024, 6).astype(np.float32),
            'robot0_eef_pos': np.random.randn(self.n_envs, 3).astype(np.float32),
            'robot0_eef_quat': np.random.randn(self.n_envs, 4).astype(np.float32),
            'robot0_gripper_qpos': np.random.randn(self.n_envs, 2).astype(np.float32),
        }

    def step(self, actions):
        """Step environment and return next observations, rewards, dones."""
        self.step_count += 1

        # Next observations
        obs = {
            'point_cloud': np.random.randn(self.n_envs, 1024, 6).astype(np.float32),
            'robot0_eef_pos': np.random.randn(self.n_envs, 3).astype(np.float32),
            'robot0_eef_quat': np.random.randn(self.n_envs, 4).astype(np.float32),
            'robot0_gripper_qpos': np.random.randn(self.n_envs, 2).astype(np.float32),
        }

        # Random rewards
        rewards = np.random.randn(self.n_envs).astype(np.float32) * 0.1

        # Random episode termination
        dones = np.random.random(self.n_envs) < 0.05  # 5% chance per step
        if self.step_count >= self.max_steps:
            dones = np.ones(self.n_envs, dtype=bool)

        infos = [{}] * self.n_envs

        return obs, rewards, dones, infos

    def render(self):
        """Mock render (no videos for pure testing)."""
        return [None] * self.n_envs

    def call(self, method, attr=None):
        """Mock environment calls."""
        if method == 'get_attr' and attr == 'reward':
            return [np.random.randn() * 2] * self.n_envs

    def call_each(self, method, args_list):
        """Mock method calls."""
        pass


class MinimalRolloutCollector:
    """
    Minimal rollout collector that focuses only on the core collection logic.
    No inheritance from complex base classes.
    """

    def __init__(self, max_episode_steps: int = 50, n_episodes: int = 4):
        self.max_episode_steps = max_episode_steps
        self.n_episodes = n_episodes

    def collect_episodes(self, policy, device: str = "cpu"):
        """
        Collect episodes using the policy.

        Returns:
            episode_data: Dictionary with collected episode trajectories
        """
        device = torch.device(device)
        env = MockEnvironment(n_envs=self.n_episodes, max_steps=self.max_episode_steps)

        # Storage for all episodes
        all_episode_data = []

        print(f"Collecting {self.n_episodes} episodes...")

        # Reset environment
        obs = env.reset()
        policy.reset()

        # Episode storage
        episode_obs = {key: [] for key in obs.keys()}
        episode_actions = []
        episode_fixed_noise = []
        episode_rewards = []
        episode_dones = []

        step_count = 0
        done = False

        while not done and step_count < self.max_episode_steps:
            # Store current observations
            for key, value in obs.items():
                episode_obs[key].append(value)

            # Convert observations to torch tensors
            obs_dict = {}
            for key, value in obs.items():
                obs_dict[key] = torch.from_numpy(value).to(device)

            # Generate fixed noise for flow SDE testing
            batch_size = obs_dict['point_cloud'].shape[0]
            fixed_noise = torch.randn(batch_size, policy.horizon, policy.n_action_steps, device=device)

            # Get actions from policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict, noise=fixed_noise)

            action = action_dict['action'].cpu().numpy()

            # Store actions and noise
            episode_actions.append(action)
            episode_fixed_noise.append(fixed_noise.cpu().numpy())

            # Step environment
            obs, reward, done, info = env.step(action)

            # Store rewards and dones
            episode_rewards.append(reward)
            episode_dones.append(done)

            # Check if all episodes are done
            done = np.all(done)
            step_count += 1

        # Process collected data into individual episodes
        for env_idx in range(self.n_episodes):
            episode_data = {
                'obs': {},
                'actions': np.array([act[env_idx] for act in episode_actions]),
                'fixed_noise': np.array([noise[env_idx] for noise in episode_fixed_noise]),
                'rewards': np.array([rew[env_idx] for rew in episode_rewards]),
                'dones': np.array([done[env_idx] for done in episode_dones]),
                'episode_length': len(episode_actions),
                'total_reward': sum([rew[env_idx] for rew in episode_rewards])
            }

            # Process observations
            for key in episode_obs.keys():
                episode_data['obs'][key] = np.array([obs[env_idx] for obs in episode_obs[key]])

            all_episode_data.append(episode_data)

        print(f"✓ Collected {len(all_episode_data)} episodes")

        return {
            'episodes': all_episode_data,
            'episode_lengths': [ep['episode_length'] for ep in all_episode_data],
            'total_rewards': [ep['total_reward'] for ep in all_episode_data],
        }


def test_rollout_collector_only():
    """Test only the rollout collection logic with minimal components."""
    print("=" * 50)
    print("Testing RL Rollout Collector Only")
    print("=" * 50)

    device = "cpu"  # Use CPU to avoid GPU memory
    print(f"Using device: {device}")

    try:
        # 1. Create minimal dummy policy
        print("\n1. Creating Minimal Dummy Policy...")
        policy = MinimalDummyPolicy(
            horizon=16,
            n_action_steps=8,
            action_dim=10,
            device=device
        )
        print("   ✓ No-parameter dummy policy created")

        # 2. Test policy with sample observations
        print("\n2. Testing Policy Interface...")
        batch_size = 4
        test_obs = {
            'point_cloud': torch.randn(batch_size, 1024, 6),
            'robot0_eef_pos': torch.randn(batch_size, 3),
            'robot0_eef_quat': torch.randn(batch_size, 4),
            'robot0_gripper_qpos': torch.randn(batch_size, 2),
        }

        action_dict = policy.predict_action(test_obs)
        print(f"   ✓ Policy output shape: {action_dict['action'].shape}")

        # 3. Create rollout collector
        print("\n3. Creating Minimal Rollout Collector...")
        collector = MinimalRolloutCollector(
            max_episode_steps=30,  # Short episodes
            n_episodes=3
        )
        print("   ✓ Rollout collector created")

        # 4. Test episode collection
        print("\n4. Testing Episode Collection...")
        episode_data = collector.collect_episodes(policy, device=device)

        # 5. Verify collected data
        print("\n5. Verifying Collected Data...")
        episodes = episode_data['episodes']
        print(f"   - Number of episodes: {len(episodes)}")
        print(f"   - Episode lengths: {episode_data['episode_lengths']}")
        print(f"   - Total rewards: {[f'{r:.3f}' for r in episode_data['total_rewards']]}")

        # Check first episode in detail
        ep0 = episodes[0]
        print(f"\n   Episode 0 details:")
        print(f"   - Actions shape: {ep0['actions'].shape}")
        print(f"   - Fixed noise shape: {ep0['fixed_noise'].shape}")
        print(f"   - Rewards shape: {ep0['rewards'].shape}")
        print(f"   - Observation keys: {list(ep0['obs'].keys())}")
        print(f"   - Point cloud obs shape: {ep0['obs']['point_cloud'].shape}")

        # 6. Test data consistency
        print("\n6. Testing Data Consistency...")
        for i, ep in enumerate(episodes):
            ep_len = ep['episode_length']
            assert ep['actions'].shape[0] == ep_len, f"Episode {i} action length mismatch"
            assert ep['rewards'].shape[0] == ep_len, f"Episode {i} reward length mismatch"
            assert ep['obs']['point_cloud'].shape[0] == ep_len, f"Episode {i} obs length mismatch"

        print("   ✓ All episodes have consistent data shapes")

        # 7. Test fixed noise storage
        print("\n7. Testing Fixed Noise Storage...")
        for i, ep in enumerate(episodes):
            noise_shape = ep['fixed_noise'].shape
            expected_shape = (ep['episode_length'], policy.horizon, policy.n_action_steps)
            assert noise_shape == expected_shape, f"Episode {i} noise shape {noise_shape} != {expected_shape}"

        print("   ✓ Fixed noise stored correctly for all episodes")

        print("\n" + "=" * 50)
        print("✅ Rollout Collector Test PASSED!")
        print("   Core collection logic works correctly")
        print("=" * 50)

        return episode_data

    except Exception as e:
        print(f"\n❌ Test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_rollout_collector_only()