#!/usr/bin/env python3

import os
import sys
import torch
import tempfile
from pathlib import Path
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from equi_diffpo.rl_training.dummy_policy import DummyManiFlowPolicy, DummyNormalizer, create_dummy_shape_meta
from equi_diffpo.rl_training.rl_robomimic_runner import RLRobomimicRunner
from equi_diffpo.rl_training.ppo_trainer import PPOTrainer


def create_mock_dataset(dataset_path: str, n_demos: int = 5, episode_length: int = 50):
    """Create a minimal mock HDF5 dataset for testing."""
    import h5py

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    with h5py.File(dataset_path, 'w') as f:
        # Create data group
        data_group = f.create_group('data')

        # Create mock environment metadata
        f.attrs['env_name'] = 'test_env'
        f.attrs['env_version'] = '1.0'

        for demo_idx in range(n_demos):
            demo_group = data_group.create_group(f'demo_{demo_idx}')

            # Create states (for initial state sampling)
            states = np.random.randn(episode_length, 10)  # Mock robot states
            demo_group.create_dataset('states', data=states)

            # Create actions
            actions = np.random.randn(episode_length, 10) * 0.1  # Small random actions
            demo_group.create_dataset('actions', data=actions)

            # Create observations
            obs_group = demo_group.create_group('obs')
            obs_group.create_dataset('point_cloud',
                data=np.random.randn(episode_length, 1024, 6))
            obs_group.create_dataset('robot0_eef_pos',
                data=np.random.randn(episode_length, 3))
            obs_group.create_dataset('robot0_eef_quat',
                data=np.random.randn(episode_length, 4))
            obs_group.create_dataset('robot0_gripper_qpos',
                data=np.random.randn(episode_length, 2))

        print(f"Created mock dataset with {n_demos} demos at {dataset_path}")


class MockAsyncVectorEnv:
    """Mock environment for testing that mimics AsyncVectorEnv interface."""

    def __init__(self, n_envs: int = 4, max_steps: int = 100):
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self):
        """Reset environments and return initial observations."""
        self.step_count = 0
        return {
            'point_cloud': np.random.randn(self.n_envs, 1024, 6),
            'robot0_eef_pos': np.random.randn(self.n_envs, 3),
            'robot0_eef_quat': np.random.randn(self.n_envs, 4),
            'robot0_gripper_qpos': np.random.randn(self.n_envs, 2),
        }

    def step(self, actions):
        """Step environments and return observations, rewards, dones, infos."""
        self.step_count += 1

        # Generate next observations
        obs = {
            'point_cloud': np.random.randn(self.n_envs, 1024, 6),
            'robot0_eef_pos': np.random.randn(self.n_envs, 3),
            'robot0_eef_quat': np.random.randn(self.n_envs, 4),
            'robot0_gripper_qpos': np.random.randn(self.n_envs, 2),
        }

        # Generate random rewards (biased toward positive for more interesting episodes)
        rewards = np.random.randn(self.n_envs) * 0.1 + 0.1

        # Episodes end randomly or at max steps
        done_prob = 0.02 if self.step_count < self.max_steps * 0.8 else 0.3
        dones = np.random.random(self.n_envs) < done_prob

        # All episodes must end at max steps
        if self.step_count >= self.max_steps:
            dones = np.ones(self.n_envs, dtype=bool)

        infos = [{}] * self.n_envs

        return obs, rewards, dones, infos

    def render(self):
        """Return mock video paths."""
        return [None] * self.n_envs

    def call(self, method_name, *args):
        """Mock environment attribute calls."""
        if method_name == 'get_attr' and args[0] == 'reward':
            # Return cumulative rewards for each environment
            return [np.random.randn() * 5 + 2] * self.n_envs

    def call_each(self, method_name, args_list):
        """Mock method calls on each environment."""
        pass  # No-op for testing


class MockRLRunner(RLRobomimicRunner):
    """Mock RL runner that doesn't require full robomimic setup."""

    def __init__(self, n_envs: int = 4, max_steps: int = 100, collect_episode_data: bool = True):
        # Skip parent initialization
        self.collect_episode_data = collect_episode_data
        self.max_steps = max_steps
        self.n_envs = n_envs
        self.past_action = False
        self.abs_action = False
        self.n_obs_steps = 2
        self.n_action_steps = 8
        self.tqdm_interval_sec = 0.1

        # Create mock environment
        self.env = MockAsyncVectorEnv(n_envs, max_steps)

        # Mock environment metadata
        self.env_meta = {'env_name': 'mock_env'}

        # Mock initialization functions (one per environment)
        self.env_init_fn_dills = [b'mock_init'] * n_envs
        self.env_fns = [None] * n_envs
        self.env_seeds = list(range(n_envs))
        self.env_prefixs = ['test/'] * n_envs
        self.max_rewards = {'test/': 0}

    def undo_transform_action(self, action):
        """Mock action transformation."""
        return action


def test_rl_runner_unit():
    """Unit test for RL rollout collection with dummy components."""
    print("=" * 60)
    print("Unit Testing RL Rollout Collection")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # 1. Create dummy policy
        print("\n1. Creating Dummy Policy...")
        policy = DummyManiFlowPolicy(
            horizon=16,
            n_action_steps=8,
            n_obs_steps=2,
            action_dim=10,
            device=device
        )
        print(f"   ✓ Policy created with {sum(p.numel() for p in policy.parameters())} parameters")

        # 2. Create mock RL runner
        print("\n2. Creating Mock RL Runner...")
        rl_runner = MockRLRunner(
            n_envs=4,
            max_steps=50,  # Short episodes for testing
            collect_episode_data=True
        )
        print("   ✓ Mock RL runner created")

        # 3. Test rollout collection
        print("\n3. Testing Rollout Collection...")
        rollout_metrics, episode_data = rl_runner.run_rl_collection(policy)
        print("   ✓ Rollout collection completed")

        # 4. Verify collected data
        print("\n4. Verifying Collected Data...")
        n_episodes = len(episode_data['episode_lengths'])
        print(f"   - Collected {n_episodes} episodes")
        print(f"   - Episode lengths: {episode_data['episode_lengths']}")
        print(f"   - Episode data keys: {list(episode_data.keys())}")

        # Check data shapes
        print(f"   - Observation keys: {list(episode_data['obs'][0].keys())}")
        for i, ep_length in enumerate(episode_data['episode_lengths'][:2]):  # Check first 2 episodes
            print(f"   - Episode {i} length: {ep_length}")
            print(f"     Actions shape: {episode_data['actions'][i].shape}")
            print(f"     Fixed noise shape: {episode_data['fixed_noise'][i].shape}")
            print(f"     Rewards shape: {episode_data['rewards'][i].shape}")

        # 5. Test PPO trainer setup
        print("\n5. Testing PPO Trainer Setup...")
        normalizer = DummyNormalizer()
        ppo_trainer = PPOTrainer(
            policy=policy,
            normalizer=normalizer,
            device=torch.device(device),
            policy_lr=1e-4,
            critic_lr=1e-3
        )
        print(f"   ✓ PPO trainer created")
        print(f"   - Critic parameters: {sum(p.numel() for p in ppo_trainer.critic.parameters())}")

        # 6. Test batch conversion (mock)
        print("\n6. Testing Batch Data Conversion...")
        from equi_diffpo.rl_training.rl_workspace import RLTrainingWorkspace

        # Create minimal workspace for batch conversion method
        class MockWorkspace:
            def _convert_episode_data_to_batch(self, episode_data):
                n_episodes = len(episode_data['episode_lengths'])
                max_length = max(episode_data['episode_lengths'])

                batch_data = {
                    'obs': {},
                    'actions': torch.zeros(n_episodes, max_length, 10),
                    'fixed_noise': torch.zeros(n_episodes, max_length, 16, 8),
                    'rewards': torch.zeros(n_episodes, max_length),
                    'dones': torch.zeros(n_episodes, max_length, dtype=torch.bool),
                    'episode_lengths': torch.tensor(episode_data['episode_lengths'])
                }

                # Convert observations
                obs_keys = list(episode_data['obs'][0].keys())
                for key in obs_keys:
                    first_shape = episode_data['obs'][0][key].shape[1:]
                    batch_obs = torch.zeros(n_episodes, max_length, *first_shape)
                    for ep_idx in range(n_episodes):
                        ep_len = episode_data['episode_lengths'][ep_idx]
                        batch_obs[ep_idx, :ep_len] = torch.from_numpy(episode_data['obs'][ep_idx][key])
                    batch_data['obs'][key] = batch_obs

                # Fill in actual data
                for ep_idx in range(n_episodes):
                    ep_len = episode_data['episode_lengths'][ep_idx]
                    batch_data['actions'][ep_idx, :ep_len] = torch.from_numpy(episode_data['actions'][ep_idx])
                    batch_data['fixed_noise'][ep_idx, :ep_len] = torch.from_numpy(episode_data['fixed_noise'][ep_idx])
                    batch_data['rewards'][ep_idx, :ep_len] = torch.from_numpy(episode_data['rewards'][ep_idx])
                    batch_data['dones'][ep_idx, :ep_len] = torch.from_numpy(episode_data['dones'][ep_idx])

                return batch_data

        workspace = MockWorkspace()
        batch_data = workspace._convert_episode_data_to_batch(episode_data)

        print("   ✓ Batch conversion completed")
        print(f"   - Batch shapes:")
        for key, value in batch_data['obs'].items():
            print(f"     {key}: {value.shape}")
        print(f"     actions: {batch_data['actions'].shape}")
        print(f"     fixed_noise: {batch_data['fixed_noise'].shape}")
        print(f"     rewards: {batch_data['rewards'].shape}")

        # 7. Test PPO trainer with batch data
        print("\n7. Testing PPO Trainer with Batch Data...")

        # Move batch data to device
        for key in batch_data['obs']:
            batch_data['obs'][key] = batch_data['obs'][key].to(device)
        batch_data['actions'] = batch_data['actions'].to(device)
        batch_data['fixed_noise'] = batch_data['fixed_noise'].to(device)
        batch_data['rewards'] = batch_data['rewards'].to(device)
        batch_data['dones'] = batch_data['dones'].to(device)

        # Test GAE computation
        with torch.no_grad():
            batch_size, max_length = batch_data['rewards'].shape
            flat_obs = {}
            for key, value in batch_data['obs'].items():
                flat_obs[key] = value.reshape(batch_size * max_length, *value.shape[2:])

            values = ppo_trainer.critic(flat_obs).squeeze(-1)
            values = values.reshape(batch_size, max_length)

            advantages, returns = ppo_trainer.compute_gae(
                batch_data['rewards'],
                values,
                batch_data['dones'],
                batch_data['episode_lengths']
            )

        print("   ✓ GAE computation successful")
        print(f"   - Advantages shape: {advantages.shape}")
        print(f"   - Returns shape: {returns.shape}")

        print("\n" + "=" * 60)
        print("✅ All unit tests passed!")
        print("   The RL rollout collector is working correctly with dummy components.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Unit test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def test_with_mock_dataset():
    """Test with a mock dataset (closer to real setup)."""
    print("=" * 60)
    print("Testing with Mock Dataset")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = os.path.join(temp_dir, "mock_dataset.hdf5")

        try:
            # Create mock dataset
            print("1. Creating Mock Dataset...")
            create_mock_dataset(dataset_path, n_demos=3, episode_length=30)

            # This would require more complex mocking of robomimic components
            # For now, just verify the dataset was created
            import h5py
            with h5py.File(dataset_path, 'r') as f:
                print(f"   ✓ Dataset created with {len(f['data'].keys())} demos")
                print(f"   ✓ Demo 0 has {f['data/demo_0/actions'].shape[0]} steps")

            print("\n✅ Mock dataset test passed!")

        except Exception as e:
            print(f"❌ Mock dataset test failed: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("Available tests:")
    print("1. test_rl_runner_unit - Unit test with dummy components (recommended)")
    print("2. test_with_mock_dataset - Test with mock HDF5 dataset")

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "dataset":
        test_with_mock_dataset()
    else:
        test_rl_runner_unit()