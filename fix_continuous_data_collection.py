#!/usr/bin/env python3

import numpy as np
import torch
import collections
from typing import Dict, List


def collect_rollout_batch_continuous(self, policy) -> Dict[str, torch.Tensor]:
    """
    Fixed version that ensures data from each environment is continuous.
    """
    device = policy.device
    env = self.env

    n_episodes = self.n_episodes_per_batch
    n_envs = min(self.n_envs, n_episodes)
    n_chunks = math.ceil(n_episodes / n_envs)

    all_episode_data = []

    for chunk_idx in range(n_chunks):
        start_ep = chunk_idx * n_envs
        end_ep = min(n_episodes, start_ep + n_envs)
        this_n_active_envs = end_ep - start_ep

        # Reset environments
        obs = env.reset()
        policy.reset()

        # FIXED: Separate storage for each environment
        env_episode_data = []
        for env_idx in range(this_n_active_envs):
            env_episode_data.append({
                'obs': collections.defaultdict(list),
                'actions': [],
                'action_means': [],
                'action_stds': [],
                'values': [],
                'fixed_noise': [],
                'rewards': [],
                'dones': [],
                'step_count': 0
            })

        step_count = 0
        done = False

        while not done and step_count < self.max_episode_steps:
            # Process observations
            np_obs_dict = dict(obs)

            # Store observations PER environment
            for env_idx in range(this_n_active_envs):
                for key, value in np_obs_dict.items():
                    env_episode_data[env_idx]['obs'][key].append(value[env_idx])

            # Policy inference (batch processing for efficiency)
            if self.normalizer is not None:
                np_obs_dict = self.normalizer.normalize(np_obs_dict)

            obs_dict = dict_apply(
                np_obs_dict,
                lambda x: torch.from_numpy(x[:this_n_active_envs]).to(device=device)
            )

            # Generate fixed noise
            batch_size = obs_dict['point_cloud'].shape[0] if 'point_cloud' in obs_dict else obs_dict[list(obs_dict.keys())[0]].shape[0]
            action_shape = (batch_size, policy.horizon, policy.n_action_steps)
            fixed_noise = torch.randn(action_shape, device=device)

            # Get actions
            with torch.no_grad():
                try:
                    action_dict = policy.predict_action(obs_dict, noise=fixed_noise, return_values=True)
                except TypeError:
                    try:
                        action_dict = policy.predict_action(obs_dict, return_values=True)
                    except TypeError:
                        action_dict = policy.predict_action(obs_dict)

            np_action_dict = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())
            actions = np_action_dict['action']
            action_means = np_action_dict.get('x_mean', actions)
            action_stds = np_action_dict.get('x_std', np.zeros_like(actions))
            values = np_action_dict.get('values', np.zeros(actions.shape[0]))
            fixed_noise_np = fixed_noise.detach().cpu().numpy()

            # Store data PER environment
            for env_idx in range(this_n_active_envs):
                env_episode_data[env_idx]['actions'].append(actions[env_idx])
                env_episode_data[env_idx]['action_means'].append(action_means[env_idx])
                env_episode_data[env_idx]['action_stds'].append(action_stds[env_idx])
                env_episode_data[env_idx]['values'].append(values[env_idx])
                env_episode_data[env_idx]['fixed_noise'].append(fixed_noise_np[env_idx])

            # Step environment
            obs, reward, done, info = env.step(actions)

            # Store rewards and dones PER environment
            for env_idx in range(this_n_active_envs):
                env_episode_data[env_idx]['rewards'].append(reward[env_idx])
                env_episode_data[env_idx]['dones'].append(done[env_idx])

                # Track individual environment completion
                if done[env_idx]:
                    env_episode_data[env_idx]['step_count'] = step_count + 1

            # Check if all environments are done
            done = np.all(done[:this_n_active_envs])
            step_count += 1

        # Convert per-environment data to final format
        for env_idx in range(this_n_active_envs):
            env_data = env_episode_data[env_idx]

            # Determine actual episode length for this environment
            actual_length = env_data['step_count'] if env_data['step_count'] > 0 else step_count

            episode_data = {
                'obs': {},
                'actions': np.array(env_data['actions'][:actual_length]),
                'action_means': np.array(env_data['action_means'][:actual_length]),
                'action_stds': np.array(env_data['action_stds'][:actual_length]),
                'values': np.array(env_data['values'][:actual_length]),
                'fixed_noise': np.array(env_data['fixed_noise'][:actual_length]),
                'rewards': np.array(env_data['rewards'][:actual_length]),
                'dones': np.array(env_data['dones'][:actual_length]),
                'episode_length': actual_length,
                'total_reward': sum(env_data['rewards'][:actual_length])
            }

            # Process observations
            for key in env_data['obs'].keys():
                episode_data['obs'][key] = np.array(env_data['obs'][key][:actual_length])

            all_episode_data.append(episode_data)

    # Convert to batch format
    return self._convert_to_batch_format(all_episode_data)


def verify_data_continuity(batch_data, env_idx=0):
    """Verify that data from one environment is continuous."""

    print(f"Verifying data continuity for environment {env_idx}:")

    episode_length = batch_data['episode_lengths'][env_idx].item()

    # Check that actions form a continuous sequence
    actions = batch_data['actions'][env_idx, :episode_length]
    print(f"  Episode length: {episode_length}")
    print(f"  First 3 actions: {actions[:3]}")
    print(f"  Last 3 actions: {actions[-3:]}")

    # Check that rewards are continuous
    rewards = batch_data['rewards'][env_idx, :episode_length]
    print(f"  Reward sequence: {rewards[:10]}...")  # First 10 rewards
    print(f"  Total reward: {rewards.sum():.3f}")

    # Check observations are continuous
    if 'robot0_eef_pos' in batch_data['obs']:
        eef_pos = batch_data['obs']['robot0_eef_pos'][env_idx, :episode_length]
        print(f"  EEF position start: {eef_pos[0]}")
        print(f"  EEF position end: {eef_pos[-1]}")

        # Check for smooth transitions (no sudden jumps)
        pos_diffs = torch.diff(eef_pos, dim=0).norm(dim=1)
        max_jump = pos_diffs.max()
        print(f"  Max position jump: {max_jump:.6f}")

        if max_jump > 0.1:  # Large jump indicates discontinuity
            print(f"  ❌ WARNING: Large position jump detected!")
        else:
            print(f"  ✅ Position sequence looks continuous")


# Example usage in test
if __name__ == "__main__":

    print("=" * 60)
    print("Testing Continuous Data Collection")
    print("=" * 60)

    # Create mock data to demonstrate the issue
    def create_mock_mixed_data():
        """Simulate current problematic data collection."""
        n_envs = 3
        n_steps = 5

        # Simulate step-wise collection (current broken method)
        all_step_actions = []
        for step in range(n_steps):
            # Each step has actions from all environments
            step_actions = np.random.randn(n_envs, 7)  # 3 envs, 7-DOF actions
            all_step_actions.append(step_actions)

        # Current broken extraction
        broken_env0_actions = np.array([step_actions[0] for step_actions in all_step_actions])

        print("Broken method - Environment 0 actions:")
        print(f"  Shape: {broken_env0_actions.shape}")
        print(f"  Actions:\n{broken_env0_actions}")

        return broken_env0_actions

    def create_mock_continuous_data():
        """Simulate fixed continuous data collection."""
        n_envs = 3
        n_steps = 5

        # Simulate per-environment collection (fixed method)
        env_data = []
        for env_idx in range(n_envs):
            env_actions = []
            for step in range(n_steps):
                # Each environment has continuous action sequence
                action = np.random.randn(7) + env_idx  # Add env_idx to make distinct
                env_actions.append(action)
            env_data.append(np.array(env_actions))

        print("Fixed method - Environment 0 actions:")
        print(f"  Shape: {env_data[0].shape}")
        print(f"  Actions:\n{env_data[0]}")

        return env_data[0]

    print("1. Demonstrating the problem:")
    broken_data = create_mock_mixed_data()

    print("\n2. Showing the fix:")
    continuous_data = create_mock_continuous_data()

    print("\n3. Key differences:")
    print("  - Broken: Data from different environments gets mixed")
    print("  - Fixed: Each environment's data stays continuous")
    print("  - Fixed: Proper episode length tracking per environment")
    print("  - Fixed: Handles different episode completion times")