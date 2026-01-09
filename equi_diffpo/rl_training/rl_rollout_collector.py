import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import math
import dill
from typing import Dict, List, Optional, Tuple
import wandb.sdk.data_types.video as wv

from equi_diffpo.gym_util.async_vector_env import AsyncVectorEnv
from equi_diffpo.gym_util.sync_vector_env import SyncVectorEnv
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.model.common.normalizer import LinearNormalizer


class RLRolloutCollector:
    """
    RL Rollout collector for ManiFlow policies.
    Collects experience for reinforcement learning training.
    """

    def __init__(
        self,
        env_runner,
        normalizer: Optional[LinearNormalizer] = None,
        max_episode_steps: int = 400,
        n_episodes_per_batch: int = 16,
        n_envs: int = 16,
        render_episodes: int = 4,
        device: str = "cuda",
        tqdm_interval_sec: float = 1.0,
        record_video: bool = True,
    ):
        """
        Args:
            env_runner: Environment runner (from config)
            normalizer: Observation normalizer
            max_episode_steps: Maximum steps per episode
            n_episodes_per_batch: Number of episodes to collect per batch
            n_envs: Number of parallel environments
            render_episodes: Number of episodes to render for visualization
            device: Device for tensor operations
            tqdm_interval_sec: Progress bar update interval
            record_video: Whether to record videos
        """
        self.env_runner = env_runner
        self.normalizer = normalizer
        self.max_episode_steps = max_episode_steps
        self.n_episodes_per_batch = n_episodes_per_batch
        self.n_envs = n_envs
        self.render_episodes = render_episodes
        self.device = torch.device(device)
        self.tqdm_interval_sec = tqdm_interval_sec
        self.record_video = record_video

        # Use the environment from env_runner
        self.env = env_runner.env
        self.env_seeds = getattr(env_runner, 'env_seeds', list(range(n_episodes_per_batch)))
        self.env_prefixs = getattr(env_runner, 'env_prefixs', ['rl/'] * n_episodes_per_batch)

        # Metrics tracking
        self.episode_count = 0
        self.total_steps = 0

    def collect_rollout_batch(self, policy: BaseImagePolicy) -> Dict[str, torch.Tensor]:
        """
        Collect a batch of rollout episodes for RL training.
        FIXED: Ensures data from each environment is continuous.

        Args:
            policy: Policy to roll out

        Returns:
            batch_data: Dictionary containing collected episode data
                - obs: Dictionary of observations [n_episodes, max_steps, ...]
                - actions: Actions taken [n_episodes, max_steps, action_dim]
                - action_means: Action means from policy [n_episodes, max_steps, action_dim]
                - action_stds: Action stds from policy [n_episodes, max_steps, action_dim]
                - values: State value estimates [n_episodes, max_steps]
                - fixed_noise: Fixed noise for flow SDE [n_episodes, max_steps, ...]
                - rewards: Rewards received [n_episodes, max_steps]
                - dones: Episode termination flags [n_episodes, max_steps]
                - episode_lengths: Length of each episode [n_episodes]
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
        batch_data = self._convert_to_batch_format(all_episode_data)

        # Log metrics
        log_data = self._compute_metrics(all_episode_data, [])

        # Update counters
        self.episode_count += len(all_episode_data)
        self.total_steps += sum([ep['episode_length'] for ep in all_episode_data])

        return batch_data, log_data

    def _convert_to_batch_format(self, episode_data_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Convert list of episode data to batch tensor format.

        Args:
            episode_data_list: List of episode dictionaries

        Returns:
            batch_data: Batched tensor data
        """
        n_episodes = len(episode_data_list)
        max_length = max([ep['episode_length'] for ep in episode_data_list])

        # Initialize batch tensors
        batch_data = {
            'obs': {},
            'actions': None,
            'action_means': None,
            'action_stds': None,
            'values': None,
            'fixed_noise': None,
            'rewards': None,
            'dones': None,
            'episode_lengths': torch.tensor([ep['episode_length'] for ep in episode_data_list])
        }

        # Get observation keys from first episode
        obs_keys = list(episode_data_list[0]['obs'].keys())

        # Process observations
        for key in obs_keys:
            # Get shape from first observation
            first_obs_shape = episode_data_list[0]['obs'][key].shape[1:]
            batch_obs = torch.zeros(n_episodes, max_length, *first_obs_shape, dtype=torch.float32)

            for ep_idx, ep_data in enumerate(episode_data_list):
                ep_length = ep_data['episode_length']
                batch_obs[ep_idx, :ep_length] = torch.from_numpy(ep_data['obs'][key])

            batch_data['obs'][key] = batch_obs

        # Process actions
        action_dim = episode_data_list[0]['actions'].shape[-1]
        batch_actions = torch.zeros(n_episodes, max_length, action_dim, dtype=torch.float32)

        for ep_idx, ep_data in enumerate(episode_data_list):
            ep_length = ep_data['episode_length']
            batch_actions[ep_idx, :ep_length] = torch.from_numpy(ep_data['actions'])

        batch_data['actions'] = batch_actions

        # Process action means
        batch_action_means = torch.zeros(n_episodes, max_length, action_dim, dtype=torch.float32)
        for ep_idx, ep_data in enumerate(episode_data_list):
            ep_length = ep_data['episode_length']
            batch_action_means[ep_idx, :ep_length] = torch.from_numpy(ep_data['action_means'])
        batch_data['action_means'] = batch_action_means

        # Process action stds
        batch_action_stds = torch.zeros(n_episodes, max_length, action_dim, dtype=torch.float32)
        for ep_idx, ep_data in enumerate(episode_data_list):
            ep_length = ep_data['episode_length']
            batch_action_stds[ep_idx, :ep_length] = torch.from_numpy(ep_data['action_stds'])
        batch_data['action_stds'] = batch_action_stds

        # Process values
        batch_values = torch.zeros(n_episodes, max_length, dtype=torch.float32)
        for ep_idx, ep_data in enumerate(episode_data_list):
            ep_length = ep_data['episode_length']
            batch_values[ep_idx, :ep_length] = torch.from_numpy(ep_data['values'])
        batch_data['values'] = batch_values

        # Process fixed noise
        noise_shape = episode_data_list[0]['fixed_noise'].shape[1:]  # Skip time dimension
        batch_fixed_noise = torch.zeros(n_episodes, max_length, *noise_shape, dtype=torch.float32)

        for ep_idx, ep_data in enumerate(episode_data_list):
            ep_length = ep_data['episode_length']
            batch_fixed_noise[ep_idx, :ep_length] = torch.from_numpy(ep_data['fixed_noise'])

        batch_data['fixed_noise'] = batch_fixed_noise

        # Process rewards
        batch_rewards = torch.zeros(n_episodes, max_length, dtype=torch.float32)

        for ep_idx, ep_data in enumerate(episode_data_list):
            ep_length = ep_data['episode_length']
            batch_rewards[ep_idx, :ep_length] = torch.from_numpy(ep_data['rewards'])

        batch_data['rewards'] = batch_rewards

        # Process dones
        batch_dones = torch.zeros(n_episodes, max_length, dtype=torch.bool)

        for ep_idx, ep_data in enumerate(episode_data_list):
            ep_length = ep_data['episode_length']
            batch_dones[ep_idx, :ep_length] = torch.from_numpy(ep_data['dones'])

        batch_data['dones'] = batch_dones

        return batch_data

    def _compute_metrics(self, episode_data_list: List[Dict], video_paths: List[str]) -> Dict:
        """
        Compute logging metrics from collected episodes.

        Args:
            episode_data_list: List of episode data
            video_paths: List of video file paths

        Returns:
            log_data: Dictionary of metrics for logging
        """
        log_data = {}

        # Episode metrics
        episode_rewards = [ep['total_reward'] for ep in episode_data_list]
        episode_lengths = [ep['episode_length'] for ep in episode_data_list]

        log_data.update({
            'rl/mean_episode_reward': np.mean(episode_rewards),
            'rl/std_episode_reward': np.std(episode_rewards),
            'rl/max_episode_reward': np.max(episode_rewards),
            'rl/min_episode_reward': np.min(episode_rewards),
            'rl/mean_episode_length': np.mean(episode_lengths),
            'rl/std_episode_length': np.std(episode_lengths),
            'rl/total_episodes': len(episode_data_list),
            'rl/total_steps_collected': sum(episode_lengths),
        })

        # Log individual episode rewards
        for i, reward in enumerate(episode_rewards):
            log_data[f'rl/episode_reward_{i}'] = reward

        # Log videos if available
        if video_paths and self.record_video:
            for i, video_path in enumerate(video_paths[:self.render_episodes]):
                if video_path is not None and os.path.exists(video_path):
                    log_data[f'rl/episode_video_{i}'] = wandb.Video(video_path)

        return log_data

    def get_stats(self) -> Dict:
        """Get collector statistics."""
        return {
            'total_episodes_collected': self.episode_count,
            'total_steps_collected': self.total_steps,
        }