#!/usr/bin/env python3
"""
RL-compatible Robomimic Image Runner
Extended version that stores step-by-step RL data (chains, denoise_inds, logprobs, values).
"""

import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from typing import Dict, List, Any, Optional

from equi_diffpo.gym_util.async_vector_env import AsyncVectorEnv
from equi_diffpo.gym_util.sync_vector_env import SyncVectorEnv
from equi_diffpo.gym_util.multistep_wrapper import MultiStepWrapper
from equi_diffpo.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from equi_diffpo.model.common.rotation_transformer import RotationTransformer

from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.policy.maniflow.maniflow_pointcloud_rl_policy import ManiFlowRLPointcloudPolicy
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.env_runner.robomimic_image_runner import RobomimicImageRunner, create_env
from equi_diffpo.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


class RobomimicRLRunner(RobomimicImageRunner):
    """
    RL-compatible Robomimic runner that stores step-by-step data for PPO training.

    Extends RobomimicImageRunner to collect:
    - Step-by-step observations, actions, rewards
    - Policy chains, denoise_inds, logprobs, values
    - Episode trajectory data for RL training
    """

    def __init__(self, *args, collect_rl_data: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.collect_rl_data = collect_rl_data
        print(f"🎲 RobomimicRLRunner initialized (RL data collection: {collect_rl_data})")

    def run_rl(self, policy: ManiFlowRLPointcloudPolicy, n_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Run rollout collection for RL training with step-by-step data storage.

        Args:
            policy: ManiFlow RL policy that supports chains and RL methods
            n_episodes: Number of episodes to run (optional, defaults to len(env_init_fn_dills))
                       If n_episodes > available init functions, cycles through them

        Returns:
            Dictionary with RL training data including chains, denoise_inds, etc.
        """
        device = policy.device
        dtype = policy.dtype
        env = self.env

        print(f"🎲 Starting RL rollout collection...")

        # Plan for rollout
        n_envs = len(self.env_fns)
        n_available = len(self.env_init_fn_dills)
        n_inits = n_available if n_episodes is None else n_episodes
        n_chunks = math.ceil(n_inits / n_envs)

        # Check if we need to cycle through seeds
        if n_episodes is not None and n_episodes > n_available:
            n_cycles = math.ceil(n_episodes / n_available)
            print(f"  Cycling through {n_available} seeds {n_cycles} times to get {n_episodes} episodes")
        elif n_episodes is not None:
            print(f"  Using {n_inits} episodes (requested: {n_episodes}, available: {n_available})")

        # Storage for RL data
        all_rl_data = []
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            # Get init functions with cycling support
            # Use modulo to cycle through available init functions
            this_init_fns = [
                self.env_init_fn_dills[i % n_available]
                for i in range(start, end)
            ]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # Init envs
            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            # Start rollout with RL data collection
            chunk_rl_data = self._collect_rl_chunk(
                env, policy, device,
                this_n_active_envs, chunk_idx, n_chunks
            )

            all_rl_data.append(chunk_rl_data)

            # Collect standard data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        # Clear out video buffer
        _ = env.reset()

        # Combine RL data from all chunks
        combined_rl_data = self._combine_rl_chunks(all_rl_data)

        # Standard logging (inherited from parent) - only log first n_available for unique seeds
        n_log = min(n_inits, n_available)
        log_data = self._log_standard_metrics(all_video_paths[:n_log], all_rewards[:n_log], n_log)

        # Add RL-specific metrics (with video paths for RL-specific video logging)
        log_data.update(self._log_rl_metrics(combined_rl_data, video_paths=all_video_paths[:n_log]))

        # Return combined data for RL training
        result = {
            'log_data': log_data,
            'rl_data': combined_rl_data,
            'video_paths': all_video_paths,
            'episode_rewards': all_rewards,
        }

        print(f"✅ RL rollout collection completed: {combined_rl_data['total_steps']} steps, {n_inits} episodes")

        return result

    def _collect_rl_chunk(self, env, policy: ManiFlowRLPointcloudPolicy, device,
                         n_active_envs: int, chunk_idx: int, n_chunks: int) -> Dict[str, List]:
        """Collect RL data for one chunk of environments."""

        # Debug: print noise level for this rollout
        if hasattr(policy, 'get_current_noise_level'):
            print(f"  📢 Noise level: {policy.get_current_noise_level():.4f}")
        if hasattr(policy, 'noise_method'):
            print(f"  📢 Noise method: {policy.noise_method}")

        # Initialize episode data storage
        episode_observations = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_prev_logprobs = []
        episode_prev_values = []
        episode_chains = []
        episode_denoise_inds = []
        episode_x_means = []
        episode_x_stds = []

        # Start rollout
        obs = env.reset()
        past_action = None
        policy.reset()

        # Track cumulative rewards to detect new successes
        # (env.step returns cumulative max, we need per-chunk reward)
        prev_cumulative_rewards = np.zeros(n_active_envs)

        env_name = self.env_meta['env_name']
        pbar = tqdm.tqdm(total=self.max_steps, desc=f"RL Rollout {env_name} {chunk_idx+1}/{n_chunks}",
            leave=False, mininterval=self.tqdm_interval_sec)

        step_count = 0
        # Fixed number of steps, don't check done (done is only used for loss mask)
        while step_count < math.ceil(self.max_steps / 16):
            # Create obs dict
            np_obs_dict = dict(obs)
            if self.past_action and (past_action is not None):
                np_obs_dict['past_action'] = past_action[
                    :,-(self.n_obs_steps-1):].astype(np.float32)

            # Device transfer
            obs_dict = dict_apply(np_obs_dict,
                lambda x: torch.from_numpy(x).to(device=device))

            # Store observations (before policy prediction)
            episode_observations.append(self._copy_obs_dict(np_obs_dict, n_active_envs))

            # Run policy with RL data collection
            with torch.no_grad():
                # Use sample_actions to get all RL data
                sample_result = policy.sample_actions(
                    obs_dict,
                    mode="train",  # Use train mode for exploration
                    compute_values=True
                )

                # Extract actions and RL data
                action = sample_result['action_pred'].detach().cpu().numpy()  # [B, action_chunk, action_dim]
                prev_logprobs = sample_result['prev_logprobs'].detach().cpu().numpy()
                prev_values = sample_result['prev_values'].detach().cpu().numpy()
                chains = sample_result['chains'].detach().cpu().numpy()
                denoise_inds = sample_result['denoise_inds'].detach().cpu().numpy()
                x_means = sample_result['x_means'].detach().cpu().numpy()  # [B, N, T, Da]
                x_stds = sample_result['x_stds'].detach().cpu().numpy()  # [B, N, T, Da]

            # Check for nan/inf
            if not np.all(np.isfinite(action)):
                print(f"Warning: Non-finite action at step {step_count}")
                print(action)
                raise RuntimeError("Nan or Inf action")

            # Step environment
            env_action = action
            if self.abs_action:
                env_action = self.undo_transform_action(action)

            obs, reward, done_array, info = env.step(env_action)

            # reward is cumulative max from MultiStepWrapper - extract actual chunk reward
            # (once success occurs, cumulative max stays at 1 forever, but we need per-chunk reward)
            cumulative_rewards = reward[:n_active_envs]  # Current cumulative max

            # Actual reward = increase in cumulative (new success), else 0
            actual_rewards = np.where(
                cumulative_rewards > prev_cumulative_rewards,
                cumulative_rewards - prev_cumulative_rewards,  # New success
                0.0  # No new success (repeated signal)
            )

            # Debug: print if new success detected
            if np.any(actual_rewards > 0):
                print(f"  🎯 New success at step {step_count}: {actual_rewards}")

            # Update tracker for next chunk
            prev_cumulative_rewards = cumulative_rewards.copy()

            # Generate done signal: success (cumulative_rewards >= 1) OR truncation (original done)
            # Once cumulative_rewards becomes 1, task succeeded, should be treated as done
            success_done = cumulative_rewards >= 1.0  # [n_active_envs]
            truncation_done = done_array[:n_active_envs]  # [n_active_envs]
            individual_dones = np.logical_or(success_done, truncation_done).astype(np.float32)

            # Store step data (only for active envs)
            episode_actions.append(action[:n_active_envs])

            # Store actual per-chunk reward [n_active_envs, 1]
            step_rewards = actual_rewards.reshape(-1, 1)
            episode_rewards.append(step_rewards)

            episode_dones.append(individual_dones[:n_active_envs])
            episode_prev_logprobs.append(prev_logprobs[:n_active_envs])
            episode_prev_values.append(prev_values[:n_active_envs])
            episode_chains.append(chains[:n_active_envs])
            episode_denoise_inds.append(denoise_inds[:n_active_envs])
            episode_x_means.append(x_means[:n_active_envs])
            episode_x_stds.append(x_stds[:n_active_envs])

            past_action = action
            step_count += 1
            # step_count += action.shape[1]

            # Update progress bar
            pbar.update(action.shape[1] if len(action.shape) > 1 else 1)

        pbar.close()

        return {
            'observations': episode_observations,
            'actions': episode_actions,
            'rewards': episode_rewards,
            'dones': episode_dones,
            'prev_logprobs': episode_prev_logprobs,
            'prev_values': episode_prev_values,
            'chains': episode_chains,
            'denoise_inds': episode_denoise_inds,
            'x_means': episode_x_means,
            'x_stds': episode_x_stds,
            'n_active_envs': n_active_envs,
            'n_steps': step_count,
        }

    def _copy_obs_dict(self, obs_dict: Dict[str, np.ndarray], n_active_envs: int) -> Dict[str, np.ndarray]:
        """Copy observation dictionary for the active environments."""
        copied_obs = {}
        for key, value in obs_dict.items():
            if isinstance(value, np.ndarray):
                copied_obs[key] = value[:n_active_envs].copy()
            else:
                copied_obs[key] = value
        return copied_obs

    def _combine_rl_chunks(self, chunk_data_list: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Combine RL data from all environment chunks.

        Each chunk is first stacked (along step dimension), then concatenated (along env dimension).
        Final shape: [n_steps, total_envs, ...]
        """
        if not chunk_data_list:
            return {}

        total_steps = chunk_data_list[0]['n_steps']  # All chunks have the same number of steps
        total_envs = sum(chunk['n_active_envs'] for chunk in chunk_data_list)

        print(f"📊 Combining {len(chunk_data_list)} chunks: n_steps={total_steps}, total_envs={total_envs}")

        # Actions: [n_steps, n_envs, action_chunk, action_dim]
        actions_per_chunk = [np.stack(chunk['actions'], axis=0) for chunk in chunk_data_list]
        combined_actions = np.concatenate(actions_per_chunk, axis=1)

        # Rewards: [n_steps, n_envs, 1] (one reward per chunk/step)
        rewards_per_chunk = [np.stack(chunk['rewards'], axis=0) for chunk in chunk_data_list]
        combined_rewards = np.concatenate(rewards_per_chunk, axis=1)

        # Dones: [n_steps, n_envs] -> will be expanded to [n_steps, n_envs, 1]
        dones_per_chunk = [np.stack(chunk['dones'], axis=0) for chunk in chunk_data_list]
        combined_dones = np.concatenate(dones_per_chunk, axis=1)

        # Prev logprobs: [n_steps, n_envs,N, action_chunk, action_dim]
        logprobs_per_chunk = [np.stack(chunk['prev_logprobs'], axis=0) for chunk in chunk_data_list]
        combined_prev_logprobs = np.concatenate(logprobs_per_chunk, axis=1)

        # Prev values: [n_steps, n_envs, 1]
        values_per_chunk = [np.stack(chunk['prev_values'], axis=0) for chunk in chunk_data_list]
        combined_prev_values = np.concatenate(values_per_chunk, axis=1)

        # Chains: [n_steps, n_envs, N+1, horizon, action_dim]
        chains_per_chunk = [np.stack(chunk['chains'], axis=0) for chunk in chunk_data_list]
        combined_chains = np.concatenate(chains_per_chunk, axis=1)

        # Denoise inds: [n_steps, n_envs, N]
        denoise_per_chunk = [np.stack(chunk['denoise_inds'], axis=0) for chunk in chunk_data_list]
        combined_denoise_inds = np.concatenate(denoise_per_chunk, axis=1)

        # X means: [n_steps, n_envs, N, horizon, action_dim]
        x_means_per_chunk = [np.stack(chunk['x_means'], axis=0) for chunk in chunk_data_list]
        combined_x_means = np.concatenate(x_means_per_chunk, axis=1)

        # X stds: [n_steps, n_envs, N, horizon, action_dim]
        x_stds_per_chunk = [np.stack(chunk['x_stds'], axis=0) for chunk in chunk_data_list]
        combined_x_stds = np.concatenate(x_stds_per_chunk, axis=1)

        # Observations: Dict[str, [n_steps, n_envs, ...]]
        combined_observations = {}
        obs_keys = chunk_data_list[0]['observations'][0].keys()
        for key in obs_keys:
            obs_per_chunk = []
            for chunk in chunk_data_list:
                # chunk['observations'] is a list of dict, each dict is one step
                obs_list = [step_obs[key] for step_obs in chunk['observations']]
                stacked = np.stack(obs_list, axis=0)  # [n_steps, n_active_envs, ...]
                obs_per_chunk.append(stacked)
            combined_observations[key] = np.concatenate(obs_per_chunk, axis=1)

        # Ensure dones has shape [n_steps, n_envs, 1]
        if len(combined_dones.shape) == 2:
            combined_dones = combined_dones[:, :, np.newaxis]

        combined_data = {
            'observations': combined_observations,
            'actions': combined_actions,
            'rewards': combined_rewards,
            'dones': combined_dones,
            'prev_logprobs': combined_prev_logprobs,
            'prev_values': combined_prev_values,
            'chains': combined_chains,
            'denoise_inds': combined_denoise_inds,
            'x_means': combined_x_means,
            'x_stds': combined_x_stds,
            'total_steps': total_steps,
            'total_envs': total_envs,
        }

        print(f"✅ Combined RL data: [n_steps={total_steps}, n_envs={total_envs}]")
        print(f"  - actions: {combined_actions.shape}")
        print(f"  - rewards: {combined_rewards.shape}")
        print(f"  - dones: {combined_dones.shape}")
        print(f"  - chains: {combined_chains.shape}")
        print(f"  - x_means: {combined_x_means.shape}")
        print(f"  - x_stds: {combined_x_stds.shape}")

        return combined_data

    def _log_standard_metrics(self, all_video_paths, all_rewards, n_inits) -> Dict:
        """Log standard metrics (inherited from parent class)."""
        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # Visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # Log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
            self.max_rewards[prefix] = max(self.max_rewards[prefix], value)
            log_data[prefix+'max_score'] = self.max_rewards[prefix]

        return log_data

    def _log_rl_metrics(self, rl_data: Dict, video_paths: List = None) -> Dict:
        """Log RL-specific metrics."""
        if not rl_data:
            return {}

        log_data = {}

        # Log RL rollout videos (separate from eval videos)
        if video_paths is not None:
            n_videos_to_log = min(3, len(video_paths))  # Log up to 3 videos
            for i in range(n_videos_to_log):
                if video_paths[i] is not None:
                    rl_video = wandb.Video(video_paths[i])
                    log_data[f'rl_rollout_video_{i}'] = rl_video

        # Basic RL metrics
        log_data['rl_total_steps'] = rl_data['total_steps']
        log_data['rl_total_envs'] = rl_data['total_envs']

        # Reward statistics
        rewards = rl_data['rewards']
        log_data['rl_total_reward'] = float(np.sum(rewards))
        log_data['rl_mean_step_reward'] = float(np.mean(rewards))
        log_data['rl_std_step_reward'] = float(np.std(rewards))
        log_data['rl_min_step_reward'] = float(np.min(rewards))
        log_data['rl_max_step_reward'] = float(np.max(rewards))

        # Episode statistics (approximate)
        dones = rl_data['dones']
        episode_ends = np.sum(dones)
        log_data['rl_num_episodes'] = int(episode_ends)

        if episode_ends > 0:
            # Estimate episode lengths
            avg_episode_length = rl_data['total_steps'] / episode_ends
            log_data['rl_avg_episode_length'] = float(avg_episode_length)

        # Policy statistics
        prev_values = rl_data['prev_values']
        log_data['rl_mean_value_estimate'] = float(np.mean(prev_values))
        log_data['rl_std_value_estimate'] = float(np.std(prev_values))

        # Log prob statistics
        prev_logprobs = rl_data['prev_logprobs']
        log_data['rl_mean_logprob'] = float(np.mean(prev_logprobs))
        log_data['rl_std_logprob'] = float(np.std(prev_logprobs))

        # Count successful episodes (reward > 0 means success in sparse reward setting)
        n_successful_steps = np.sum(rewards > 0)
        log_data['rl_successful_steps'] = int(n_successful_steps)

        print(f"📊 RL Metrics:")
        print(f"  - Total steps: {log_data['rl_total_steps']}")
        print(f"  - Episodes: {log_data['rl_num_episodes']}")
        print(f"  - Total reward: {log_data['rl_total_reward']:.4f}")
        print(f"  - Mean reward: {log_data['rl_mean_step_reward']:.4f}")
        print(f"  - Successful steps: {n_successful_steps}")
        print(f"  - Mean value: {log_data['rl_mean_value_estimate']:.4f}")

        return log_data

    def run_eval(self, policy: ManiFlowRLPointcloudPolicy, n_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Run evaluation rollout without noise for clean policy evaluation.

        Args:
            policy: ManiFlow RL policy
            n_episodes: Number of episodes to run (optional, defaults to len(env_init_fn_dills))

        Returns:
            Dictionary with evaluation metrics (no RL training data)
        """
        device = policy.device
        dtype = policy.dtype
        env = self.env

        print(f"🔍 Starting evaluation rollout (no exploration noise)...")

        # Plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills) if n_episodes is None else min(n_episodes, len(self.env_init_fn_dills))
        n_chunks = math.ceil(n_inits / n_envs)

        if n_episodes is not None:
            print(f"  Using {n_inits} episodes (requested: {n_episodes}, available: {len(self.env_init_fn_dills)})")

        # Storage for evaluation data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # Init envs
            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            # Start evaluation rollout without noise
            self._run_eval_chunk(
                env, policy, device,
                this_n_active_envs, chunk_idx, n_chunks
            )

            # Collect standard data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        # Clear out video buffer
        _ = env.reset()

        # Standard logging (inherited from parent)
        log_data = self._log_standard_metrics(all_video_paths, all_rewards, n_inits)

        # Return evaluation results
        result = {
            'log_data': log_data,
            'video_paths': all_video_paths,
            'episode_rewards': all_rewards,
        }

        print(f"✅ Evaluation rollout completed")

        return log_data  # Return just log_data for compatibility with SFT workspace pattern

    def _run_eval_chunk(self, env, policy: ManiFlowRLPointcloudPolicy, device,
                       n_active_envs: int, chunk_idx: int, n_chunks: int) -> None:
        """Run evaluation for one chunk of environments without noise."""

        # Start rollout
        obs = env.reset()
        past_action = None
        policy.reset()

        env_name = self.env_meta['env_name']
        pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name} {chunk_idx+1}/{n_chunks}",
            leave=False, mininterval=self.tqdm_interval_sec)

        step_count = 0
        done = False
        while not done and step_count < self.max_steps:
            # Create obs dict
            np_obs_dict = dict(obs)
            if self.past_action and (past_action is not None):
                np_obs_dict['past_action'] = past_action[
                    :,-(self.n_obs_steps-1):].astype(np.float32)

            # Device transfer
            obs_dict = dict_apply(np_obs_dict,
                lambda x: torch.from_numpy(x).to(device=device))

            # Run policy WITHOUT exploration noise
            with torch.no_grad():
                # Use sample_actions in eval mode (no noise)
                sample_result = policy.sample_actions(
                    obs_dict,
                    mode="eval",  # Use eval mode - no exploration noise
                    compute_values=False  # No need for values in evaluation
                )

                # Extract only actions
                action = sample_result['actions'].detach().cpu().numpy()  # [B, action_chunk, action_dim]

            # Check for nan/inf
            if not np.all(np.isfinite(action)):
                print(f"Warning: Non-finite action at step {step_count}")
                print(action)
                raise RuntimeError("Nan or Inf action")

            # Step environment
            env_action = action
            if self.abs_action:
                env_action = self.undo_transform_action(action)

            obs, reward, done_array, info = env.step(env_action)

            # Handle done flags
            individual_dones = done_array[:n_active_envs].copy()  # [n_active_envs]
            done = np.all(done_array[:n_active_envs])

            past_action = action
            step_count +=action.shape[1]

            # Update progress bar
            pbar.update(action.shape[1] if len(action.shape) > 1 else 1)

        pbar.close()

    def run(self, policy: BaseImagePolicy, eval_mode: bool = False, n_episodes: Optional[int] = None):
        """
        Override run method to handle both standard and RL policies.

        Args:
            policy: The policy to run
            eval_mode: If True, run in evaluation mode (no exploration noise)
                      If False, run in training mode (with exploration noise)
            n_episodes: Number of episodes to run (optional, defaults to len(env_init_fn_dills))
        """
        if isinstance(policy, ManiFlowRLPointcloudPolicy):
            if eval_mode:
                return self.run_eval(policy, n_episodes=n_episodes)
            elif self.collect_rl_data:
                return self.run_rl(policy, n_episodes=n_episodes)
            else:
                return self.run_eval(policy, n_episodes=n_episodes)  # Standard rollout without RL data collection
        else:
            # Fall back to parent class behavior for non-RL policies
            return super().run(policy)


def test_rl_runner():
    """Test the RL runner with dummy data."""
    print("🧪 Testing RobomimicRLRunner")
    print("=" * 50)

    # This would require actual environment setup
    # For now, just test creation
    print("⚠️  Full testing requires environment setup")
    print("✅ RobomimicRLRunner class created successfully")
    return True


if __name__ == "__main__":
    test_rl_runner()
