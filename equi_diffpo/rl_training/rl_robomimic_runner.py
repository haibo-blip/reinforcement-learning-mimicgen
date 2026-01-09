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

from equi_diffpo.env_runner.robomimic_image_runner import RobomimicImageRunner
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.common.pytorch_util import dict_apply


class RLRobomimicRunner(RobomimicImageRunner):
    """
    RL-specific Robomimic runner that extends the base runner to collect
    episode data for PPO training instead of just evaluation metrics.
    """

    def __init__(self, *args, collect_episode_data=True, **kwargs):
        """
        Args:
            collect_episode_data: Whether to collect full episode trajectories
        """
        super().__init__(*args, **kwargs)
        self.collect_episode_data = collect_episode_data

    def run_rl_collection(self, policy: BaseImagePolicy) -> Tuple[Dict, Optional[Dict]]:
        """
        Run policy rollouts and collect episode data for RL training.

        This follows the same pattern as the base runner's `run` method but
        collects full episode trajectories instead of just final rewards.

        Args:
            policy: Policy to roll out

        Returns:
            log_data: Logging metrics (same as base runner)
            episode_data: Full episode trajectories for RL training (if collect_episode_data=True)
        """
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout (same as base runner)
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data (extended for RL)
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        # RL-specific data collection
        if self.collect_episode_data:
            all_episode_obs = [None] * n_inits
            all_episode_actions = [None] * n_inits
            all_episode_fixed_noise = [None] * n_inits
            all_episode_rewards = [None] * n_inits
            all_episode_dones = [None] * n_inits
            all_episode_lengths = [None] * n_inits

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

            # init envs (same as base runner)
            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            # start rollout (same as base runner)
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"RL {env_name} {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)

            # RL-specific episode data collection
            if self.collect_episode_data:
                chunk_episode_obs = collections.defaultdict(list)
                chunk_episode_actions = []
                chunk_episode_fixed_noise = []
                chunk_episode_rewards = []
                chunk_episode_dones = []

            step_count = 0
            done = False
            while not done and step_count < self.max_steps:
                # create obs dict (same as base runner)
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)

                # Store observations for RL
                if self.collect_episode_data:
                    for key, value in np_obs_dict.items():
                        chunk_episode_obs[key].append(value[:this_n_active_envs])

                # device transfer (same as base runner)
                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x[:this_n_active_envs]).to(device=device))

                # Generate fixed noise for flow SDE
                batch_size = obs_dict['point_cloud'].shape[0] if 'point_cloud' in obs_dict else obs_dict[list(obs_dict.keys())[0]].shape[0]
                action_shape = (batch_size, policy.horizon, policy.n_action_steps)
                fixed_noise = torch.randn(action_shape, device=device)

                # run policy (extended for flow SDE)
                with torch.no_grad():
                    try:
                        action_dict = policy.predict_action(obs_dict, noise=fixed_noise)
                    except TypeError:
                        # Fallback if policy doesn't support noise parameter
                        action_dict = policy.predict_action(obs_dict)

                # device_transfer (same as base runner)
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                # Store actions and fixed noise for RL
                if self.collect_episode_data:
                    chunk_episode_actions.append(action)
                    chunk_episode_fixed_noise.append(fixed_noise.detach().cpu().numpy())

                # step env (same as base runner)
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)

                # Store rewards and dones for RL
                if self.collect_episode_data:
                    chunk_episode_rewards.append(reward[:this_n_active_envs])
                    chunk_episode_dones.append(done[:this_n_active_envs])

                done = np.all(done[:this_n_active_envs])
                past_action = action
                step_count += 1

                # update pbar (same as base runner)
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round (same as base runner)
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

            # Store episode data for RL training
            if self.collect_episode_data:
                for env_idx in range(this_n_active_envs):
                    global_idx = start + env_idx

                    # Collect episode data
                    episode_obs = {}
                    for key in chunk_episode_obs.keys():
                        episode_obs[key] = np.array([obs[env_idx] for obs in chunk_episode_obs[key]])

                    episode_actions = np.array([act[env_idx] for act in chunk_episode_actions])
                    episode_fixed_noise = np.array([noise[env_idx] for noise in chunk_episode_fixed_noise])
                    episode_rewards = np.array([rew[env_idx] for rew in chunk_episode_rewards])
                    episode_dones = np.array([done[env_idx] for done in chunk_episode_dones])

                    all_episode_obs[global_idx] = episode_obs
                    all_episode_actions[global_idx] = episode_actions
                    all_episode_fixed_noise[global_idx] = episode_fixed_noise
                    all_episode_rewards[global_idx] = episode_rewards
                    all_episode_dones[global_idx] = episode_dones
                    all_episode_lengths[global_idx] = len(episode_actions)

        # clear out video buffer (same as base runner)
        _ = env.reset()

        # log (same as base runner)
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics (same as base runner)
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
            self.max_rewards[prefix] = max(self.max_rewards[prefix], value)
            log_data[prefix+'max_score'] = self.max_rewards[prefix]

        # Prepare episode data for RL training
        episode_data = None
        if self.collect_episode_data:
            episode_data = {
                'obs': all_episode_obs,
                'actions': all_episode_actions,
                'fixed_noise': all_episode_fixed_noise,
                'rewards': all_episode_rewards,
                'dones': all_episode_dones,
                'episode_lengths': all_episode_lengths,
            }

        return log_data, episode_data

    def run(self, policy: BaseImagePolicy):
        """
        Standard run method that maintains compatibility with base runner.
        This just calls run_rl_collection without episode data collection.
        """
        original_collect_setting = self.collect_episode_data
        self.collect_episode_data = False
        log_data, _ = self.run_rl_collection(policy)
        self.collect_episode_data = original_collect_setting
        return log_data