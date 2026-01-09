import os
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np

from equi_diffpo.workspace.base_workspace import BaseWorkspace
from equi_diffpo.rl_training.rl_robomimic_runner import RLRobomimicRunner
from equi_diffpo.rl_training.ppo_trainer import PPOTrainer
from equi_diffpo.common.pytorch_util import dict_apply


class RLTrainingWorkspace(BaseWorkspace):
    """RL training workspace for ManiFlow policies."""

    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: DictConfig, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # Set random seeds
        torch.manual_seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)

        # Setup device
        device = torch.device(cfg.training.device)
        self.device = device

        # Load pretrained policy
        if 'pretrained_policy_path' in cfg:
            print(f"Loading pretrained policy from: {cfg.pretrained_policy_path}")
            self.policy = self._load_pretrained_policy(cfg.pretrained_policy_path)
        else:
            # Initialize policy from config (for testing without pretrained model)
            self.policy = hydra.utils.instantiate(cfg.policy)

        self.policy.to(device)
        self.policy.eval()  # Start in eval mode for rollouts

        # Setup normalizer
        if hasattr(self.policy, 'normalizer'):
            self.normalizer = self.policy.normalizer
        else:
            # Create dummy normalizer if not available
            from equi_diffpo.model.common.normalizer import LinearNormalizer
            self.normalizer = LinearNormalizer()

        # Setup RL environment runner (extends base runner)
        env_runner_cfg = cfg.task.env_runner.copy()
        # Override target to use RL-specific runner
        env_runner_cfg._target_ = "equi_diffpo.rl_training.rl_robomimic_runner.RLRobomimicRunner"

        self.rl_runner = hydra.utils.instantiate(
            env_runner_cfg,
            output_dir=self.output_dir,
            collect_episode_data=True
        )

        self.ppo_trainer = PPOTrainer(
            policy=self.policy,
            normalizer=self.normalizer,
            device=device,
            **cfg.rl_training.ppo_trainer
        )

        # Setup optimizers (will be created in PPO trainer if not provided)
        self.policy_optimizer = self.ppo_trainer.policy_optimizer
        self.critic_optimizer = self.ppo_trainer.critic_optimizer

        # Training state
        self.global_step = 0
        self.epoch = 0

    def _load_pretrained_policy(self, checkpoint_path):
        """Load pretrained policy from checkpoint."""
        checkpoint_path = Path(checkpoint_path).expanduser()

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract policy config and state dict
        if 'cfg' in checkpoint:
            policy_cfg = checkpoint['cfg'].policy
            policy = hydra.utils.instantiate(policy_cfg)
        else:
            # Fallback: use current config
            policy = hydra.utils.instantiate(self.cfg.policy)

        # Load state dict
        if 'state_dicts' in checkpoint:
            policy.load_state_dict(checkpoint['state_dicts']['ema_model'])
        elif 'model' in checkpoint:
            policy.load_state_dict(checkpoint['model'])
        else:
            raise KeyError("Could not find model state dict in checkpoint")

        return policy

    def run(self):
        """Run RL training loop."""
        cfg = self.cfg

        # Initialize wandb logging
        if cfg.logging.mode != 'disabled':
            wandb.init(
                project=cfg.logging.project,
                name=cfg.logging.name,
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=cfg.logging.mode,
                tags=cfg.logging.tags,
            )

        # Training loop
        for epoch in range(cfg.rl_training.num_epochs):
            self.epoch = epoch

            # Collect rollout batch using RL runner
            print(f"Epoch {epoch}: Collecting rollouts...")
            self.policy.eval()  # Set to eval mode for rollout

            rollout_metrics, episode_data = self.rl_runner.run_rl_collection(self.policy)

            # Convert episode data to batch format for PPO
            batch_data = self._convert_episode_data_to_batch(episode_data)

            # Log rollout metrics
            if cfg.logging.mode != 'disabled':
                wandb.log(rollout_metrics, step=self.global_step)

            print(f"Collected {len(episode_data['episode_lengths'])} episodes, "
                  f"mean reward: {rollout_metrics.get('train/mean_score', rollout_metrics.get('test/mean_score', 0.0)):.3f}")

            # Update policy with PPO
            print(f"Epoch {epoch}: Updating policy...")
            self.policy.train()  # Set to train mode for updates

            ppo_metrics = self.ppo_trainer.update_policy_and_critic(batch_data)

            # Log training metrics
            if cfg.logging.mode != 'disabled':
                ppo_metrics_prefixed = {f'ppo/{k}': v for k, v in ppo_metrics.items()}
                wandb.log(ppo_metrics_prefixed, step=self.global_step)

            print(f"PPO update - Policy loss: {ppo_metrics['policy_loss']:.4f}, "
                  f"Value loss: {ppo_metrics['value_loss']:.4f}")

            # Save checkpoint
            if epoch % cfg.rl_training.checkpoint_every == 0:
                self.save_checkpoint()

            # Validation rollout (optional)
            if hasattr(cfg.rl_training, 'val_every') and epoch % cfg.rl_training.val_every == 0:
                self.run_validation()

            self.global_step += 1

        # Final checkpoint
        self.save_checkpoint()

        if cfg.logging.mode != 'disabled':
            wandb.finish()

    def _convert_episode_data_to_batch(self, episode_data):
        """Convert episode data from RL runner to batch format for PPO."""
        import torch

        n_episodes = len(episode_data['episode_lengths'])
        max_length = max(episode_data['episode_lengths'])

        # Initialize batch tensors
        batch_data = {
            'obs': {},
            'actions': None,
            'fixed_noise': None,
            'rewards': None,
            'dones': None,
            'episode_lengths': torch.tensor(episode_data['episode_lengths'])
        }

        # Get observation keys from first episode
        obs_keys = list(episode_data['obs'][0].keys())

        # Process observations
        for key in obs_keys:
            first_obs_shape = episode_data['obs'][0][key].shape[1:]  # Skip time dimension
            batch_obs = torch.zeros(n_episodes, max_length, *first_obs_shape, dtype=torch.float32)

            for ep_idx in range(n_episodes):
                ep_length = episode_data['episode_lengths'][ep_idx]
                batch_obs[ep_idx, :ep_length] = torch.from_numpy(episode_data['obs'][ep_idx][key])

            batch_data['obs'][key] = batch_obs

        # Process actions
        action_dim = episode_data['actions'][0].shape[-1]
        batch_actions = torch.zeros(n_episodes, max_length, action_dim, dtype=torch.float32)

        for ep_idx in range(n_episodes):
            ep_length = episode_data['episode_lengths'][ep_idx]
            batch_actions[ep_idx, :ep_length] = torch.from_numpy(episode_data['actions'][ep_idx])

        batch_data['actions'] = batch_actions

        # Process fixed noise
        noise_shape = episode_data['fixed_noise'][0].shape[1:]  # Skip time dimension
        batch_fixed_noise = torch.zeros(n_episodes, max_length, *noise_shape, dtype=torch.float32)

        for ep_idx in range(n_episodes):
            ep_length = episode_data['episode_lengths'][ep_idx]
            batch_fixed_noise[ep_idx, :ep_length] = torch.from_numpy(episode_data['fixed_noise'][ep_idx])

        batch_data['fixed_noise'] = batch_fixed_noise

        # Process rewards
        batch_rewards = torch.zeros(n_episodes, max_length, dtype=torch.float32)

        for ep_idx in range(n_episodes):
            ep_length = episode_data['episode_lengths'][ep_idx]
            batch_rewards[ep_idx, :ep_length] = torch.from_numpy(episode_data['rewards'][ep_idx])

        batch_data['rewards'] = batch_rewards

        # Process dones
        batch_dones = torch.zeros(n_episodes, max_length, dtype=torch.bool)

        for ep_idx in range(n_episodes):
            ep_length = episode_data['episode_lengths'][ep_idx]
            batch_dones[ep_idx, :ep_length] = torch.from_numpy(episode_data['dones'][ep_idx])

        batch_data['dones'] = batch_dones

        return batch_data

    def run_validation(self):
        """Run validation rollouts."""
        print("Running validation...")
        self.policy.eval()

        # Collect validation rollouts using RL runner
        val_metrics, _ = self.rl_runner.run_rl_collection(self.policy)

        # Add validation prefix to metrics
        val_metrics_prefixed = {f'val/{k}': v for k, v in val_metrics.items()}

        if self.cfg.logging.mode != 'disabled':
            wandb.log(val_metrics_prefixed, step=self.global_step)

        mean_score = val_metrics.get('train/mean_score', val_metrics.get('test/mean_score', 0.0))
        print(f"Validation - Mean reward: {mean_score:.3f}")

    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.ppo_trainer.critic.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'cfg': self.cfg,
        }

        checkpoint_path = Path(self.output_dir) / 'checkpoints' / f'epoch_{self.epoch:04d}.ckpt'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Also save latest checkpoint
        latest_path = Path(self.output_dir) / 'checkpoints' / 'latest.ckpt'
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.ppo_trainer.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        print(f"Loaded checkpoint from epoch {self.epoch}")