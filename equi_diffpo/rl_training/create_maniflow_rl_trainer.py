#!/usr/bin/env python3
"""
Factory function to create ManiFlow RL trainer compatible with existing Hydra configs.
"""

import hydra
from omegaconf import OmegaConf
from typing import Dict, Any

from .maniflow_ppo_workspace import ManiFlowPPOTrainer, PPOConfig
from .maniflow_advantage_calculator import AdvantageConfig
from .maniflow_rollout_collector import ManiFlowRolloutCollector
from equi_diffpo.policy.maniflow.maniflow_pointcloud_rl_policy import ManiFlowRLPointcloudPolicy
from equi_diffpo.env_runner.robomimic_rl_runner import RobomimicRLRunner


def create_maniflow_rl_trainer_from_config(cfg: OmegaConf,
                                          pretrained_policy_path: str = None,
                                          device: str = "cuda") -> ManiFlowPPOTrainer:
    """
    Create ManiFlow RL trainer from Hydra config (compatible with existing configs).

    Args:
        cfg: Hydra configuration (e.g., from train_maniflow_pointcloud_rl.yaml)
        pretrained_policy_path: Path to pretrained policy checkpoint
        device: Training device

    Returns:
        ManiFlowPPOTrainer: Ready-to-use RL trainer
    """

    # 1. Create RL policy from config
    policy: ManiFlowRLPointcloudPolicy = hydra.utils.instantiate(cfg.policy)

    # 2. Load pretrained weights if provided
    if pretrained_policy_path:
        print(f"📂 Loading pretrained policy from: {pretrained_policy_path}")
        # TODO: Load checkpoint and transfer weights to RL policy
        # checkpoint = torch.load(pretrained_policy_path)
        # policy.load_state_dict(checkpoint['policy_state_dict'])

    # 3. Create RL-compatible environment runner from config
    # Use RobomimicRLRunner instead of regular RobomimicImageRunner
    env_runner_config = cfg.task.env_runner.copy()
    env_runner_config._target_ = "equi_diffpo.env_runner.robomimic_rl_runner.RobomimicRLRunner"
    env_runner_config.collect_rl_data = True
    env_runner = hydra.utils.instantiate(env_runner_config, output_dir="./rl_outputs")

    # 4. Set up normalizer (load from dataset or checkpoint)
    if hasattr(cfg.task, 'dataset'):
        # Load normalizer from dataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()
        policy.set_normalizer(normalizer)
        print("✅ Loaded normalizer from dataset")
    else:
        print("⚠️  No dataset config found - normalizer must be set manually")

    # 5. Create PPO config from hydra config
    rl_config = cfg.get('rl_training', {})
    ppo_trainer_config = rl_config.get('ppo_trainer', {})

    ppo_config = PPOConfig(
        total_timesteps=rl_config.get('num_epochs', 100) * 10000,  # Convert epochs to timesteps
        num_envs=cfg.task.env_runner.get('n_envs', 8),
        num_steps_per_rollout=256,
        batch_size=ppo_trainer_config.get('mini_batch_size', 128),
        num_epochs=ppo_trainer_config.get('ppo_epochs', 4),
        learning_rate=ppo_trainer_config.get('policy_lr', 3e-4),
        clip_range=ppo_trainer_config.get('clip_ratio', 0.2),
        entropy_coef=ppo_trainer_config.get('entropy_coef', 0.01),
        value_coef=ppo_trainer_config.get('value_loss_coef', 0.5),
        max_grad_norm=ppo_trainer_config.get('max_grad_norm', 0.5),
        target_kl=0.01,
        wandb_project=cfg.get('logging', {}).get('project', 'maniflow_rl'),
        wandb_run_name=cfg.get('logging', {}).get('name', 'ppo_training'),
    )

    # 6. Create advantage config
    advantage_config = AdvantageConfig(
        gamma=ppo_trainer_config.get('gamma', 0.99),
        gae_lambda=ppo_trainer_config.get('gae_lambda', 0.95),
        advantage_type="gae",
        normalize_advantages=True,
    )

    # 7. Create rollout collector
    rollout_collector = ManiFlowRolloutCollector(
        policy=policy,
        env_runner=env_runner,
        max_steps_per_episode=cfg.task.env_runner.get('max_steps', 400),
        action_chunk_size=cfg.get('n_action_steps', 8),
        obs_chunk_size=cfg.get('n_obs_steps', 2),
        device=device
    )

    # 8. Create RL trainer
    trainer = ManiFlowPPOTrainer(
        policy=policy,
        env_runner=env_runner,
        config=ppo_config,
        advantage_config=advantage_config,
        device=device,
        use_wandb=True
    )

    # Override rollout collector to use the configured one
    trainer.rollout_collector = rollout_collector

    print(f"🚀 ManiFlow RL Trainer created from config")
    print(f"  - Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  - Number of environments: {ppo_config.num_envs}")
    print(f"  - Action chunk size: {cfg.get('n_action_steps', 8)}")
    print(f"  - Observation steps: {cfg.get('n_obs_steps', 2)}")

    return trainer


def create_maniflow_rl_trainer_simple(
        task_name: str = "stack_d1",
        dataset_path: str = None,
        pretrained_policy_path: str = None,
        device: str = "cuda",
        **kwargs) -> ManiFlowPPOTrainer:
    """
    Simple factory function with minimal configuration.

    Args:
        task_name: Name of the task (e.g., "stack_d1")
        dataset_path: Path to dataset HDF5 file
        pretrained_policy_path: Path to pretrained policy checkpoint
        device: Training device
        **kwargs: Additional configuration overrides

    Returns:
        ManiFlowPPOTrainer: Ready-to-use RL trainer
    """

    # Create minimal config structure
    cfg = OmegaConf.create({
        'task_name': task_name,
        'n_action_steps': kwargs.get('n_action_steps', 8),
        'n_obs_steps': kwargs.get('n_obs_steps', 2),
        'horizon': kwargs.get('horizon', 16),

        'policy': {
            '_target_': 'equi_diffpo.policy.maniflow.maniflow_pointcloud_rl_policy.ManiFlowRLPointcloudPolicy',
            'horizon': kwargs.get('horizon', 16),
            'n_action_steps': kwargs.get('n_action_steps', 8),
            'n_obs_steps': kwargs.get('n_obs_steps', 2),
            'num_inference_steps': kwargs.get('num_inference_steps', 10),
            'noise_method': kwargs.get('noise_method', 'flow_sde'),
            'n_layer': kwargs.get('n_layer', 4),
            'n_head': kwargs.get('n_head', 8),
            'n_emb': kwargs.get('n_emb', 256),
            'encoder_output_dim': kwargs.get('encoder_output_dim', 128),
            'visual_cond_len': kwargs.get('visual_cond_len', 256),
            'add_value_head': True,
            'shape_meta': {
                'obs': {
                    'robot0_eye_in_hand_image': {'shape': [3, 84, 84], 'type': 'rgb'},
                    'point_cloud': {'shape': [1024, 6], 'type': 'point_cloud'},
                    'robot0_eef_pos': {'shape': [3]},
                    'robot0_eef_quat': {'shape': [4]},
                    'robot0_gripper_qpos': {'shape': [2]},
                },
                'action': {'shape': [10]}
            }
        },

        'task': {
            'env_runner': {
                '_target_': 'equi_diffpo.env_runner.robomimic_rl_runner.RobomimicRLRunner',
                'dataset_path': dataset_path or f"data/robomimic/datasets/{task_name}/{task_name}_voxel_abs.hdf5",
                'n_train': 6,
                'n_test': 20,
                'max_steps': kwargs.get('max_steps', 400),
                'n_obs_steps': kwargs.get('n_obs_steps', 2),
                'n_action_steps': kwargs.get('n_action_steps', 8),
                'n_envs': kwargs.get('n_envs', 8),
                'collect_rl_data': True,
            },
            'dataset': {
                '_target_': 'equi_diffpo.dataset.robomimic_replay_point_cloud_dataset.RobomimicReplayPointCloudDataset',
                'dataset_path': dataset_path or f"data/robomimic/datasets/{task_name}/{task_name}_voxel_abs.hdf5",
                'horizon': kwargs.get('horizon', 16),
                'n_demo': kwargs.get('n_demo', 100),
            } if dataset_path else None
        },

        'rl_training': {
            'num_epochs': kwargs.get('num_epochs', 100),
            'ppo_trainer': {
                'clip_ratio': kwargs.get('clip_ratio', 0.2),
                'entropy_coef': kwargs.get('entropy_coef', 0.01),
                'value_loss_coef': kwargs.get('value_loss_coef', 0.5),
                'gamma': kwargs.get('gamma', 0.99),
                'gae_lambda': kwargs.get('gae_lambda', 0.95),
                'policy_lr': kwargs.get('learning_rate', 3e-4),
                'mini_batch_size': kwargs.get('batch_size', 128),
                'ppo_epochs': kwargs.get('ppo_epochs', 4),
            }
        },

        'logging': {
            'project': kwargs.get('wandb_project', 'maniflow_rl'),
            'name': kwargs.get('wandb_run_name', f'ppo_{task_name}'),
        }
    })

    return create_maniflow_rl_trainer_from_config(
        cfg=cfg,
        pretrained_policy_path=pretrained_policy_path,
        device=device
    )


# Example usage for testing
def test_rl_trainer_creation():
    """Test creating RL trainer with simple config."""
    print("🧪 Testing RL Trainer Creation")
    print("=" * 50)

    try:
        trainer = create_maniflow_rl_trainer_simple(
            task_name="stack_d1",
            device="cpu",  # Use CPU for testing
            n_envs=2,      # Small number for testing
            num_epochs=5,  # Short training for testing
            n_layer=2,     # Smaller model for testing
            n_emb=64,
        )

        print("✅ RL trainer created successfully!")
        print(f"  - Device: {trainer.device}")
        print(f"  - Policy type: {type(trainer.policy).__name__}")
        print(f"  - Environment runner type: {type(trainer.env_runner).__name__}")

        return True

    except Exception as e:
        print(f"❌ RL trainer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_rl_trainer_creation()