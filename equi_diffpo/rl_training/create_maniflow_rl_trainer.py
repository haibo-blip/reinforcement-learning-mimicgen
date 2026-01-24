#!/usr/bin/env python3
"""
Factory function to create ManiFlow RL trainer compatible with existing Hydra configs.
"""

import hydra
from omegaconf import OmegaConf
from typing import Dict, Any, Optional
from pathlib import Path
import torch
from .maniflow_ppo_workspace import ManiFlowPPOTrainer, PPOConfig
from .maniflow_advantage_calculator import AdvantageConfig
from equi_diffpo.policy.maniflow.maniflow_pointcloud_rl_policy import ManiFlowRLPointcloudPolicy
from equi_diffpo.env_runner.robomimic_rl_runner import RobomimicRLRunner
from equi_diffpo.model.common.normalizer import LinearNormalizer


def get_normalizer_cache_path(dataset_path: str, n_demo: int = 100) -> Path:
    """Get the cache path for a normalizer based on dataset path."""
    dataset_path = Path(dataset_path)
    cache_dir = dataset_path.parent / "normalizer_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"{dataset_path.stem}_n{n_demo}_normalizer.pt"
    return cache_dir / cache_name


def save_normalizer(normalizer: LinearNormalizer, cache_path: Path) -> None:
    """Save normalizer to cache file."""
    torch.save(normalizer.state_dict(), cache_path)
    print(f"💾 Saved normalizer to: {cache_path}")


def load_normalizer(cache_path: Path) -> Optional[LinearNormalizer]:
    """Load normalizer from cache file if it exists."""
    if not cache_path.exists():
        return None
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(torch.load(cache_path, weights_only=False))
    print(f"⚡ Loaded cached normalizer from: {cache_path}")
    return normalizer


def get_or_create_normalizer(cfg: OmegaConf) -> Optional[LinearNormalizer]:
    """
    Get normalizer from cache or create from dataset.

    This avoids the slow 25+ minute dataset loading by caching the normalizer
    after the first run.
    """
    if not hasattr(cfg.task, 'dataset') or cfg.task.dataset is None:
        return None

    # Get dataset path and n_demo for cache key
    dataset_path = OmegaConf.select(cfg, 'task.dataset.dataset_path') or OmegaConf.select(cfg, 'dataset_path')
    n_demo = OmegaConf.select(cfg, 'task.dataset.n_demo') or OmegaConf.select(cfg, 'n_demo') or 100

    if not dataset_path:
        return None

    # Check for cached normalizer
    cache_path = get_normalizer_cache_path(dataset_path, n_demo)
    normalizer = load_normalizer(cache_path)

    if normalizer is not None:
        return normalizer

    # No cache found - need to load dataset (slow)
    print(f"⏳ No cached normalizer found. Loading dataset (this may take ~25 minutes)...")
    print(f"   After this, subsequent runs will use the cached normalizer.")

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()

    # Save to cache for future runs
    save_normalizer(normalizer, cache_path)

    return normalizer


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
        checkpoint = torch.load(pretrained_policy_path, map_location=device)
        # Handle different checkpoint formats
        if 'state_dicts' in checkpoint and 'model' in checkpoint['state_dicts']:
            # PyTorch Lightning / diffusion policy checkpoint format
            state_dict = checkpoint['state_dicts']['model']
        elif 'policy_state_dict' in checkpoint:
            # Direct policy state dict format
            state_dict = checkpoint['policy_state_dict']
        elif 'state_dict' in checkpoint:
            # Standard PyTorch checkpoint format
            state_dict = checkpoint['state_dict']
        else:
            # Assume the checkpoint itself is the state dict
            state_dict = checkpoint

        # Load with strict=False to allow for missing/extra keys (e.g., value_head)
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️  Missing keys (will be initialized randomly): {missing}")
        if unexpected:
            print(f"⚠️  Unexpected keys (ignored): {unexpected}")

        # Check if normalizer was loaded from checkpoint
        normalizer_keys = [k for k in state_dict.keys() if k.startswith('normalizer.')]
        normalizer_loaded_from_checkpoint = len(normalizer_keys) > 0
    else:
        normalizer_loaded_from_checkpoint = False

    # 3. Create RL-compatible environment runner from config
    # Use RobomimicRLRunner instead of regular RobomimicImageRunner
    env_runner_config = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
    env_runner_config['_target_'] = "equi_diffpo.env_runner.robomimic_rl_runner.RobomimicRLRunner"
    env_runner_config['collect_rl_data'] = True
    env_runner_config = OmegaConf.create(env_runner_config)
    env_runner = hydra.utils.instantiate(env_runner_config, output_dir="./rl_outputs")

    # 4. Set up normalizer - prefer checkpoint, fallback to dataset
    if normalizer_loaded_from_checkpoint:
        print("✅ Using normalizer from checkpoint (skipping dataset load)")
    else:
        # Only load dataset if normalizer not in checkpoint
        normalizer = get_or_create_normalizer(cfg)
        if normalizer is not None:
            policy.set_normalizer(normalizer)
            print("✅ Loaded normalizer from dataset")
        else:
            print("⚠️  No normalizer found - must be set manually")

    # 5. Create PPO config from hydra config - all parameters from rl_training
    rl_config = cfg.get('rl_training', {})

    ppo_config = PPOConfig(
        # Core training parameters
        total_timesteps=rl_config.get('total_timesteps', 1000000),
        num_envs=rl_config.get('num_envs', 14),
        batch_size=rl_config.get('batch_size', 32),

        # PPO hyperparameters
        learning_rate=rl_config.get('learning_rate', 1e-5),
        value_lr=rl_config.get('value_lr', 2e-4),  # Critic LR (~20x higher, like RLinf)
        clip_range=rl_config.get('clip_range', 0.2),
        entropy_coef=rl_config.get('entropy_coef', 0),
        value_coef=rl_config.get('value_coef', 0.5),
        max_grad_norm=rl_config.get('max_grad_norm', 0.5),
        target_kl=rl_config.get('target_kl', 0.03),

        # Learning rate schedule
        lr_schedule=rl_config.get('lr_schedule', 'linear'),
        warmup_steps=rl_config.get('warmup_steps', 10000),

        # Logging and checkpointing
        eval_interval=rl_config.get('eval_interval', 3),
        log_interval=rl_config.get('log_interval', 10),
        save_interval=rl_config.get('save_interval', 100),
        wandb_project=cfg.get('logging', {}).get('project', 'maniflow_rl'),
        wandb_run_name=cfg.get('logging', {}).get('name', 'ppo_training'),

        # Environment parameters (from task config)
        action_chunk_size=cfg.get('n_action_steps', 8),
        obs_chunk_size=cfg.get('n_obs_steps', 2),

        # Critic warmup
        critic_warmup_rollouts=rl_config.get('critic_warmup_rollouts', 0),
    )

    # 6. Create advantage config - parameters from rl_training
    advantage_config = AdvantageConfig(
        gamma=rl_config.get('gamma', 0.99),
        gae_lambda=rl_config.get('gae_lambda', 0.95),
        advantage_type="gae",
        normalize_advantages=True,
    )

    # 7. Create RL trainer (rollout collector is created internally with correct config)
    trainer = ManiFlowPPOTrainer(
        policy=policy,
        env_runner=env_runner,
        config=ppo_config,
        advantage_config=advantage_config,
        device=device,
        use_wandb=True
    )

    print(f"🚀 ManiFlow RL Trainer created from config")
    print(f"  - Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  - Number of environments: {ppo_config.num_envs}")
    print(f"  - Denoising steps: {cfg.policy.get('num_inference_steps', 4)}")
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
