#!/usr/bin/env python3
"""
ManiFlow RL Training Script
Main entry point for RL fine-tuning of ManiFlow policies using PPO.

Usage:
    python train_maniflow_rl.py --config-name=train_maniflow_pointcloud_rl
    python train_maniflow_rl.py --config-path=configs --config-name=my_rl_config
    python train_maniflow_rl.py task.dataset_path=/path/to/dataset policy.checkpoint=/path/to/checkpoint.ckpt
"""

import os
import sys
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import logging

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from equi_diffpo.rl_training.create_maniflow_rl_trainer import create_maniflow_rl_trainer_from_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_wandb(cfg: DictConfig):
    """Initialize Weights & Biases logging if configured."""
    if cfg.get('use_wandb', True):
        wandb_config = {
            'project': cfg.get('wandb_project', 'maniflow_rl'),
            'name': cfg.get('wandb_run_name', f'rl_training_{cfg.task_name}'),
            'config': OmegaConf.to_container(cfg, resolve=True),
            'save_code': True,
        }

        # Add tags if specified
        if cfg.get('wandb_tags'):
            wandb_config['tags'] = cfg.wandb_tags

        wandb.init(**wandb_config)
        logger.info(f"🚀 Initialized W&B project: {wandb_config['project']}")
        logger.info(f"📊 Run name: {wandb_config['name']}")
    else:
        logger.info("📝 W&B logging disabled")


def validate_config(cfg: DictConfig):
    """Validate the training configuration."""
    required_keys = [
        'policy',
        'task.env_runner',
        'training'
    ]

    for key in required_keys:
        if not OmegaConf.select(cfg, key):
            raise ValueError(f"Missing required config key: {key}")

    # Check if dataset path exists
    dataset_path = OmegaConf.select(cfg, 'task.dataset_path')
    if dataset_path and not Path(dataset_path).exists():
        logger.warning(f"⚠️  Dataset path does not exist: {dataset_path}")

    # Check if checkpoint path exists
    checkpoint_path = OmegaConf.select(cfg, 'policy.checkpoint')
    if checkpoint_path and not Path(checkpoint_path).exists():
        logger.warning(f"⚠️  Checkpoint path does not exist: {checkpoint_path}")

    logger.info("✅ Configuration validation passed")


def print_config_summary(cfg: DictConfig):
    """Print a summary of the training configuration."""
    print("=" * 80)
    print("🎯 MANIFLOW RL TRAINING CONFIGURATION")
    print("=" * 80)

    # Environment
    env_name = OmegaConf.select(cfg, 'task.env_runner.env_meta.env_name') or 'Unknown'
    dataset_path = OmegaConf.select(cfg, 'task.dataset_path') or 'Not specified'

    print(f"📦 Environment: {env_name}")
    print(f"📊 Dataset: {dataset_path}")

    # Policy
    policy_checkpoint = OmegaConf.select(cfg, 'policy.checkpoint') or 'None (training from scratch)'
    horizon = OmegaConf.select(cfg, 'policy.horizon') or 'Unknown'
    n_action_steps = OmegaConf.select(cfg, 'policy.n_action_steps') or 'Unknown'

    print(f"🧠 Policy checkpoint: {policy_checkpoint}")
    print(f"⏱️  Horizon: {horizon}")
    print(f"🎬 Action steps: {n_action_steps}")

    # Training parameters
    training = cfg.get('training', {})
    total_timesteps = training.get('total_timesteps', 1000000)
    num_envs = training.get('num_envs', 8)
    learning_rate = training.get('learning_rate', 3e-4)
    batch_size = training.get('batch_size', 512)

    print(f"🔄 Total timesteps: {total_timesteps:,}")
    print(f"🌍 Number of environments: {num_envs}")
    print(f"📈 Learning rate: {learning_rate}")
    print(f"📦 Batch size: {batch_size}")

    # Output directories
    output_dir = OmegaConf.select(cfg, 'output_dir') or './outputs'
    print(f"📁 Output directory: {output_dir}")

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🔥 Device: {device} ({gpu_count}x {gpu_name})")
    else:
        print(f"💻 Device: {device}")

    print("=" * 80)


def setup_output_directories(cfg: DictConfig):
    """Create output directories for training artifacts."""
    output_dir = Path(cfg.get('output_dir', './outputs'))

    # Create subdirectories
    dirs_to_create = [
        output_dir,
        output_dir / 'checkpoints',
        output_dir / 'logs',
        output_dir / 'videos',
        output_dir / 'configs'
    ]

    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Save the config for reproducibility
    config_path = output_dir / 'configs' / 'training_config.yaml'
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)

    logger.info(f"📁 Created output directories in: {output_dir}")
    logger.info(f"💾 Saved config to: {config_path}")

    return output_dir


@hydra.main(version_base=None, config_path="config", config_name="train_maniflow_pointcloud_rl")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    try:
        # Setup
        print_config_summary(cfg)
        validate_config(cfg)
        output_dir = setup_output_directories(cfg)

        # Initialize wandb if configured
        setup_wandb(cfg)

        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            logger.info(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("💻 Using CPU (GPU not available)")

        # Create the RL trainer from config
        logger.info("🏗️  Creating ManiFlow RL trainer...")
        pretrained_policy_path = OmegaConf.select(cfg, 'policy.checkpoint')

        trainer = create_maniflow_rl_trainer_from_config(
            cfg=cfg,
            pretrained_policy_path=pretrained_policy_path,
            device=device
        )

        logger.info("✅ RL trainer created successfully!")

        # Start training
        logger.info("🚀 Starting RL training...")
        print("\n" + "=" * 80)
        print("🎯 BEGINNING MANIFLOW RL TRAINING")
        print("=" * 80)

        trainer.train()

        print("\n" + "=" * 80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        # Final checkpoint save
        final_checkpoint_path = output_dir / 'checkpoints' / 'final_policy.ckpt'
        trainer._save_checkpoint(final=True)
        logger.info(f"💾 Final checkpoint saved to: {final_checkpoint_path}")

        # Close wandb
        if cfg.get('use_wandb', True):
            wandb.finish()

    except KeyboardInterrupt:
        logger.info("⚠️  Training interrupted by user")
        if cfg.get('use_wandb', True):
            wandb.finish()
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

        if cfg.get('use_wandb', True):
            wandb.finish()
        sys.exit(1)


if __name__ == "__main__":
    # Example usage and help
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print(__doc__)
        print("\nExample commands:")
        print("  # Use default config")
        print("  python train_maniflow_rl.py")
        print("")
        print("  # Specify different config")
        print("  python train_maniflow_rl.py --config-name=my_custom_rl_config")
        print("")
        print("  # Override specific parameters")
        print("  python train_maniflow_rl.py training.learning_rate=1e-4 training.num_envs=16")
        print("")
        print("  # Use specific dataset and checkpoint")
        print("  python train_maniflow_rl.py task.dataset_path=/data/robomimic/lift.hdf5 policy.checkpoint=./checkpoints/pretrained.ckpt")
        print("")
        print("  # Disable wandb logging")
        print("  python train_maniflow_rl.py use_wandb=false")
        sys.exit(0)

    main()
