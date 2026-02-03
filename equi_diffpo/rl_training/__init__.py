"""
ManiFlow RL Training Package

Complete RL training pipeline for ManiFlow pointcloud policy following RLinf pattern.
"""

from .maniflow_rollout_collector import (
    ManiFlowRolloutCollector,
    ManiFlowRolloutStep,
    ManiFlowRolloutBatch,
    ManiFlowDummyEnvRunner
)

from .maniflow_advantage_calculator import (
    ManiFlowAdvantageCalculator,
    AdvantageConfig
)

from .maniflow_ppo_workspace import (
    ManiFlowPPOTrainer,
    PPOConfig,
    create_maniflow_ppo_trainer
)

from .create_maniflow_rl_trainer import (
    create_maniflow_rl_trainer_from_config,
    create_maniflow_rl_trainer_simple
)

# Import RL runner for direct usage
try:
    from equi_diffpo.env_runner.robomimic_rl_runner import RobomimicRLRunner
    _has_rl_runner = True
except ImportError:
    _has_rl_runner = False

__all__ = [
    # Rollout collection
    'ManiFlowRolloutCollector',
    'ManiFlowRolloutStep',
    'ManiFlowRolloutBatch',
    'ManiFlowDummyEnvRunner',

    # Advantage calculation
    'ManiFlowAdvantageCalculator',
    'AdvantageConfig',

    # PPO training
    'ManiFlowPPOTrainer',
    'PPOConfig',
    'create_maniflow_ppo_trainer',

    # Factory functions
    'create_maniflow_rl_trainer_from_config',
    'create_maniflow_rl_trainer_simple',
]

# Add RobomimicRLRunner to exports if available
if _has_rl_runner:
    __all__.append('RobomimicRLRunner')