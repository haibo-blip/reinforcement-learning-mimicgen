#!/usr/bin/env python3
"""
Main script for RL training with ManiFlow policies.
"""

import os
import sys
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Register hydra resolvers (from train.py)
max_steps = {
    'stack_d1': 400,
    'stack_three_d1': 400,
    'square_d2': 400,
    'threading_d2': 400,
    'coffee_d2': 400,
    'three_piece_assembly_d2': 500,
    'hammer_cleanup_d1': 500,
    'mug_cleanup_d1': 500,
    'kitchen_d1': 800,
    'nut_assembly_d0': 500,
    'pick_place_d0': 1000,
    'coffee_preparation_d1': 800,
    'tool_hang': 700,
    'can': 400,
    'lift': 400,
    'square': 400,
}

def get_ws_x_center(task_name):
    if task_name.startswith('kitchen_') or task_name.startswith('hammer_cleanup_'):
        return -0.2
    else:
        return 0.

def get_ws_y_center(task_name):
    return 0.

OmegaConf.register_new_resolver("get_max_steps", lambda x: max_steps[x], replace=True)
OmegaConf.register_new_resolver("get_ws_x_center", get_ws_x_center, replace=True)
OmegaConf.register_new_resolver("get_ws_y_center", get_ws_y_center, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)

from equi_diffpo.rl_training.rl_workspace import RLTrainingWorkspace


@hydra.main(version_base=None, config_path="../equi_diffpo/config", config_name="train_maniflow_pointcloud_rl")
def main(cfg: DictConfig):
    """Main RL training entry point."""

    print("=" * 80)
    print("ManiFlow RL Training")
    print("=" * 80)

    # Create and run workspace
    workspace = RLTrainingWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()