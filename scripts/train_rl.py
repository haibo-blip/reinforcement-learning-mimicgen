#!/usr/bin/env python3
"""
Main script for RL training with ManiFlow policies.
"""

import os
import sys
import hydra
from pathlib import Path
from omegaconf import DictConfig

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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