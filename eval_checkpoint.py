#!/usr/bin/env python3
"""
Simple evaluation script to load a checkpoint and run evaluation.

Usage:
    python eval_checkpoint.py --checkpoint path/to/checkpoint.ckpt --task_name nut_assembly_d0
"""

import sys
import torch
import hydra
from omegaconf import OmegaConf
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from equi_diffpo.workspace.base_workspace import BaseWorkspace


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--task_name', type=str, default='nut_assembly_d0',
                        help='Task name')
    parser.add_argument('--n_test', type=int, default=20,
                        help='Number of test episodes')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    import dill
    payload = torch.load(checkpoint_path.open('rb'), pickle_module=dill, weights_only=False)

    # Get config from checkpoint
    cfg = payload['cfg']

    # Override task name if needed
    if args.task_name:
        cfg.task_name = args.task_name

    print(f"Task: {cfg.task_name}")
    print(f"Policy: {cfg.policy._target_}")

    # Create workspace from checkpoint
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)

    # Load the model weights
    workspace.load_payload(payload)

    # Get the model and env_runner
    model = workspace.model
    model.eval()
    model.to(args.device)

    # Get env_runner from task config
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=str(checkpoint_path.parent.parent)
    )

    print(f"\nRunning evaluation with {args.n_test} test episodes...")

    # Run evaluation
    with torch.no_grad():
        runner_log = env_runner.run(model)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for key, value in runner_log.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    return runner_log


if __name__ == "__main__":
    main()
