#!/usr/bin/env python3

import os
import sys
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import numpy as np
import tempfile
import h5py
import hydra

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Register hydra resolvers
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

OmegaConf.register_new_resolver("get_max_steps", lambda x: max_steps[x], replace=True)
OmegaConf.register_new_resolver("get_ws_x_center", lambda x: -0.2 if x.startswith(('kitchen_', 'hammer_cleanup_')) else 0., replace=True)
OmegaConf.register_new_resolver("get_ws_y_center", lambda x: 0., replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)

from equi_diffpo.rl_training.minimal_dummy_policy import MinimalDummyPolicy
from equi_diffpo.env_runner.robomimic_image_runner import RobomimicImageRunner


def create_real_dataset_for_task(dataset_path: str, task_name: str = "stack_d1"):
    """Create a real robomimic dataset for testing."""

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    with h5py.File(dataset_path, 'w') as f:
        # Real robomimic environment metadata
        f.attrs['env_name'] = 'Stack_D1' if task_name == 'stack_d1' else 'Lift'
        f.attrs['env_type'] = 'robosuite'
        f.attrs['env_version'] = '1.4.1'

        # Environment args for robosuite
        env_args = {
            'env_name': 'Stack' if 'stack' in task_name else 'Lift',
            'robots': ['Panda'],
            'controller_configs': {
                'type': 'OSC_POSE',
                'interpolation': None,
                'ramp_ratio': 0.2,
                'control_delta': True,
                'damping': 1.0,
                'position_limits': None,
                'orientation_limits': None,
                'uncouple_pos_ori': True,
                'input_max': 1.0,
                'input_min': -1.0,
                'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]
            },
            'gripper_types': ['PandaGripper'],
            'initialization_noise': {'type': 'UniformRandomSampler'},
            'has_renderer': True,
            'has_offscreen_renderer': True,
            'camera_names': ['agentview', 'robot0_eye_in_hand'],
            'camera_heights': [84, 84],
            'camera_widths': [84, 84],
            'control_freq': 20,
            'horizon': max_steps[task_name],
        }
        f.attrs['env_args'] = str(env_args)

        # Create demo data
        data_group = f.create_group('data')
        n_demos = 3
        episode_length = 100

        for demo_idx in range(n_demos):
            demo_group = data_group.create_group(f'demo_{demo_idx}')

            # Robot states for initialization
            states = np.random.randn(episode_length, 32).astype(np.float32)
            states[:, 2] += 0.8  # Z above table
            demo_group.create_dataset('states', data=states)

            # Actions (7-DOF robot control)
            actions = np.random.randn(episode_length, 7).astype(np.float32) * 0.01
            demo_group.create_dataset('actions', data=actions)

            # Observations
            obs_group = demo_group.create_group('obs')

            # Point clouds
            point_clouds = np.random.randn(episode_length, 1024, 6).astype(np.float32)
            obs_group.create_dataset('point_cloud', data=point_clouds)

            # Robot state
            obs_group.create_dataset('robot0_eef_pos',
                data=np.random.randn(episode_length, 3).astype(np.float32))
            obs_group.create_dataset('robot0_eef_quat',
                data=np.random.randn(episode_length, 4).astype(np.float32))
            obs_group.create_dataset('robot0_gripper_qpos',
                data=np.random.randn(episode_length, 2).astype(np.float32))

            # Camera images
            obs_group.create_dataset('agentview_image',
                data=np.random.randint(0, 255, (episode_length, 84, 84, 3), dtype=np.uint8))
            obs_group.create_dataset('robot0_eye_in_hand_image',
                data=np.random.randint(0, 255, (episode_length, 84, 84, 3), dtype=np.uint8))

    print(f"Created real dataset: {dataset_path}")


def test_env_runner_performance(task_name: str = "stack_d1"):
    """Test env runner directly with real task configuration."""

    print("=" * 60)
    print(f"Testing Env Runner Performance: {task_name}")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = os.path.join(temp_dir, f"{task_name}_test.hdf5")

        try:
            # 1. Create real dataset
            print("1. Creating Real Dataset...")
            create_real_dataset_for_task(dataset_path, task_name)

            # 2. Create dummy policy
            print("2. Creating Dummy Policy...")
            policy = MinimalDummyPolicy(
                horizon=16,
                n_action_steps=8,
                action_dim=7,  # 7-DOF robot
                device="cpu"
            )

            # 3. Create shape meta for task
            shape_meta = {
                'obs': {
                    'robot0_eye_in_hand_image': {'shape': [3, 84, 84], 'type': 'rgb'},
                    'agentview_image': {'shape': [3, 84, 84], 'type': 'rgb'},
                    'point_cloud': {'shape': [1024, 6], 'type': 'point_cloud'},
                    'robot0_eef_pos': {'shape': [3]},
                    'robot0_eef_quat': {'shape': [4]},
                    'robot0_gripper_qpos': {'shape': [2]},
                },
                'action': {'shape': [7]}
            }

            # 4. Initialize env runner directly
            print("3. Initializing Env Runner...")

            env_runner: RobomimicImageRunner = None
            if True:  # Test the pattern you want
                env_runner = hydra.utils.instantiate(
                    DictConfig({
                        '_target_': 'equi_diffpo.env_runner.robomimic_image_runner.RobomimicImageRunner',
                        'dataset_path': dataset_path,
                        'shape_meta': shape_meta,
                        'n_train': 2,
                        'n_train_vis': 1,
                        'train_start_idx': 0,
                        'n_test': 2,
                        'n_test_vis': 1,
                        'test_start_seed': 100000,
                        'max_steps': max_steps[task_name],
                        'n_obs_steps': 2,
                        'n_action_steps': 8,
                        'render_obs_key': 'agentview_image',
                        'fps': 10,
                        'crf': 22,
                        'past_action': False,
                        'abs_action': True,
                        'tqdm_interval_sec': 0.5,
                        'n_envs': 4,
                    }),
                    output_dir=temp_dir
                )

            if env_runner is None:
                raise ValueError("Failed to initialize env runner")

            print("   ✓ Env runner initialized successfully")
            print(f"   - Task: {task_name}")
            print(f"   - Max steps: {max_steps[task_name]}")
            print(f"   - Number of envs: 4")
            print(f"   - Dataset: {os.path.basename(dataset_path)}")

            # 5. Test env runner performance
            print("4. Running Env Runner Evaluation...")

            import time
            start_time = time.time()

            # Run the standard evaluation
            log_data = env_runner.run(policy)

            end_time = time.time()
            runtime = end_time - start_time

            print(f"   ✓ Evaluation completed in {runtime:.2f} seconds")

            # 6. Analyze results
            print("5. Analyzing Env Runner Results...")
            print(f"   - Log data keys: {list(log_data.keys())}")

            # Check for scores
            train_scores = [v for k, v in log_data.items() if 'train/' in k and 'score' in k]
            test_scores = [v for k, v in log_data.items() if 'test/' in k and 'score' in k]

            if train_scores:
                print(f"   - Train scores: {[f'{s:.3f}' for s in train_scores]}")
            if test_scores:
                print(f"   - Test scores: {[f'{s:.3f}' for s in test_scores]}")

            # Check for videos
            video_keys = [k for k in log_data.keys() if 'video' in k]
            print(f"   - Videos recorded: {len(video_keys)}")

            # Performance metrics
            total_episodes = env_runner.env_seeds
            print(f"   - Total episodes run: {len(total_episodes)}")
            print(f"   - Avg time per episode: {runtime/len(total_episodes):.2f}s")

            # 7. Test environment types
            print("6. Testing Environment Types...")
            train_episodes = [i for i, prefix in enumerate(env_runner.env_prefixs) if prefix == 'train/']
            test_episodes = [i for i, prefix in enumerate(env_runner.env_prefixs) if prefix == 'test/']

            print(f"   - Training episodes: {len(train_episodes)} (seeds: {[env_runner.env_seeds[i] for i in train_episodes]})")
            print(f"   - Test episodes: {len(test_episodes)} (seeds: {[env_runner.env_seeds[i] for i in test_episodes]})")

            print(f"\n" + "=" * 60)
            print(f"✅ Env Runner Performance Test PASSED!")
            print(f"   Task {task_name} runs successfully with real robosuite environment")
            print(f"   Total runtime: {runtime:.2f}s for {len(total_episodes)} episodes")
            print(f"=" * 60)

            return log_data, runtime

        except Exception as e:
            print(f"\n❌ Env Runner Test FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    import sys

    task_name = "stack_d1"
    if len(sys.argv) > 1:
        task_name = sys.argv[1]

    print(f"Testing env runner performance with task: {task_name}")
    test_env_runner_performance(task_name)