#!/usr/bin/env python3

import os
import sys
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import numpy as np
import tempfile
import h5py

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

from equi_diffpo.rl_training.minimal_dummy_policy import MinimalDummyPolicy
from equi_diffpo.rl_training.rl_robomimic_runner import RLRobomimicRunner


def create_minimal_dataset(dataset_path: str, task_name: str = "stack_d1", n_demos: int = 2, episode_length: int = 50):
    """Create a minimal real HDF5 dataset that robomimic can load."""

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    with h5py.File(dataset_path, 'w') as f:
        # Environment metadata (required by robomimic)
        f.attrs['env_name'] = 'Stack_D1'  # Robomimic environment name
        f.attrs['env_type'] = 'robosuite'
        f.attrs['env_version'] = '1.4.1'

        # Environment args (required)
        env_args = {
            'env_name': 'Stack',
            'robots': ['Panda'],
            'controller_configs': {
                'type': 'OSC_POSE',
                'interpolation': None,
                'ramp_ratio': 0.2,
                'control_delta': True,  # Use delta control
                'damping': 1.0,
                'damping_limits': [0, 10],
                'position_limits': None,
                'orientation_limits': None,
                'uncouple_pos_ori': True,
                'input_max': 1.0,
                'input_min': -1.0,
                'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]
            },
            'gripper_types': ['PandaGripper'],
            'initialization_noise': {
                'type': 'UniformRandomSampler',
                'x': [-0.02, 0.02],
                'y': [-0.02, 0.02],
                'z': [0.0, 0.01],
                'yaw': [-5.0, 5.0]
            },
            'table_full_size': [0.8, 0.8, 0.05],
            'table_friction': [1.0, 5e-3, 1e-4],
            'use_camera_obs': True,
            'use_object_obs': False,  # Disable for simplicity
            'reward_shaping': True,
            'placement_initializer': {
                'type': 'UniformRandomSampler',
                'x': [-0.08, 0.08],
                'y': [-0.08, 0.08],
                'z_rot': [0.0, 6.28318531],
                'reference': [0, 0, 0.8]
            },
            'has_renderer': True,
            'has_offscreen_renderer': True,
            'render_camera': 'frontview',
            'render_collision_mesh': False,
            'render_visual_mesh': True,
            'render_gpu_device_id': -1,
            'control_freq': 20,
            'horizon': episode_length,
            'ignore_done': True,
            'hard_reset': False,
            'camera_names': ['agentview', 'robot0_eye_in_hand'],
            'camera_heights': [84, 84],
            'camera_widths': [84, 84],
            'camera_depths': [False, False]
        }

        f.attrs['env_args'] = str(env_args)  # Store as string

        # Create data group
        data_group = f.create_group('data')

        for demo_idx in range(n_demos):
            demo_group = data_group.create_group(f'demo_{demo_idx}')

            # Robot states (for initial state sampling)
            # Format: [x, y, z, qx, qy, qz, qw, gripper_pos1, gripper_pos2, ...]
            states = np.random.randn(episode_length, 32).astype(np.float32)  # 32D state vector
            states[:, 2] += 0.8  # Z-position above table
            states[:, 6] = 1.0   # W quaternion component
            demo_group.create_dataset('states', data=states)

            # Actions (7-DOF: 3 pos + 3 rot + 1 gripper)
            actions = np.random.randn(episode_length, 7).astype(np.float32) * 0.01  # Small delta actions
            demo_group.create_dataset('actions', data=actions)

            # Rewards
            rewards = np.random.exponential(0.1, episode_length).astype(np.float32)
            demo_group.create_dataset('rewards', data=rewards)

            # Dones
            dones = np.zeros(episode_length, dtype=bool)
            dones[-1] = True  # Last step is done
            demo_group.create_dataset('dones', data=dones)

            # Observations
            obs_group = demo_group.create_group('obs')

            # Point cloud observations
            point_clouds = np.random.randn(episode_length, 1024, 6).astype(np.float32)
            obs_group.create_dataset('point_cloud', data=point_clouds)

            # Robot state observations
            eef_pos = np.random.randn(episode_length, 3).astype(np.float32)
            eef_pos[:, 2] += 0.8  # Above table
            obs_group.create_dataset('robot0_eef_pos', data=eef_pos)

            eef_quat = np.random.randn(episode_length, 4).astype(np.float32)
            eef_quat[:, 3] = 1.0  # Normalize quaternion
            obs_group.create_dataset('robot0_eef_quat', data=eef_quat)

            gripper_qpos = np.random.randn(episode_length, 2).astype(np.float32) * 0.1
            obs_group.create_dataset('robot0_gripper_qpos', data=gripper_qpos)

            # Camera observations (for env runner compatibility)
            agentview_images = np.random.randint(0, 255, (episode_length, 84, 84, 3), dtype=np.uint8)
            obs_group.create_dataset('agentview_image', data=agentview_images)

            robot_eye_images = np.random.randint(0, 255, (episode_length, 84, 84, 3), dtype=np.uint8)
            obs_group.create_dataset('robot0_eye_in_hand_image', data=robot_eye_images)

    print(f"Created robomimic-compatible dataset: {dataset_path}")
    print(f"  - Task: {task_name}")
    print(f"  - Episodes: {n_demos}")
    print(f"  - Episode length: {episode_length}")


def test_rollout_with_real_env(task_name: str = "stack_d1", use_gpu: bool = False):
    """Test rollout collector with real robomimic environment."""

    print("=" * 60)
    print(f"Testing Rollout with REAL Environment: {task_name}")
    print("=" * 60)

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # 1. Create minimal dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = os.path.join(temp_dir, f"{task_name}_dataset.hdf5")

            print(f"\n1. Creating Real Dataset...")
            create_minimal_dataset(dataset_path, task_name, n_demos=2, episode_length=50)

            # 2. Create dummy policy
            print(f"\n2. Creating Dummy Policy...")
            policy = MinimalDummyPolicy(
                horizon=16,
                n_action_steps=8,
                action_dim=7,  # Real robomimic action space
                device=device
            )
            print(f"   ✓ Policy created for 7-DOF robotic control")

            # 3. Create shape meta for the task
            shape_meta = {
                'obs': {
                    'robot0_eye_in_hand_image': {'shape': [3, 84, 84], 'type': 'rgb'},
                    'agentview_image': {'shape': [3, 84, 84], 'type': 'rgb'},
                    'point_cloud': {'shape': [1024, 6], 'type': 'point_cloud'},
                    'robot0_eef_pos': {'shape': [3]},
                    'robot0_eef_quat': {'shape': [4]},
                    'robot0_gripper_qpos': {'shape': [2]},
                },
                'action': {'shape': [7]}  # Real robotic action space
            }

            print(f"\n3. Creating Real RL Runner...")

            # 4. Create RL runner with real robomimic environment
            rl_runner = RLRobomimicRunner(
                output_dir=temp_dir,
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                n_train=1,      # Single training episode
                n_train_vis=1,  # With video
                train_start_idx=0,
                n_test=1,       # Single test episode
                n_test_vis=1,   # With video
                test_start_seed=100000,
                max_steps=max_steps[task_name],
                n_obs_steps=2,
                n_action_steps=8,
                render_obs_key='agentview_image',
                fps=10,
                crf=22,
                past_action=False,
                abs_action=True,  # Use absolute actions
                tqdm_interval_sec=0.5,
                n_envs=2,        # Total environments
                collect_episode_data=True
            )

            print(f"   ✓ Real RL runner created")
            print(f"   - Max episode steps: {max_steps[task_name]}")
            print(f"   - Number of environments: 2")
            print(f"   - Using absolute actions: True")

            # 5. Test rollout collection with real environment
            print(f"\n4. Running Real Environment Rollout...")

            # This will create actual robosuite environments and run rollouts
            rollout_metrics, episode_data = rl_runner.run_rl_collection(policy)

            print(f"   ✓ Real environment rollout completed!")

            # 6. Analyze results
            print(f"\n5. Analyzing Real Environment Results...")

            n_episodes = len(episode_data['episode_lengths'])
            total_rewards = [ep['total_reward'] for ep in episode_data['episodes']]

            print(f"   - Episodes collected: {n_episodes}")
            print(f"   - Episode lengths: {episode_data['episode_lengths']}")
            print(f"   - Total rewards: {[f'{r:.3f}' for r in total_rewards]}")
            print(f"   - Rollout metrics: {list(rollout_metrics.keys())}")

            # Check if we got train and test episodes
            if 'train/mean_score' in rollout_metrics:
                print(f"   - Train mean score: {rollout_metrics['train/mean_score']:.3f}")
            if 'test/mean_score' in rollout_metrics:
                print(f"   - Test mean score: {rollout_metrics['test/mean_score']:.3f}")

            # 7. Verify real environment data
            print(f"\n6. Verifying Real Environment Data...")

            ep0 = episode_data['episodes'][0]
            print(f"   - Real observation keys: {list(ep0['obs'].keys())}")
            print(f"   - Point cloud shape: {ep0['obs']['point_cloud'].shape}")
            print(f"   - Camera image shape: {ep0['obs']['agentview_image'].shape if 'agentview_image' in ep0['obs'] else 'N/A'}")
            print(f"   - Robot EEF pos shape: {ep0['obs']['robot0_eef_pos'].shape}")
            print(f"   - Action shape: {ep0['actions'].shape} (7-DOF robotic control)")
            print(f"   - Fixed noise shape: {ep0['fixed_noise'].shape}")

            # 8. Check for videos
            video_count = sum(1 for k, v in rollout_metrics.items()
                            if k.endswith('_video') and v is not None)
            print(f"   - Videos recorded: {video_count}")

            print(f"\n" + "=" * 60)
            print(f"✅ REAL Environment Test PASSED!")
            print(f"   Successfully collected rollouts from real robosuite {task_name} environment")
            print(f"   Rollout collector works with actual robotic simulation")
            print(f"=" * 60)

            return rollout_metrics, episode_data

    except Exception as e:
        print(f"\n❌ REAL Environment Test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import sys

    task_name = "stack_d1"
    use_gpu = False

    # Parse command line arguments
    if len(sys.argv) > 1:
        task_name = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2].lower() == 'gpu':
        use_gpu = True

    print(f"Testing with real environment: {task_name}")
    if use_gpu:
        print("GPU acceleration enabled")

    test_rollout_with_real_env(task_name, use_gpu)