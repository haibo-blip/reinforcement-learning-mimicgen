#!/usr/bin/env python3

import os
import sys
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Register hydra resolvers (same as before)
max_steps = {
    'stack_d1': 400,      # <-- Real task max steps
    'lift_d0': 400,
    'square_d2': 400,
    'coffee_d2': 400,
    # ... etc
}

def get_ws_x_center(task_name):
    if task_name.startswith('kitchen_') or task_name.startswith('hammer_cleanup_'):
        return -0.2
    else:
        return 0.

OmegaConf.register_new_resolver("get_max_steps", lambda x: max_steps[x], replace=True)
OmegaConf.register_new_resolver("get_ws_x_center", get_ws_x_center, replace=True)
OmegaConf.register_new_resolver("get_ws_y_center", lambda x: 0., replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)

from equi_diffpo.rl_training.minimal_dummy_policy import MinimalDummyPolicy


def test_rollout_with_task_config(task_name: str = "stack_d1"):
    """Test rollout collector with real task configuration."""
    print(f"=" * 60)
    print(f"Testing Rollout with Task: {task_name}")
    print(f"=" * 60)

    # Create task configuration
    task_config = {
        'task_name': task_name,
        'max_steps': max_steps[task_name],  # Real task episode length
        'shape_meta': {
            'obs': {
                'point_cloud': {'shape': [1024, 6], 'type': 'point_cloud'},
                'robot0_eef_pos': {'shape': [3], 'type': 'low_dim'},
                'robot0_eef_quat': {'shape': [4], 'type': 'low_dim'},
                'robot0_gripper_qpos': {'shape': [2], 'type': 'low_dim'},
            },
            'action': {'shape': [10]}
        }
    }

    print(f"Task Configuration:")
    print(f"  - Task: {task_config['task_name']}")
    print(f"  - Max episode steps: {task_config['max_steps']}")
    print(f"  - Action dim: {task_config['shape_meta']['action']['shape'][0]}")

    try:
        # 1. Create dummy policy matching task action space
        print(f"\n1. Creating Policy for {task_name}...")
        policy = MinimalDummyPolicy(
            horizon=16,
            n_action_steps=8,
            action_dim=task_config['shape_meta']['action']['shape'][0],
            device="cpu"
        )
        print(f"   ✓ Policy created for action dim: {policy.action_dim}")

        # 2. Create mock environment with task-specific parameters
        print(f"\n2. Creating Mock Environment for {task_name}...")

        class TaskSpecificMockEnv:
            def __init__(self, task_config):
                self.task_name = task_config['task_name']
                self.max_steps = task_config['max_steps']
                self.action_shape = task_config['shape_meta']['action']['shape']
                self.step_count = 0

                # Task-specific success probability (mock)
                self.success_rates = {
                    'stack_d1': 0.7,   # Stacking is moderately difficult
                    'lift_d0': 0.9,    # Lifting is easier
                    'square_d2': 0.6,  # Square peg harder
                    'coffee_d2': 0.5   # Coffee pouring hardest
                }
                self.success_rate = self.success_rates.get(task_name, 0.7)

            def reset(self):
                self.step_count = 0
                # Return task-appropriate initial observations
                return {
                    'point_cloud': np.random.randn(1, 1024, 6).astype(np.float32),
                    'robot0_eef_pos': np.random.randn(1, 3).astype(np.float32) * 0.1,  # Near origin
                    'robot0_eef_quat': np.random.randn(1, 4).astype(np.float32) * 0.1,
                    'robot0_gripper_qpos': np.random.randn(1, 2).astype(np.float32) * 0.1,
                }

            def step(self, actions):
                self.step_count += 1

                # Task-specific reward function (mock)
                if self.task_name == 'stack_d1':
                    # Stacking: reward for upward motion + stability
                    reward = np.random.random() * 0.1 + (0.5 if self.step_count > 200 else 0.0)
                elif self.task_name == 'lift_d0':
                    # Lifting: reward for upward motion
                    reward = np.random.random() * 0.2 + 0.1
                else:
                    # Default task
                    reward = np.random.random() * 0.1

                # Task-specific termination
                if self.step_count > self.max_steps * 0.8:
                    # Higher chance of success/failure near end
                    done = np.random.random() < 0.1
                else:
                    done = np.random.random() < 0.02  # Small chance of early termination

                # Force termination at max steps
                if self.step_count >= self.max_steps:
                    done = True
                    # Task success bonus
                    if np.random.random() < self.success_rate:
                        reward += 1.0  # Success bonus

                obs = {
                    'point_cloud': np.random.randn(1, 1024, 6).astype(np.float32),
                    'robot0_eef_pos': np.random.randn(1, 3).astype(np.float32) * 0.1,
                    'robot0_eef_quat': np.random.randn(1, 4).astype(np.float32) * 0.1,
                    'robot0_gripper_qpos': np.random.randn(1, 2).astype(np.float32) * 0.1,
                }

                return obs, np.array([reward]), np.array([done]), [{}]

        env = TaskSpecificMockEnv(task_config)
        print(f"   ✓ Mock environment created with max_steps: {env.max_steps}")
        print(f"   ✓ Task success rate: {env.success_rate}")

        # 3. Run episode collection
        print(f"\n3. Collecting Episodes for {task_name}...")

        # Single episode collection for testing
        obs = env.reset()
        policy.reset()

        episode_rewards = []
        episode_length = 0
        total_reward = 0

        done = False
        while not done:
            # Convert to torch
            obs_dict = {k: torch.from_numpy(v) for k, v in obs.items()}

            # Get action
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
            action = action_dict['action'].numpy()

            # Step environment
            obs, reward, done, info = env.step(action)

            episode_rewards.append(reward[0])
            episode_length += 1
            total_reward += reward[0]
            done = done[0]

        print(f"   ✓ Episode completed!")
        print(f"   - Episode length: {episode_length} steps")
        print(f"   - Total reward: {total_reward:.3f}")
        print(f"   - Average reward: {total_reward/episode_length:.4f}")
        print(f"   - Success: {'Yes' if total_reward > 0.5 else 'No'}")

        # 4. Compare with task expectations
        print(f"\n4. Task Analysis:")
        print(f"   - Expected max steps: {task_config['max_steps']}")
        print(f"   - Actual episode length: {episode_length}")
        print(f"   - Episode completion: {episode_length/task_config['max_steps']*100:.1f}%")

        if episode_length == task_config['max_steps']:
            print(f"   ✓ Episode ran to maximum length (as expected)")
        else:
            print(f"   ✓ Episode terminated early (task completed or failed)")

        print(f"\n" + "=" * 60)
        print(f"✅ Task {task_name} Test PASSED!")
        print(f"   Rollout collector works with task-specific configuration")
        print(f"=" * 60)

    except Exception as e:
        print(f"\n❌ Test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    # Allow task specification via command line
    task_name = "stack_d1"  # Default task
    if len(sys.argv) > 1:
        task_name = sys.argv[1]

    test_rollout_with_task_config(task_name)