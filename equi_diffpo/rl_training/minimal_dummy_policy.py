import torch
import numpy as np
from typing import Dict, Optional


class MinimalDummyPolicy:
    """
    Minimal dummy policy for testing RL rollout collection.
    Uses zero GPU memory and no parameters - just returns random actions.
    """

    def __init__(
        self,
        horizon: int = 16,
        n_action_steps: int = 8,
        action_dim: int = 10,
        device: str = "cpu"
    ):
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self._device = torch.device(device)

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return torch.float32

    def reset(self):
        """Reset policy state (no-op)."""
        pass

    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                      noise: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Generate random actions for testing.

        Args:
            obs_dict: Dictionary of observations (ignored)
            noise: Optional fixed noise (ignored for now)

        Returns:
            action_dict: Dictionary containing random actions
        """
        batch_size = list(obs_dict.values())[0].shape[0]
        device = list(obs_dict.values())[0].device

        # Generate random actions in reasonable range
        actions = torch.randn(batch_size, self.n_action_steps, self.action_dim, device=device) * 0.1

        return {'action': actions}


def create_minimal_test_obs(batch_size: int = 4, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Create minimal observation dictionary for testing."""
    device = torch.device(device)

    return {
        'point_cloud': torch.randn(batch_size, 1024, 6, device=device),
        'robot0_eef_pos': torch.randn(batch_size, 3, device=device),
        'robot0_eef_quat': torch.randn(batch_size, 4, device=device),
        'robot0_gripper_qpos': torch.randn(batch_size, 2, device=device),
    }