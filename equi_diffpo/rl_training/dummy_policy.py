import torch
import numpy as np
from typing import Dict, Optional
from equi_diffpo.policy.base_image_policy import BaseImagePolicy


class DummyManiFlowPolicy(BaseImagePolicy):
    """
    Dummy policy for testing RL rollout collection without requiring
    a full ManiFlow model. Mimics the interface of ManiFlowTransformerPointcloudPolicy.
    """

    def __init__(
        self,
        horizon: int = 16,
        n_action_steps: int = 8,
        n_obs_steps: int = 2,
        action_dim: int = 10,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self._device = torch.device(device)

        # Simple linear layer to generate actions from observations
        # This simulates the policy network
        self.dummy_net = torch.nn.Sequential(
            torch.nn.Linear(100, 256),  # Simplified obs encoding
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_action_steps * action_dim)
        ).to(self._device)

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return torch.float32

    def reset(self):
        """Reset policy state (no-op for dummy policy)."""
        pass

    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                      noise: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predict actions from observations.

        Args:
            obs_dict: Dictionary of observations
            noise: Optional fixed noise for flow SDE (for testing)

        Returns:
            action_dict: Dictionary containing predicted actions
        """
        batch_size = list(obs_dict.values())[0].shape[0]
        device = list(obs_dict.values())[0].device

        # Simple observation encoding (flatten everything)
        obs_features = []

        for key, value in obs_dict.items():
            if key == 'point_cloud':
                # Flatten point cloud: [batch_size, n_points, features] -> [batch_size, n_points * features]
                # Take mean across points for simplicity
                flattened = value.mean(dim=1).flatten(start_dim=1)
                obs_features.append(flattened)
            else:
                # Flatten other observations
                obs_features.append(value.flatten(start_dim=1))

        # Concatenate all features
        if obs_features:
            obs_concat = torch.cat(obs_features, dim=-1)
        else:
            obs_concat = torch.randn(batch_size, 100, device=device)

        # Pad or truncate to fixed size for dummy network
        if obs_concat.shape[-1] > 100:
            obs_concat = obs_concat[:, :100]
        elif obs_concat.shape[-1] < 100:
            padding = torch.zeros(batch_size, 100 - obs_concat.shape[-1], device=device)
            obs_concat = torch.cat([obs_concat, padding], dim=-1)

        # Generate actions through dummy network
        action_flat = self.dummy_net(obs_concat)
        actions = action_flat.reshape(batch_size, self.n_action_steps, self.action_dim)

        # Add some randomness for realistic behavior
        noise_scale = 0.1
        if noise is not None:
            # Use provided noise (for flow SDE testing)
            # Reshape noise to match action dimensions
            if noise.shape != actions.shape:
                # Take subset of noise dimensions if needed
                noise_reshaped = noise[:, :self.n_action_steps, :self.action_dim]
            else:
                noise_reshaped = noise
            actions = actions + noise_scale * noise_reshaped[:, :self.n_action_steps, :self.action_dim]
        else:
            # Generate random noise
            actions = actions + noise_scale * torch.randn_like(actions)

        # Clamp actions to reasonable range
        actions = torch.clamp(actions, -1.0, 1.0)

        return {'action': actions}

    def forward(self, *args, **kwargs):
        """Forward pass (required by base class)."""
        return self.predict_action(*args, **kwargs)


class DummyNormalizer:
    """Dummy normalizer for testing."""

    def normalize(self, data):
        """No-op normalization for testing."""
        return data

    def unnormalize(self, data):
        """No-op unnormalization for testing."""
        return data


def create_dummy_obs_dict(batch_size: int = 4, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Create dummy observation dictionary for testing."""
    device = torch.device(device)

    return {
        'point_cloud': torch.randn(batch_size, 1024, 6, device=device),
        'robot0_eef_pos': torch.randn(batch_size, 3, device=device),
        'robot0_eef_quat': torch.randn(batch_size, 4, device=device),
        'robot0_gripper_qpos': torch.randn(batch_size, 2, device=device),
    }


def create_dummy_shape_meta() -> Dict:
    """Create dummy shape meta for testing."""
    return {
        'obs': {
            'point_cloud': {'shape': [1024, 6], 'type': 'point_cloud'},
            'robot0_eef_pos': {'shape': [3], 'type': 'low_dim'},
            'robot0_eef_quat': {'shape': [4], 'type': 'low_dim'},
            'robot0_gripper_qpos': {'shape': [2], 'type': 'low_dim'},
        },
        'action': {'shape': [10]}
    }