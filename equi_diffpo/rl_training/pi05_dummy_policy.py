import torch
import numpy as np
from typing import Dict, Optional, Tuple
from equi_diffpo.policy.base_image_policy import BaseImagePolicy


class Pi05DummyPolicy(BaseImagePolicy):
    """
    Dummy policy that mimics pi0.5 model behavior for RL training with mimicgen settings.

    This policy generates dummy x_mean, x_std, and values similar to pi0.5's default_forward method.
    Uses action_dim=10 to match mimicgen.
    """

    def __init__(
        self,
        horizon: int = 16,
        n_action_steps: int = 8,
        action_dim: int = 10,  # Mimicgen setting
        num_steps: int = 5,    # Flow denoising steps (like pi0.5)
        noise_method: str = "flow_sde",
        value_head: bool = True,
        device: str = "cpu"
    ):
        super().__init__()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.num_steps = num_steps
        self.noise_method = noise_method
        self.value_head = value_head
        self._device = torch.device(device)

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return torch.float32

    def reset(self):
        """Reset policy state between episodes."""
        pass

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        return_values: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Predict actions using dummy pi0.5-like behavior.

        Args:
            obs_dict: Dictionary of real observations (preserved)
            noise: Optional fixed noise for flow SDE (matches pi0.5 interface)
            return_values: Whether to return value estimates

        Returns:
            Dictionary containing actions and optionally values
        """
        batch_size = self._get_batch_size(obs_dict)
        device = self._get_device(obs_dict)

        # Generate x_mean and x_std like pi0.5 model
        x_mean, x_std = self._generate_action_distribution(batch_size, device, noise)

        # Sample actions from the distribution (like pi0.5)
        if self.noise_method == "flow_sde":
            # For SDE, sample from the distribution
            eps = torch.randn_like(x_mean)
            actions = x_mean + x_std * eps
        else:
            # For deterministic, just use mean
            actions = x_mean

        # Clamp to reasonable ranges for robotic control
        actions = torch.clamp(actions, -1.0, 1.0)

        result = {'action': actions, 'x_mean': x_mean, 'x_std': x_std}

        # Add value estimates if requested (for RL training)
        if return_values or self.value_head:
            values = self._estimate_values(obs_dict)
            result['values'] = values

        return result

    def _get_batch_size(self, obs_dict: Dict[str, torch.Tensor]) -> int:
        """Get batch size from observation dictionary."""
        return list(obs_dict.values())[0].shape[0]

    def _get_device(self, obs_dict: Dict[str, torch.Tensor]) -> torch.device:
        """Get device from observation dictionary."""
        return list(obs_dict.values())[0].device

    def _generate_action_distribution(
        self,
        batch_size: int,
        device: torch.device,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate x_mean and x_std like pi0.5 model does.

        Returns:
            x_mean: Mean of action distribution [B, n_action_steps, action_dim]
            x_std: Std of action distribution [B, n_action_steps, action_dim]
        """
        # Generate mean actions
        if noise is not None:
            # Use provided noise if available (for flow SDE)
            x_mean = noise[:, :self.n_action_steps, :self.action_dim].clone() * 0.1
        else:
            # Generate random mean actions
            x_mean = torch.randn(
                batch_size, self.n_action_steps, self.action_dim,
                device=device
            ) * 0.1

        # Generate std (uncertainty) - typically smaller values
        if self.noise_method == "flow_sde":
            # For SDE, std varies with timestep (dummy implementation)
            x_std = torch.ones_like(x_mean) * 0.05  # Small noise level
        elif self.noise_method == "flow_noise":
            # For flow_noise, std is learnable (dummy implementation)
            x_std = torch.rand_like(x_mean) * 0.1 + 0.01  # Between 0.01-0.11
        else:
            # Deterministic case
            x_std = torch.zeros_like(x_mean)

        return x_mean, x_std

    def _estimate_values(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Estimate state values based on real observations (dummy but uses obs info).
        """
        batch_size = self._get_batch_size(obs_dict)
        device = self._get_device(obs_dict)

        # Use observation features to generate more realistic values
        if 'robot0_eef_pos' in obs_dict:
            # Use robot position to create some variation in values
            pos = obs_dict['robot0_eef_pos']  # B, 3
            # Simple heuristic: lower values when robot is further from origin
            distances = torch.norm(pos, dim=-1)  # B
            values = 1.0 - torch.clamp(distances / 2.0, 0, 1)  # B
        else:
            # Fallback to random values
            values = torch.randn(batch_size, device=device) * 0.1 + 0.5

        values = torch.clamp(values, 0.0, 1.0)
        return values

    def get_logprobs(
        self,
        obs_dict: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        fixed_noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute action log probabilities for RL training.

        This simulates the log probability calculation that pi0.5 does
        based on the action distribution.
        """
        batch_size = actions.shape[0]
        device = actions.device

        # Get the action distribution
        x_mean, x_std = self._generate_action_distribution(batch_size, device, fixed_noise)

        # Compute log probabilities under Gaussian distribution (dummy)
        # log p(a) = -0.5 * ((a - mean) / std)^2 - log(std) - 0.5 * log(2π)
        if torch.all(x_std > 0):
            # Proper Gaussian log probability
            normalized_actions = (actions - x_mean) / (x_std + 1e-8)
            log_probs = -0.5 * (normalized_actions ** 2 + torch.log(2 * torch.pi * x_std ** 2))
            log_probs = log_probs.sum(dim=[1, 2])  # Sum over action steps and dimensions
        else:
            # Deterministic case (zero std)
            log_probs = torch.zeros(batch_size, device=device)

        return log_probs

    def get_action_and_logprobs(
        self,
        obs_dict: Dict[str, torch.Tensor],
        fixed_noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both actions and their log probabilities in one call.
        """
        action_dict = self.predict_action(obs_dict, noise=fixed_noise)
        actions = action_dict['action']
        logprobs = self.get_logprobs(obs_dict, actions, fixed_noise)
        return actions, logprobs

    def get_action_mean_std_values(
        self,
        obs_dict: Dict[str, torch.Tensor],
        fixed_noise: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get action mean, std, and values (mimics pi0.5's output structure).
        """
        return self.predict_action(obs_dict, noise=fixed_noise, return_values=True)


def create_pi05_test_policy(device: str = "cpu", **kwargs) -> Pi05DummyPolicy:
    """
    Factory function to create a pi0.5 dummy policy with mimicgen defaults.
    """
    return Pi05DummyPolicy(
        horizon=16,
        n_action_steps=8,
        action_dim=10,  # Mimicgen setting
        num_steps=5,
        noise_method="flow_sde",
        value_head=True,
        device=device,
        **kwargs
    )


if __name__ == "__main__":
    # Test the dummy policy with real observation structure
    policy = create_pi05_test_policy()

    # Test with mimicgen-like observations
    batch_size = 2
    obs_dict = {
        'point_cloud': torch.randn(batch_size, 1024, 6),
        'robot0_eef_pos': torch.tensor([[0.5, 0.2, 0.8], [0.1, 0.3, 0.9]]),
        'robot0_eef_quat': torch.randn(batch_size, 4),
        'robot0_gripper_qpos': torch.randn(batch_size, 2),
    }

    # Test action prediction (pi0.5 style)
    result = policy.get_action_mean_std_values(obs_dict)
    print(f"Actions shape: {result['action'].shape}")
    print(f"x_mean shape: {result['x_mean'].shape}")
    print(f"x_std shape: {result['x_std'].shape}")
    print(f"Values shape: {result['values'].shape}")
    print(f"Action range: {result['action'].min():.3f} to {result['action'].max():.3f}")
    print(f"x_mean range: {result['x_mean'].min():.3f} to {result['x_mean'].max():.3f}")
    print(f"x_std range: {result['x_std'].min():.3f} to {result['x_std'].max():.3f}")

    # Test log probabilities
    actions, logprobs = policy.get_action_and_logprobs(obs_dict)
    print(f"Log probabilities shape: {logprobs.shape}")
    print(f"Log probabilities: {logprobs}")

    print("Pi0.5 dummy policy test completed successfully!")