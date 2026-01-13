"""
RL ManiFlow Pointcloud Policy
Based on ManiFlowTransformerPointcloudPolicy with RL enhancements.

Supports:
- flow-sde: Stochastic differential equations for flow sampling
- flow-noise: Learnable noise for exploration
- Value estimation for RL training
- Action sampling with log probabilities
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint

from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.common.model_util import print_params
from equi_diffpo.model.vision_3d.pointnet_extractor import DP3Encoder
from equi_diffpo.model.diffusion.ditx import DiTX
from equi_diffpo.model.common.sample_util import *


class ExploreNoiseNet(nn.Module):
    """Learnable noise network for flow-noise exploration."""

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dims: list = [128, 64],
                 activation_type: str = "tanh",
                 noise_logvar_range: list = [0.08, 0.16],
                 noise_scheduler_type: str = "learn"):
        super().__init__()

        self.noise_logvar_range = noise_logvar_range
        self.noise_scheduler_type = noise_scheduler_type

        # Build MLP
        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if activation_type == "relu" else nn.Tanh()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

        # Initialize to output small values initially
        with torch.no_grad():
            self.mlp[-1].weight.data *= 0.1
            self.mlp[-1].bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, in_dim]
        Returns:
            log_std: Log standard deviations [B, out_dim]
        """
        log_std = self.mlp(x)

        # Clamp to noise range
        min_log_std = torch.log(torch.tensor(self.noise_logvar_range[0], device=x.device))
        max_log_std = torch.log(torch.tensor(self.noise_logvar_range[1], device=x.device))
        log_std = torch.clamp(log_std, min_log_std, max_log_std)

        return log_std


class ValueHead(nn.Module):
    """Value estimation head for RL training."""

    def __init__(self,
                 input_dim: int,
                 hidden_sizes: tuple = (512, 256, 128),
                 output_dim: int = 1,
                 activation: str = "relu",
                 bias_last: bool = True):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.ReLU() if activation == "relu" else nn.Tanh()
            ])
            prev_dim = hidden_size

        layers.append(nn.Linear(prev_dim, output_dim, bias=bias_last))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, input_dim]
        Returns:
            values: State values [B, 1]
        """
        return self.mlp(x)


class ManiFlowRLPointcloudPolicy(BaseImagePolicy):
    """
    ManiFlow RL Policy for pointcloud observations.

    Key features:
    - Supports flow-sde and flow-noise for action sampling
    - Provides log probabilities for RL training
    - Includes value head for critic estimation
    - Compatible with PPO/SAC style RL algorithms

    Time parameterization: t=0 is clean data, t=1 is full noise
    """

    def __init__(self,
                 shape_meta: dict,
                 horizon: int,
                 n_action_steps: int,
                 n_obs_steps: int,
                 # ManiFlow parameters
                 num_inference_steps: int = 10,
                 obs_as_global_cond: bool = True,
                 diffusion_timestep_embed_dim: int = 256,
                 diffusion_target_t_embed_dim: int = 256,
                 visual_cond_len: int = 1024,
                 # Model architecture
                 n_layer: int = 3,
                 n_head: int = 4,
                 n_emb: int = 256,
                 qkv_bias: bool = False,
                 qk_norm: bool = False,
                 block_type: str = "DiTX",
                 # Encoder parameters
                 encoder_type: str = "DP3Encoder",
                 encoder_output_dim: int = 256,
                 crop_shape=None,
                 use_pc_color: bool = False,
                 pointnet_type: str = "pointnet",
                 pointcloud_encoder_cfg=None,
                 downsample_points: bool = False,
                 pre_norm_modality: bool = False,
                 language_conditioned: bool = False,
                 # RL-specific parameters
                 noise_method: str = "flow_sde",  # flow_sde, flow_noise
                 noise_level: float = 0.5,
                 noise_anneal: bool = False,
                 noise_params: list = [0.7, 0.3, 400],  # noise_start, noise_end, noise_anneal_steps
                 noise_logvar_range: list = [0.08, 0.16],  # for flow_noise
                 add_value_head: bool = True,
                 value_hidden_sizes: tuple = (512, 256, 128),
                 safe_get_logprob: bool = False,
                 joint_logprob: bool = False,  # like Pi0.5
                 # Flow training parameters (kept for compatibility)
                 flow_batch_ratio: float = 0.75,
                 consistency_batch_ratio: float = 0.25,
                 denoise_timesteps: int = 10,
                 sample_t_mode_flow: str = "beta",
                 sample_t_mode_consistency: str = "discrete",
                 sample_dt_mode_consistency: str = "uniform",
                 sample_target_t_mode: str = "relative",
                 **kwargs):
        super().__init__()

        # Store configuration
        self.noise_method = noise_method
        self.noise_level = noise_level
        self.noise_anneal = noise_anneal
        self.noise_params = noise_params
        self.noise_logvar_range = noise_logvar_range
        self.safe_get_logprob = safe_get_logprob
        self.joint_logprob = joint_logprob
        self.global_step = 0  # For noise annealing

        # Parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # Create observation encoder
        self.encoder_type = encoder_type
        if encoder_type == "DP3Encoder":
            obs_encoder = DP3Encoder(observation_space=obs_dict,
                                   img_crop_shape=crop_shape,
                                   out_channel=encoder_output_dim,
                                   pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                   use_pc_color=use_pc_color,
                                   pointnet_type=pointnet_type,
                                   downsample_points=downsample_points)
        else:
            raise ValueError(f"Unsupported encoder type {encoder_type}")

        # Create ManiFlow model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim

        model = DiTX(
            input_dim=input_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=global_cond_dim,
            visual_cond_len=visual_cond_len,
            diffusion_timestep_embed_dim=diffusion_timestep_embed_dim,
            diffusion_target_t_embed_dim=diffusion_target_t_embed_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            block_type=block_type,
            pre_norm_modality=pre_norm_modality,
            language_conditioned=language_conditioned,
        )

        self.obs_encoder = obs_encoder
        self.model = model

        # RL-specific components
        if noise_method == "flow_noise":
            # Learnable noise head
            self.noise_head = ExploreNoiseNet(
                in_dim=global_cond_dim or obs_feature_dim,
                out_dim=action_dim * horizon,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=noise_logvar_range,
                noise_scheduler_type="learn"
            )
        else:
            self.noise_head = None

        if add_value_head:
            # Value estimation head
            self.value_head = ValueHead(
                input_dim=global_cond_dim or obs_feature_dim,
                hidden_sizes=value_hidden_sizes,
                output_dim=1,
                activation="relu",
                bias_last=True
            )
        else:
            self.value_head = None

        # Store other parameters (following original maniflow_pointcloud_policy)
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.language_conditioned = language_conditioned
        self.num_inference_steps = num_inference_steps
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        # RL training mode control (like Pi0.5)
        self._training_mode = True  # True for train, False for eval

        # Flow parameters (kept for training compatibility)
        self.flow_batch_ratio = flow_batch_ratio
        self.consistency_batch_ratio = consistency_batch_ratio
        self.denoise_timesteps = denoise_timesteps
        self.sample_t_mode_flow = sample_t_mode_flow
        self.sample_t_mode_consistency = sample_t_mode_consistency
        self.sample_dt_mode_consistency = sample_dt_mode_consistency
        self.sample_target_t_mode = sample_target_t_mode

        cprint(f"[ManiFlowRLPointcloudPolicy] Initialized with:", "yellow")
        cprint(f"  - noise_method: {self.noise_method}", "yellow")
        cprint(f"  - horizon: {self.horizon}", "yellow")
        cprint(f"  - n_action_steps: {self.n_action_steps}", "yellow")
        cprint(f"  - n_obs_steps: {self.n_obs_steps}", "yellow")
        cprint(f"  - num_inference_steps: {self.num_inference_steps}", "yellow")
        cprint(f"  - add_value_head: {add_value_head}", "yellow")

        print_params(self)

    def set_global_step(self, global_step: int):
        """Set global step for noise annealing."""
        self.global_step = global_step

    def compute_value(self, vis_cond: torch.Tensor) -> torch.Tensor:
        """
        Compute state value from observation features (called ONCE per sample).

        This method should be called once per observation, not per denoising step,
        since V(s) is a function of state only.

        Args:
            vis_cond: Visual conditioning [B, n_obs_steps*L, obs_feature_dim]

        Returns:
            value: State value [B]
        """
        if self.value_head is None:
            return torch.zeros(vis_cond.shape[0], device=vis_cond.device)

        # Pool observation features across sequence dimension for value estimation
        obs_features_pooled = vis_cond.mean(dim=1)  # [B, obs_feature_dim]
        return self.value_head(obs_features_pooled).squeeze(-1)  # [B]

    def train(self, mode: bool = True):
        """Set training mode (adds noise like Pi0.5)."""
        self._training_mode = True

    def eval(self):
        """Set evaluation mode (deterministic, no noise like Pi0.5)."""
        self._training_mode = False

    def get_current_noise_level(self) -> float:
        """Get current noise level (with annealing if enabled)."""
        if not self.noise_anneal:
            return self.noise_level

        noise_start, noise_end, noise_anneal_steps = self.noise_params
        progress = min(self.global_step / noise_anneal_steps, 1.0)
        return noise_start + (noise_end - noise_start) * progress

    def encode_observations(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode observations to features.

        Args:
            obs_dict: Dictionary of observations

        Returns:
            obs_features: Encoded observation features [B, n_obs_steps*L, obs_feature_dim]
        """
        # Normalize input
        if 'agentview_image' in obs_dict:
            del obs_dict['agentview_image']
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # Encode observations
        device = self.device
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]).to(device))
        nobs_features = self.obs_encoder(this_nobs)
        obs_features = nobs_features.reshape(B, -1, Do)  # B, n_obs_steps*L, obs_feature_dim

        return obs_features

    def sample_noise(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """
        Sample noise exactly like Pi0.5.

        Args:
            shape: Shape of noise tensor
            device: Device to place tensor on

        Returns:
            noise: Standard normal noise tensor
        """
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def get_step_prediction(self,
                           x_t: torch.Tensor,
                           t,
                           target_t,
                           vis_cond: Optional[torch.Tensor] = None,
                           lang_cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get step prediction for mean and std only (value computed separately via compute_value).

        Args:
            x_t: Current state [B, T, Da]
            t: Current time (float or tensor [B])
            target_t: Target time (float or tensor [B])
            vis_cond: Visual conditioning
            lang_cond: Language conditioning

        Returns:
            x_t_mean: Predicted mean [B, T, Da]
            x_t_std: Predicted std [B, T, Da]
        """
        device = x_t.device
        B = x_t.shape[0]

        # Create time tensors - handle both scalar and tensor inputs
        if isinstance(t, (int, float)):
            t_tensor = torch.full((B,), t, device=device, dtype=x_t.dtype)
            t_scalar = t
        else:
            t_tensor = t.to(device=device, dtype=x_t.dtype)
            t_scalar = t.mean().item()  # For step_idx calculation

        if isinstance(target_t, (int, float)):
            target_t_tensor = torch.full((B,), target_t, device=device, dtype=x_t.dtype)
        else:
            target_t_tensor = target_t.to(device=device, dtype=x_t.dtype)
        target_t_tensor_for_model = torch.full((B,), 0, device=device, dtype=x_t.dtype)

        # Get velocity prediction
        v_pred = self.model(
            sample=x_t,
            timestep=t_tensor,
            target_t=target_t_tensor_for_model,
            vis_cond=vis_cond,
            lang_cond=lang_cond
        )

        # Compute predictions using flow matching (like Pi0.5)
        # dt can be tensor or scalar
        dt = target_t_tensor - t_tensor if torch.is_tensor(target_t_tensor) else target_t - t
        t_input = t_tensor[:, None, None].expand_as(x_t)
        if torch.is_tensor(dt):
            delta = dt.abs()[:, None, None].expand_as(x_t)
        else:
            delta = torch.full_like(t_input, abs(dt))

        # Flow predictions (matching Pi0.5 logic)
        x0_pred = x_t - v_pred * t_input  # Predict clean data
        x1_pred = x_t + v_pred * (1 - t_input)  # Predict pure noise

        # Compute weights and std based on mode and noise method (exactly like Pi0.5)
        mode = "train" if self._training_mode else "eval"

        if mode == "eval":
            # Pi0.5 eval mode: deterministic, no noise
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)

        elif mode == "train":
            # Pi0.5 train mode: add noise based on method
            if self.noise_method == "flow_sde":
                # Exactly like Pi0.5 flow_sde
                current_noise_level = self.get_current_noise_level()

                # Create timesteps for sigma calculation (like Pi0.5)
                timesteps = torch.linspace(1, 0, self.num_inference_steps + 1, device=device)
                sigmas = (
                    current_noise_level
                    * timesteps[:-1]
                    / (1 - torch.where(timesteps[:-1] == 1, timesteps[1], timesteps[:-1]))
                )

                # Get current step index (approximate) - use mean t for batch
                step_idx = int(t_scalar * self.num_inference_steps)
                step_idx = max(0, min(step_idx, len(sigmas) - 1))
                sigma_i = sigmas[step_idx]
                sigma_i = sigma_i.expand_as(t_input)

                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i

            elif self.noise_method == "flow_noise" and self.noise_head is not None:
                    # To be implement
                    raise Exception()

            else:
                # Default train case
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = torch.full_like(x_t, 0.1)  # Small default std

        else:
            raise ValueError(f"Invalid mode: {mode}")

        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight

        return x_t_mean, x_t_std

    def sample_ode(self,
                   x0: torch.Tensor,
                   N: int,
                   vis_cond: Optional[torch.Tensor] = None,
                   lang_cond: Optional[torch.Tensor] = None,
                   return_chains: bool = False,
                   denoise_inds: Optional[torch.Tensor] = None,
                   **kwargs) -> Dict[str, torch.Tensor]:
        """
        Sample using SDE-style sampling (like Pi0.5, not pure ODE).

        This matches Pi0.5's approach of adding fresh noise at each step
        rather than deterministic ODE solving.

        Args:
            x0: Initial noise [B, T, Da] where t=1 (full noise)
            N: Number of inference steps
            vis_cond: Visual conditioning [B, n_obs_steps*L, Do]
            lang_cond: Language conditioning (optional)
            return_chains: Whether to return sampling chains (like Pi0.5)
            denoise_inds: Denoise indices for training [B, N] (like Pi0.5)

        Returns:
            Dictionary containing:
            - 'actions': Final sampled trajectory [B, T, Da]
            - 'chains': Sampling chain if return_chains=True [B, N+1, T, Da]
            - 'prev_values': State value if return_chains=True [B, 1]
            - 'denoise_inds': Denoise indices if provided [B, N]
        """
        device = x0.device
        B = x0.shape[0]
        dt = 1.0 / N

        # Compute value ONCE before the denoising loop (not per step)
        # V(s) is a function of state only, not denoising timestep
        value = self.compute_value(vis_cond)  # [B]

        # Initialize like Pi0.5
        x_t = x0.clone()  # Start with initial noise
        chains = []
        log_probs = []

        # Record initial state (like Pi0.5)
        chains.append(x_t)

        denoise_inds = torch.arange(N).unsqueeze(0).repeat(B, 1).to(device)

        # Denoise steps (like Pi0.5) - no value computation inside loop
        for i in range(N):
            t = 1.0 - i * dt  # Start from t=1 (noise) to t=0 (data)
            target_t = 1.0 - (i + 1) * dt

            # Get only mean and std (value computed once above)
            x_t_mean, x_t_std = self.get_step_prediction(
                x_t, t, target_t, vis_cond, lang_cond
            )

            # SDE step (exactly like Pi0.5): add fresh noise
            # Pi0.5: x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            fresh_noise = self.sample_noise(x_t.shape, device)
            x_t = x_t_mean + fresh_noise * x_t_std

            # Compute log probability (like Pi0.5)
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)

            # Store results (like Pi0.5)
            chains.append(x_t)
            log_probs.append(log_prob)

        # Final trajectory
        x_0 = x_t

        # Prepare output (like Pi0.5)
        result = {'actions': x_0}

        if return_chains:
            chains = torch.stack(chains, dim=1)  # [B, N+1, T, Da]
            log_probs = torch.stack(log_probs, dim=1)  # [B, N, ...]

            result.update({
                'chains': chains,
                'prev_logprobs': log_probs,
                'prev_values': value.unsqueeze(-1),  # [B, 1] - single value per sample
                'denoise_inds': denoise_inds,
            })

        return result

    def get_logprob_norm(self, sample: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability under Gaussian distribution (like Pi0.5).

        Args:
            sample: Sampled values [B, T, Da]
            mu: Mean values [B, T, Da]
            sigma: Standard deviation [B, T, Da]

        Returns:
            log_prob: Log probabilities [B, T, Da]
        """
        # import ipdb; ipdb.set_trace()
        if self.safe_get_logprob:
            # Simplified version for numerical stability
            log_prob = -torch.pow((sample - mu), 2)
        else:
            # Full Gaussian log probability
            eps = 1e-8
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)

            # Split into constant and exponent terms (like OpenPI)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term


            # Handle zero std case
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)

        return log_prob

    def gaussian_entropy(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian entropy (like OpenPI).

        Args:
            sigma: Standard deviation tensor

        Returns:
            entropy: Gaussian entropy 0.5 * log(2πeσ²)
        """
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * torch.pi * torch.e * (sigma_safe ** 2))
        return entropy

    def get_log_prob_value(self,
                          observation: Dict[str, torch.Tensor],
                          chains: torch.Tensor,
                          denoise_inds: torch.Tensor,
                          compute_values: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities, values, and entropy from chains (like OpenPI).

        Args:
            observation: Observation dictionary
            chains: Sampling chains [B, N+1, horizon, action_dim]
            denoise_inds: Denoise indices [B, N]
            compute_values: Whether to compute values

        Returns:
            log_probs: Log probabilities [B, num_steps, horizon, action_dim]
            values: Value estimates [B] - single value per sample (not per denoising step)
            entropy: Entropy estimates [B, num_steps, horizon, action_dim]
        """
        B = chains.shape[0]
        N = denoise_inds.shape[1]

        # Encode observations
        vis_cond = self.encode_observations(observation)

        # Language conditioning
        lang_cond = None
        if self.language_conditioned:
            lang_cond = observation.get('task_name', None)

        # Compute value ONCE (not per denoising step)
        # V(s) is a function of state only
        if compute_values:
            value = self.compute_value(vis_cond)  # [B]
        else:
            value = torch.zeros(B, device=chains.device)

        chains_log_probs = []
        chains_entropy = []

        # Process denoise steps (like OpenPI)
        joint_logprob = getattr(self, 'joint_logprob', True)
        if joint_logprob:
            # Joint estimation: process all N steps
            num_steps = N
            # Initial log prob for joint estimation
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)

            # Process each step - no value computation inside loop
            for idx in range(num_steps):
                denoise_ind = denoise_inds[:, idx]
                chains_pre = chains[torch.arange(B), denoise_ind]
                chains_next = chains[torch.arange(B), denoise_ind + 1]

                # Get step prediction (mean and std only)
                x_t_mean, x_t_std = self.get_step_prediction_for_logprob(
                    chains_pre, denoise_ind, vis_cond, lang_cond
                )

                # Compute log probability and entropy
                log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
                entropy = self.gaussian_entropy(x_t_std)

                chains_log_probs.append(log_probs)
                chains_entropy.append(entropy)
        else:
            # Single step: use only the first denoise index per batch element
            # This matches the RLinf pattern where only one random timestep per batch is sampled
            denoise_ind = denoise_inds[:, 0]  # [B] - use first (and likely only) index
            chains_pre = chains[torch.arange(B), denoise_ind]
            chains_next = chains[torch.arange(B), denoise_ind + 1]

            # Get step prediction (mean and std only)
            x_t_mean, x_t_std = self.get_step_prediction_for_logprob(
                chains_pre, denoise_ind, vis_cond, lang_cond
            )

            # Compute log probability and entropy
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)

            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)

        # Stack results (handling both joint and single cases)
        chains_log_probs = torch.stack(chains_log_probs, dim=1)  # [B, num_log_probs, ...]

        # Entropy handling (like OpenPI)
        if self.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)

        # Return single value per sample [B], not [B, N]
        return chains_log_probs, value, chains_entropy

    def get_step_prediction_for_logprob(self,
                                       x_t: torch.Tensor,
                                       denoise_ind: torch.Tensor,
                                       vis_cond: torch.Tensor,
                                       lang_cond: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get step prediction for log probability computation.

        Note: Value is computed separately via compute_value(), not here.

        Args:
            x_t: Current state [B, horizon, action_dim]
            denoise_ind: Denoise indices [B]
            vis_cond: Visual conditioning
            lang_cond: Language conditioning

        Returns:
            x_t_mean: Predicted mean [B, horizon, action_dim]
            x_t_std: Predicted std [B, horizon, action_dim]
        """
        # Convert denoise_ind to timestep
        N = self.num_inference_steps
        dt = 1.0 / N
        t = 1.0 - denoise_ind.float() * dt
        target_t = 1.0 - (denoise_ind.float() + 1) * dt

        # Get prediction (mean and std only)
        x_t_mean, x_t_std = self.get_step_prediction(
            x_t, t, target_t, vis_cond, lang_cond
        )

        return x_t_mean, x_t_std

    def conditional_sample(self,
                          condition_data: torch.Tensor,
                          vis_cond: Optional[torch.Tensor] = None,
                          lang_cond: Optional[torch.Tensor] = None,
                          fixed_noise: Optional[torch.Tensor] = None,
                          return_chains: bool = False,
                          **kwargs) -> torch.Tensor:
        """
        Sample actions conditionally (exactly like Pi0.5).

        Args:
            condition_data: Shape for sampling [B, T, Da]
            vis_cond: Visual conditioning
            lang_cond: Language conditioning
            fixed_noise: Fixed noise for deterministic sampling
            return_chains: Whether to return sampling chains

        Returns:
            sampled_actions or dict with chains: Sampled action trajectory [B, T, Da] or full result
        """
        device = condition_data.device

        # Generate initial noise exactly like Pi0.5
        if fixed_noise is not None:
            noise = fixed_noise.to(device)
        else:
            # Pi0.5 pattern: noise = self.sample_noise(actions_shape, device)
            noise = self.sample_noise(condition_data.shape, device)

        # Sample using SDE (not pure ODE, matching Pi0.5)
        result = self.sample_ode(
            x0=noise,
            N=self.num_inference_steps,
            vis_cond=vis_cond,
            lang_cond=lang_cond,
            return_chains=return_chains,
            **kwargs
        )

        # Return just actions or full result based on return_chains
        if return_chains:
            return result
        else:
            return result['actions']

    def predict_action(self,
                      obs_dict: Dict[str, torch.Tensor],
                      return_chains: bool = False) -> Dict[str, torch.Tensor]:
        """
        Predict actions from observations (supporting chains like Pi0.5).

        Args:
            obs_dict: Dictionary of observations
            return_chains: Whether to return sampling chains (like Pi0.5)

        Returns:
            result: Dictionary containing:
            - 'action': Action steps [B, n_action_steps, Da]
            - 'action_pred': Full trajectory [B, horizon, Da]
            - 'chains': Sampling chain if return_chains=True [B, N+1, horizon, Da]
            - 'prev_logprobs': Log probabilities if return_chains=True [B, N, ...]
            - 'prev_values': Values if return_chains=True [B, N]
            - 'denoise_inds': Denoise indices if return_chains=True [B, N]
        """
        # Encode observations
        vis_cond = self.encode_observations(obs_dict)

        value = next(iter(obs_dict.values()))
        B = value.shape[0]
        T = self.horizon
        Da = self.action_dim

        # Language conditioning (if supported)
        lang_cond = None
        if self.language_conditioned:
            lang_cond = obs_dict.get('task_name', None)

        # Create condition data template
        cond_data = torch.zeros(size=(B, T, Da), device=self.device, dtype=self.dtype)

        # Sample actions (with optional chains like Pi0.5)
        if return_chains:
            sample_result = self.conditional_sample(
                cond_data,
                vis_cond=vis_cond,
                lang_cond=lang_cond,
                return_chains=True
            )
            nsample = sample_result['actions']
        else:
            nsample = self.conditional_sample(
                cond_data,
                vis_cond=vis_cond,
                lang_cond=lang_cond,
                return_chains=False
            )

        # Unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # Extract action steps
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        # Prepare result
        result = {
            'action': action,
            'action_pred': action_pred,
        }

        # Add chains and related data if requested (like Pi0.5)
        if return_chains:
            # Unnormalize chains as well
            # chains_unnorm = self.normalizer['action'].unnormalize(sample_result['chains'])

            result.update({
                'chains': sample_result['chains'],
                'prev_logprobs': sample_result['prev_logprobs'],
                'prev_values': sample_result['prev_values'],
                'denoise_inds': sample_result['denoise_inds'],
            })

        return result

    def sample_actions(self,
                      observation,
                      noise: Optional[torch.Tensor] = None,
                      mode: str = "train",
                      compute_values: bool = True,
                      **kwargs) -> Dict[str, torch.Tensor]:
        """
        Sample actions with chains support (like Pi0.5 sample_actions).

        Args:
            observation: Observation object or dict
            noise: Fixed noise for deterministic sampling
            mode: 'train' or 'eval'
            compute_values: Whether to compute values

        Returns:
            Dictionary containing actions, chains, logprobs, values, denoise_inds
        """
        # Convert observation to dict format if needed
        if hasattr(observation, '__dict__'):
            obs_dict = observation.__dict__
        else:
            obs_dict = observation

        # Set mode
        if mode == "train":
            self.train()
        else:
            self.eval()

        # Get result with chains
        result = self.predict_action(obs_dict, return_chains=True)

        # Extract action steps for final output
        action_steps = result['action']  # Already extracted by predict_action

        return {
            'actions': action_steps,
            'action_pred': result['action_pred'],
            'chains': result['chains'],
            'prev_logprobs': result['prev_logprobs'],
            'prev_values': result['prev_values'],
            'denoise_inds': result['denoise_inds'],
        }

    def default_forward(self,
                       data: dict,
                       **kwargs) -> Dict[str, torch.Tensor]:
        """
        Default forward pass for RL training (like OpenPI).

        Expects chains and denoise_inds in data, computes logprobs, values, entropy.

        Args:
            data: Dictionary with 'observation', 'chains', 'denoise_inds'
            **kwargs: Additional arguments (compute_values, etc.)

        Returns:
            Dictionary with 'logprobs', 'values', 'entropy'
        """
        # Extract arguments (like OpenPI)
        compute_values = kwargs.get("compute_values", False)
        chains = data["chains"]  # [B, N+1, horizon, action_dim]
        denoise_inds = data["denoise_inds"]  # [B, N]
        observation = data["observation"]

        # Get log probs, values, entropy from chains (like OpenPI)
        # value_t is now [B] (single value per sample, not per denoising step)
        log_probs, value_t, entropy = self.get_log_prob_value(
            observation,
            chains,
            denoise_inds,
            compute_values,
        )

        # Extract action chunk dimensions (like OpenPI)
        action_chunk = self.n_action_steps
        action_env_dim = self.action_dim

        # Post-process outputs (like OpenPI)
        # fix for now
        log_probs = log_probs[
            :, :, :, :action_env_dim
        ]
        entropy = entropy[
            :, :, :, :action_env_dim
        ]

        # Average over denoise steps and dimensions (like OpenPI)
        log_probs = log_probs.mean(dim=1)  # Average over denoise steps [B, action_chunk, action_dim]
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[:, None]  # [B, 1] to align with loss mask
        # value_t is already [B] - no need to average over denoising steps anymore

        return {
            "logprobs": log_probs,
            "values": value_t,  # [B] - single value per sample
            "entropy": entropy,
        }

    def get_logprobs(self,
                    obs_dict: Dict[str, torch.Tensor],
                    actions: torch.Tensor,
                    fixed_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute log probabilities of actions.

        Args:
            obs_dict: Observation dictionary
            actions: Actions to evaluate [B, n_action_steps, action_dim]
            fixed_noise: Fixed noise for reproducible evaluation

        Returns:
            log_probs: Log probabilities [B]
        """
        # Get action distribution
        data = {'observation': obs_dict}
        forward_result = self.default_forward(data, fixed_noise=fixed_noise, return_values=False)

        x_mean = forward_result['x_mean']  # [B, horizon, action_dim]
        x_std = forward_result['x_std']    # [B, horizon, action_dim]

        # Extract relevant time steps
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action_mean = x_mean[:, start:end]  # [B, n_action_steps, action_dim]
        action_std = x_std[:, start:end]    # [B, n_action_steps, action_dim]

        # Compute log probabilities under Gaussian distribution
        if self.safe_get_logprob:
            # Simplified version without std for stability
            log_probs = -0.5 * ((actions - action_mean) ** 2).sum(dim=[1, 2])
        else:
            # Full Gaussian log probability
            # log p(a) = -0.5 * ((a - mean) / std)^2 - log(std) - 0.5 * log(2π)
            eps = 1e-8
            normalized_actions = (actions - action_mean) / (action_std + eps)
            log_probs = -0.5 * (normalized_actions ** 2 + torch.log(2 * torch.pi * (action_std + eps) ** 2))
            log_probs = log_probs.sum(dim=[1, 2])  # Sum over action steps and dimensions

        return log_probs

    def get_action_and_logprobs(self,
                               obs_dict: Dict[str, torch.Tensor],
                               fixed_noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both actions and their log probabilities.

        Args:
            obs_dict: Observation dictionary
            fixed_noise: Fixed noise for reproducible sampling

        Returns:
            actions: Sampled actions [B, n_action_steps, action_dim]
            log_probs: Log probabilities [B]
        """
        # Get actions
        action_result = self.predict_action(obs_dict)
        actions = action_result['action']

        # Get log probabilities
        log_probs = self.get_logprobs(obs_dict, actions, fixed_noise)

        return actions, log_probs

    # ========= Training methods =========
    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set the data normalizer."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: Dict, ema_model=None):
        """Compute training loss (keep original training interface)."""
        # This should call the original ManiFlow training logic
        # For now, raise not implemented to focus on RL interface
        raise NotImplementedError("Training loss computation not implemented in RL policy. Use original ManiFlowTransformerPointcloudPolicy for training.")

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
