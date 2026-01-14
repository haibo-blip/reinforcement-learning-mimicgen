"""
Value Head for RL training.

A simple MLP that estimates state values from observation features.
"""

import torch.nn as nn


class ValueHead(nn.Module):
    """Value estimation head for RL training."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes=(512, 128),
        output_dim: int = 1,
        activation: str = "gelu",  # 'relu' or 'gelu'
        bias_last: bool = False,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        if activation.lower() == "relu":
            act = nn.ReLU
        elif activation.lower() == "gelu":
            act = nn.GELU
        elif activation.lower() == "tanh":
            act = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act())
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim, bias=bias_last))

        self.mlp = nn.Sequential(*layers)

        self._init_weights(activation.lower())

    def _init_weights(self, nonlinearity="relu"):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                if m is self.mlp[-1]:
                    # Final layer: small init for stable value estimates
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    # Hidden layers: kaiming init
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity=nonlinearity
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Input features [B, input_dim]
        Returns:
            values: State values [B, output_dim]
        """
        return self.mlp(x)
