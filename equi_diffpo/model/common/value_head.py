"""
Value Head for RL training.

A simple MLP that estimates state values from observation features.
Includes attention pooling for processing variable-length visual features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """
    Attention-based pooling for variable-length sequences.

    Learns to weight tokens based on their relevance for value estimation,
    preserving more information than simple mean pooling.
    """

    def __init__(self, feature_dim: int, num_heads: int = 1):
        """
        Args:
            feature_dim: Dimension of input features
            num_heads: Number of attention heads (1 = single attention, >1 = multi-head)
        """
        super().__init__()
        self.num_heads = num_heads

        if num_heads == 1:
            # Simple single-head attention
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.Tanh(),
                nn.Linear(feature_dim // 4, 1),
            )
        else:
            # Multi-head attention with learnable query
            self.query = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
            self.attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)

        self._init_weights()

    def _init_weights(self):
        if self.num_heads == 1:
            for m in self.attention:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, L, D] where L is sequence length
        Returns:
            Pooled features [B, D]
        """
        if self.num_heads == 1:
            # Single-head attention pooling
            weights = self.attention(x)  # [B, L, 1]
            weights = F.softmax(weights, dim=1)  # normalize over sequence
            pooled = (x * weights).sum(dim=1)  # [B, D]
        else:
            # Multi-head attention pooling
            B = x.size(0)
            query = self.query.expand(B, -1, -1)  # [B, 1, D]
            pooled, _ = self.attn(query, x, x)  # [B, 1, D]
            pooled = pooled.squeeze(1)  # [B, D]

        return pooled


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
