"""Observation reconstruction decoder: z_t -> x_hat_t."""

import torch
import torch.nn as nn


class ObservationDecoder(nn.Module):
    """2-layer MLP that reconstructs the full observable vector from z_t."""

    def __init__(self, embed_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, [seq_len,] embed_dim)
        Returns:
            x_hat: (batch, [seq_len,] output_dim)
        """
        return self.net(z)
