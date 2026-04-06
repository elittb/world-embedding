"""Vector Quantization layer for unsupervised regime discovery.

Implements VQ-VAE codebook with EMA updates (van den Oord et al. 2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Quantizes continuous embeddings to a discrete codebook via nearest-neighbor
    lookup, with EMA codebook updates for training stability."""

    def __init__(
        self,
        embed_dim: int,
        codebook_size: int = 16,
        ema_decay: float = 0.99,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.ema_decay = ema_decay
        self.commitment_weight = commitment_weight

        self.codebook = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed_sum", self.codebook.weight.data.clone())

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (..., embed_dim) continuous embeddings

        Returns:
            z_q: (..., embed_dim) quantized embeddings (straight-through)
            indices: (...) codebook indices (regime labels)
            vq_loss: scalar VQ loss
        """
        flat_z = z.reshape(-1, self.embed_dim)

        # Nearest codebook entry
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = distances.argmin(dim=1)
        z_q_flat = self.codebook(indices)

        if self.training:
            self._ema_update(flat_z, indices)

        # Losses
        commitment_loss = F.mse_loss(z.reshape(-1, self.embed_dim), z_q_flat.detach())
        codebook_loss = F.mse_loss(z_q_flat, flat_z.detach())
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator
        z_q_flat = flat_z + (z_q_flat - flat_z).detach()

        z_q = z_q_flat.reshape(z.shape)
        indices = indices.reshape(z.shape[:-1])

        return z_q, indices, vq_loss

    @torch.no_grad()
    def _ema_update(self, flat_z: torch.Tensor, indices: torch.Tensor):
        one_hot = F.one_hot(indices, self.codebook_size).float()
        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.t() @ flat_z

        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        self.ema_embed_sum.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
        self.codebook.weight.data.copy_(self.ema_embed_sum / smoothed.unsqueeze(1))
