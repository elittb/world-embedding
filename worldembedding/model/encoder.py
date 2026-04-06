"""Modality encoders, cross-modal attention fusion, and GRU state transition."""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Sequence
from torch.nn.utils.parametrizations import spectral_norm


class ModalityEncoder(nn.Module):
    """2-layer MLP encoder for a single modality block."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalFusion(nn.Module):
    """Multi-head attention with a learnable query to fuse modality embeddings
    into a single innovation vector h_t."""

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, modality_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modality_embeddings: (batch, M, hidden_dim) -- M modality tokens
        Returns:
            h_t: (batch, hidden_dim) -- fused innovation vector
        """
        B = modality_embeddings.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, modality_embeddings, modality_embeddings)
        return self.norm(out.squeeze(1))


class GRUTransition(nn.Module):
    """State transition via GRU cell with a linear projection head.

    The GRU operates in its own internal space (avoiding tanh saturation
    in the output embedding), then a learned linear projection maps to
    the final embedding space.
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        gru_hidden = hidden_dim
        self.input_proj = nn.Linear(hidden_dim, gru_hidden)
        self.cell = nn.GRUCell(input_size=gru_hidden, hidden_size=gru_hidden)
        self.output_proj = nn.Sequential(
            nn.Linear(gru_hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.gru_hidden = gru_hidden

    def forward(self, z_prev: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_prev: (batch, embed_dim) -- previous *projected* embedding
            h_t:    (batch, hidden_dim) -- innovation vector
        Returns:
            z_t:    (batch, embed_dim)
        """
        h_proj = self.input_proj(h_t)
        # Run GRU in its internal space; z_prev feeds through a learned
        # gate so dimension mismatch is handled by storing internal state
        if not hasattr(self, '_h_internal') or self._h_internal is None or self._h_internal.shape[0] != z_prev.shape[0]:
            self._h_internal = torch.zeros(z_prev.shape[0], self.gru_hidden, device=z_prev.device)
        self._h_internal = self.cell(h_proj, self._h_internal)
        return self.output_proj(self._h_internal)

    def reset_state(self):
        self._h_internal = None


class AdditiveTransition(nn.Module):
    """Additive SSM ablation variant: z_t = f(z_{t-1}) + g(h_t)."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.g = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z_prev: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        return self.norm(self.f(z_prev) + self.g(h_t))


class DSSEncoder(nn.Module):
    """Full encoder pipeline: modality encoders -> fusion -> state transition.

    Processes an entire sequence and returns the embedding trajectory.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 128,
        embed_dim: int = 64,
        n_attention_heads: int = 4,
        dropout: float = 0.1,
        transition_type: str = "gru",
        hierarchical_windows: Sequence[int] = (1,),
    ):
        super().__init__()
        self.modality_names: List[str] = sorted(modality_dims.keys())
        self.embed_dim = embed_dim
        self.hierarchical_windows = tuple(int(w) for w in hierarchical_windows)
        if len(self.hierarchical_windows) == 0:
            self.hierarchical_windows = (1,)
        if any(w <= 0 for w in self.hierarchical_windows):
            raise ValueError("hierarchical_windows must be positive integers")

        # If hierarchical, each modality gets concatenated [day, week, month]-style views
        enc_dims = {
            name: dim * len(self.hierarchical_windows) for name, dim in modality_dims.items()
        }
        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(enc_dims[name], hidden_dim, dropout)
            for name in modality_dims.keys()
        })

        self.fusion = CrossModalFusion(hidden_dim, n_attention_heads, dropout)

        if transition_type == "gru":
            self.transition = GRUTransition(embed_dim, hidden_dim)
        elif transition_type == "additive":
            self.transition = AdditiveTransition(embed_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown transition_type: {transition_type}")

    def apply_spectral_norm(self):
        """Apply spectral normalization to Linear layers in encoder blocks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                try:
                    spectral_norm(m)
                except Exception:
                    # Best-effort; if already parametrized, ignore
                    pass

    @staticmethod
    def _rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor:
        """Causal rolling mean over time dimension (includes current t)."""
        if window == 1:
            return x
        B, T, D = x.shape
        c = torch.cumsum(x, dim=1)
        c0 = torch.zeros((B, 1, D), device=x.device, dtype=x.dtype)
        c = torch.cat([c0, c], dim=1)  # (B, T+1, D)
        idx = torch.arange(T, device=x.device)
        start = (idx + 1 - window).clamp(min=0)
        end = idx + 1
        sum_w = c[:, end, :] - c[:, start, :]
        denom = (end - start).to(x.dtype).clamp(min=1).view(1, T, 1)
        return sum_w / denom

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        z_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            modality_inputs: dict mapping modality name -> (batch, seq_len, dim_m)
            z_init: optional (batch, embed_dim) initial state; zeros if None
        Returns:
            z_seq: (batch, seq_len, embed_dim) -- embedding trajectory
        """
        first_key = self.modality_names[0]
        B, T, _ = modality_inputs[first_key].shape
        device = modality_inputs[first_key].device

        if z_init is None:
            z_init = torch.zeros(B, self.embed_dim, device=device)

        # Reset GRU internal state for each new sequence
        if hasattr(self.transition, 'reset_state'):
            self.transition.reset_state()

        # Hierarchical temporal aggregation: concat rolling means at multiple horizons
        if len(self.hierarchical_windows) > 1 or self.hierarchical_windows[0] != 1:
            hier_inputs: Dict[str, torch.Tensor] = {}
            for name in self.modality_names:
                x = modality_inputs[name]
                views = [self._rolling_mean(x, w) for w in self.hierarchical_windows]
                hier_inputs[name] = torch.cat(views, dim=-1)
        else:
            hier_inputs = modality_inputs

        z_seq = []
        z_prev = z_init

        for t in range(T):
            mod_embs = []
            for name in self.modality_names:
                x_m = hier_inputs[name][:, t, :]
                mod_embs.append(self.encoders[name](x_m))

            mod_stack = torch.stack(mod_embs, dim=1)  # (B, M, hidden_dim)
            h_t = self.fusion(mod_stack)
            z_t = self.transition(z_prev, h_t)

            z_seq.append(z_t)
            z_prev = z_t

        return torch.stack(z_seq, dim=1)  # (B, T, embed_dim)
