"""Deep State-Space Day Encoder (DSSDE) v3 -- full model."""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .encoder import DSSEncoder
from .decoder import ObservationDecoder
from .vq import VectorQuantizer
from .loss import CompositeLoss


class DSSDE(nn.Module):
    """Semantic Day Embedding model (v3).

    Combines:
      1. Modality encoders + cross-modal fusion + GRU state transition
      2. Observation reconstruction decoder (modality-weighted)
      3. Optional VQ layer for regime discovery
      4. Composite loss: CPC + weighted_rec + smoothness + VQ + ADS + macro
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        total_feature_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_attention_heads: int = 4,
        dropout: float = 0.1,
        transition_type: str = "gru",
        hierarchical_windows: tuple = (1, 5, 21),
        use_spectral_norm: bool = False,
        use_vq: bool = True,
        vq_codebook_size: int = 16,
        vq_ema_decay: float = 0.99,
        vq_commitment_weight: float = 0.25,
        cpc_horizons: tuple = (1, 5, 21, 63, 126),
        cpc_temperature: float = 0.07,
        cpc_n_negatives: int = 128,
        alpha_rec: float = 1.0,
        beta_smo: float = 0.01,
        beta_var: float = 0.0,
        use_vicreg: bool = False,
        vicreg_var_weight: float = 0.0,
        vicreg_cov_weight: float = 0.0,
        vicreg_gamma: float = 1.0,
        use_hmm_prior: bool = False,
        hmm_n_states: int = 8,
        hmm_weight: float = 0.0,
        gamma_vq: float = 0.1,
        lambda_ads: float = 0.5,
        lambda_macro: float = 0.3,
        n_macro_targets: int = 3,
        macro_horizon: int = 21,
        feature_weights: Optional[torch.Tensor] = None,
        regime_aware_smooth: bool = False,
        regime_contrastive_weight: float = 0.0,
        regime_contrastive_temp: float = 0.5,
        macro_contrastive_n_bins: int = 8,
        smooth_transition_penalty: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_vq = use_vq
        self.modality_names = sorted(modality_dims.keys())

        self.encoder = DSSEncoder(
            modality_dims=modality_dims,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            n_attention_heads=n_attention_heads,
            dropout=dropout,
            transition_type=transition_type,
            hierarchical_windows=hierarchical_windows,
        )
        if use_spectral_norm:
            self.encoder.apply_spectral_norm()

        self.decoder = ObservationDecoder(
            embed_dim=embed_dim,
            output_dim=total_feature_dim,
            hidden_dim=hidden_dim * 2,
        )

        self.vq = (
            VectorQuantizer(embed_dim, vq_codebook_size, vq_ema_decay, vq_commitment_weight)
            if use_vq
            else None
        )

        self.loss_fn = CompositeLoss(
            embed_dim=embed_dim,
            cpc_horizons=list(cpc_horizons),
            cpc_temperature=cpc_temperature,
            cpc_n_negatives=cpc_n_negatives,
            alpha_rec=alpha_rec,
            beta_smo=beta_smo,
            beta_var=beta_var,
            use_vicreg=use_vicreg,
            vicreg_var_weight=vicreg_var_weight,
            vicreg_cov_weight=vicreg_cov_weight,
            vicreg_gamma=vicreg_gamma,
            use_hmm_prior=use_hmm_prior,
            hmm_n_states=hmm_n_states,
            hmm_weight=hmm_weight,
            gamma_vq=gamma_vq,
            lambda_ads=lambda_ads,
            lambda_macro=lambda_macro,
            n_macro_targets=n_macro_targets,
            macro_horizon=macro_horizon,
            feature_weights=feature_weights,
            regime_aware_smooth=regime_aware_smooth,
            regime_contrastive_weight=regime_contrastive_weight,
            regime_contrastive_temp=regime_contrastive_temp,
            macro_contrastive_n_bins=macro_contrastive_n_bins,
            smooth_transition_penalty=smooth_transition_penalty,
        )

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        x_full: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ads_targets: Optional[torch.Tensor] = None,
        ads_mask: Optional[torch.Tensor] = None,
        macro_targets: Optional[torch.Tensor] = None,
        macro_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        z_seq = self.encoder(modality_inputs)

        vq_loss = None
        regime_indices = None
        z_for_decode = z_seq

        if self.use_vq and self.vq is not None:
            z_q, regime_indices, vq_loss = self.vq(z_seq)
            z_for_decode = z_q

        x_hat = self.decoder(z_for_decode)

        losses = self.loss_fn(
            z_seq, x_hat, x_full,
            vq_loss=vq_loss, mask=mask,
            ads_targets=ads_targets, ads_mask=ads_mask,
            macro_targets=macro_targets, macro_mask=macro_mask,
            regime_indices=regime_indices,
        )

        result = {
            "z_seq": z_seq,
            "z_quantized": z_for_decode,
            "x_hat": x_hat,
            "losses": losses,
        }
        if regime_indices is not None:
            result["regime_indices"] = regime_indices
        return result

    @torch.no_grad()
    def embed(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        return self.encoder(modality_inputs)

    @torch.no_grad()
    def embed_and_quantize(self, modality_inputs: Dict[str, torch.Tensor]) -> tuple:
        self.eval()
        z_seq = self.encoder(modality_inputs)
        if self.use_vq and self.vq is not None:
            z_q, indices, _ = self.vq(z_seq)
            return z_seq, z_q, indices
        return z_seq, z_seq, None
