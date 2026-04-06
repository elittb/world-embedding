"""Composite loss functions for DSSDE v4 training.

Includes:
  - CPC (multi-horizon InfoNCE with inverse-sqrt horizon weights)
  - Weighted reconstruction (modality-aware feature weights)
  - Smoothness regularizer
  - ADS nowcasting loss (semi-supervised)
  - Multi-horizon macro forecasting loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CPCLoss(nn.Module):
    """Contrastive Predictive Coding loss with inverse-sqrt horizon weights.

    Longer horizons get lower weight to prevent the model from learning
    dominant periodic modes just to satisfy long-horizon predictability.
    """

    def __init__(
        self,
        embed_dim: int,
        horizons: List[int] = (1, 5, 21),
        temperature: float = 0.07,
        n_negatives: int = 128,
    ):
        super().__init__()
        self.horizons = list(horizons)
        self.temperature = temperature
        self.n_negatives = n_negatives

        self.predictors = nn.ModuleDict({
            str(k): nn.Linear(embed_dim, embed_dim, bias=False)
            for k in self.horizons
        })

        raw_w = {k: 1.0 / math.sqrt(k) for k in self.horizons}
        total_w = sum(raw_w.values())
        self.horizon_weights = {k: w / total_w for k, w in raw_w.items()}

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        B, T, D = z_seq.shape
        total_loss = torch.tensor(0.0, device=z_seq.device)

        for k in self.horizons:
            if k >= T:
                continue

            z_anchor = z_seq[:, :T - k, :]
            z_positive = z_seq[:, k:, :]
            L = T - k

            z_pred = self.predictors[str(k)](z_anchor)

            z_pred_flat = F.normalize(z_pred.reshape(B * L, D), dim=-1)
            z_pos_flat = F.normalize(z_positive.reshape(B * L, D), dim=-1).detach()

            all_z = F.normalize(z_seq.reshape(B * T, D), dim=-1).detach()
            neg_idx = torch.randint(0, B * T, (B * L, self.n_negatives), device=z_seq.device)
            z_neg = all_z[neg_idx]

            pos_logit = (z_pred_flat * z_pos_flat).sum(dim=-1, keepdim=True) / self.temperature
            neg_logits = torch.bmm(z_neg, z_pred_flat.unsqueeze(-1)).squeeze(-1) / self.temperature

            logits = torch.cat([pos_logit, neg_logits], dim=1)
            labels = torch.zeros(B * L, dtype=torch.long, device=z_seq.device)

            total_loss = total_loss + self.horizon_weights[k] * F.cross_entropy(logits, labels)

        # Return computed CPC in both train and eval (do not zero out under torch.no_grad()).
        return total_loss


class WeightedReconstructionLoss(nn.Module):
    """MSE reconstruction with per-feature weights."""

    def __init__(self, feature_weights: Optional[torch.Tensor] = None):
        super().__init__()
        if feature_weights is not None:
            self.register_buffer("w", feature_weights)
        else:
            self.w = None

    def forward(
        self,
        x_hat: torch.Tensor,
        x_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = (x_hat - x_true).pow(2)
        if self.w is not None:
            diff = diff * self.w.unsqueeze(0).unsqueeze(0)
        if mask is not None:
            diff = diff * mask
            return diff.sum() / mask.sum().clamp(min=1)
        return diff.mean()


class SmoothnessLoss(nn.Module):
    """Temporal smoothness regularizer: mean ||z_t - z_{t-1}||^2."""

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        diffs = z_seq[:, 1:, :] - z_seq[:, :-1, :]
        return diffs.pow(2).mean()


class RegimeAwareSmoothnessLoss(nn.Module):
    """Smoothness that respects regime boundaries.

    Penalizes temporal jitter *within* the same VQ code but allows sharp
    transitions *across* different codes.  This creates the critical property:
    smooth intra-regime dynamics + discrete inter-regime jumps.
    """

    def __init__(self, transition_penalty: float = 0.0):
        super().__init__()
        self.transition_penalty = transition_penalty

    def forward(self, z_seq: torch.Tensor, regime_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        diffs = z_seq[:, 1:, :] - z_seq[:, :-1, :]
        sq_diffs = diffs.pow(2).mean(dim=-1)

        if regime_indices is None:
            return sq_diffs.mean()

        same_regime = (regime_indices[:, 1:] == regime_indices[:, :-1]).float()
        n_same = same_regime.sum().clamp(min=1)
        intra_loss = (sq_diffs * same_regime).sum() / n_same

        if self.transition_penalty > 0:
            diff_regime = 1.0 - same_regime
            n_diff = diff_regime.sum().clamp(min=1)
            inter_loss = (sq_diffs * diff_regime).sum() / n_diff
            return intra_loss - self.transition_penalty * inter_loss.clamp(max=intra_loss.detach())
        return intra_loss


class MacroContrastiveLoss(nn.Module):
    """Contrastive loss grounded in observable macro data (ADS).

    Uses ADS business conditions values to define "same regime" vs "different
    regime".  Days whose ADS falls in the same quantile bin are positives;
    different bins are negatives.

    v14 fixes over v13:
      - Global ADS percentile boundaries (computed once from training data)
        instead of per-batch quantiles, giving consistent bin assignments.
      - 8 bins (octiles) instead of 4, halving positives per anchor, making
        the task harder and the gradient signal ~50× stronger.
      - Temperature 0.1 instead of 0.5, sharper softmax gradients, matching
        TS2Vec / SupCon best practices.

    Design follows Khosla et al. (2020) Supervised Contrastive Learning and
    the temporal contrastive paradigm of TS2Vec (Yue et al., AAAI 2022),
    adapted for macro-finance.
    """

    def __init__(self, temperature: float = 0.1, n_bins: int = 8,
                 max_pairs: int = 2048):
        super().__init__()
        self.temperature = temperature
        self.n_bins = n_bins
        self.max_pairs = max_pairs
        self.register_buffer(
            "global_boundaries", torch.zeros(max(n_bins - 1, 1))
        )

    @property
    def boundaries_set(self) -> bool:
        """True when global boundaries have been computed (survives save/load)."""
        return bool(self.global_boundaries.abs().sum() > 0)

    def set_global_boundaries(self, ads_values, ads_mask=None):
        """Pre-compute fixed ADS quantile boundaries from training data.

        Call once before training begins so every batch uses the same bin
        edges, eliminating the label-noise from per-batch quantiles.
        """
        import numpy as np
        if ads_mask is not None:
            valid = ads_values[ads_mask > 0.5]
        else:
            valid = ads_values[~np.isnan(ads_values)]
        if len(valid) < self.n_bins:
            return
        quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        boundaries = np.quantile(valid, quantiles)
        self.global_boundaries = torch.tensor(
            boundaries, dtype=torch.float32
        )

    def forward(self, z_seq: torch.Tensor,
                ads_targets: torch.Tensor,
                ads_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = z_seq.shape
        z_flat = z_seq.reshape(B * T, D)
        ads_flat = ads_targets.reshape(B * T)
        mask_flat = ads_mask.reshape(B * T)

        valid = mask_flat > 0.5
        if valid.sum() < 8:
            return torch.tensor(0.0, device=z_seq.device, requires_grad=True)

        z_v = z_flat[valid]
        ads_v = ads_flat[valid]
        N = z_v.shape[0]

        n_sample = min(self.max_pairs, N)
        idx = torch.randperm(N, device=z_seq.device)[:n_sample]
        z_s = F.normalize(z_v[idx], dim=-1)
        ads_s = ads_v[idx]

        if self.boundaries_set:
            boundaries = self.global_boundaries.to(z_seq.device)
        else:
            quantiles = torch.linspace(0, 1, self.n_bins + 1,
                                       device=z_seq.device)
            boundaries = torch.quantile(ads_s, quantiles)[1:-1]

        bins = torch.bucketize(ads_s, boundaries)

        sim = z_s @ z_s.t() / self.temperature
        same_bin = (bins.unsqueeze(0) == bins.unsqueeze(1)).float()
        diag_mask = 1.0 - torch.eye(n_sample, device=z_seq.device)
        same_bin = same_bin * diag_mask

        if same_bin.sum() < 1:
            return torch.tensor(0.0, device=z_seq.device, requires_grad=True)

        log_exp_sim = torch.log(
            (torch.exp(sim) * diag_mask).sum(dim=1).clamp(min=1e-8)
        )
        log_prob = sim - log_exp_sim.unsqueeze(1)

        pos_per_row = same_bin.sum(dim=1).clamp(min=1)
        loss = -(same_bin * log_prob).sum(dim=1) / pos_per_row
        has_pos = same_bin.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=z_seq.device, requires_grad=True)
        return loss[has_pos].mean()


class EmbeddingVarianceLoss(nn.Module):
    """Penalize low variance across batch to prevent collapse.
    Encourages embeddings to stay spread out (addresses semantic similarity ~1)."""

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        z_flat = z_seq.reshape(-1, z_seq.shape[-1])
        var = z_flat.var(dim=0).mean()
        return -torch.log(var + 1e-6)  # encourage variance


class VICRegLoss(nn.Module):
    """VICReg-style collapse prevention.

    Adds:
      - variance term: encourages per-dimension std >= gamma
      - covariance term: decorrelates dimensions (off-diagonal covariance -> 0)
    """

    def __init__(self, gamma: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, z_seq: torch.Tensor) -> dict:
        # Flatten across (batch, time) to maximize sample size
        z = z_seq.reshape(-1, z_seq.shape[-1])
        z = z - z.mean(dim=0, keepdim=True)

        # Variance: hinge on std
        std = torch.sqrt(z.var(dim=0) + self.eps)
        var_loss = torch.mean(F.relu(self.gamma - std))

        # Covariance: off-diagonal penalty on normalized covariance
        n = z.shape[0]
        if n <= 1:
            cov_loss = torch.tensor(0.0, device=z_seq.device, requires_grad=True)
        else:
            cov = (z.T @ z) / (n - 1)
            d = cov.shape[0]
            off = cov - torch.diag(torch.diag(cov))
            cov_loss = (off.pow(2).sum() / (d * (d - 1))).clamp(min=0.0)

        return {"vicreg_var": var_loss, "vicreg_cov": cov_loss}


class HMMPriorLoss(nn.Module):
    """Discrete HMM prior on latent trajectory z_t.

    Emissions: diagonal Gaussian for each state k over z_t.
    Transitions: learned KxK matrix.
    Loss: negative log-likelihood via forward algorithm (log-space).
    """

    def __init__(self, embed_dim: int, n_states: int = 8, eps: float = 1e-4):
        super().__init__()
        self.n_states = int(n_states)
        self.embed_dim = int(embed_dim)
        self.eps = eps

        self.init_logits = nn.Parameter(torch.zeros(self.n_states))
        self.trans_logits = nn.Parameter(torch.zeros(self.n_states, self.n_states))
        self.means = nn.Parameter(torch.randn(self.n_states, self.embed_dim) * 0.02)
        self.log_vars = nn.Parameter(torch.zeros(self.n_states, self.embed_dim))

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        # z_seq: (B,T,D)
        B, T, D = z_seq.shape
        K = self.n_states

        z = z_seq
        means = self.means  # (K,D)
        vars_ = torch.exp(self.log_vars).clamp(min=self.eps)  # (K,D)

        # log emission prob p(z_t | s=k)
        # shape: (B,T,K)
        diff = z.unsqueeze(2) - means.view(1, 1, K, D)
        log_det = torch.sum(torch.log(vars_), dim=-1).view(1, 1, K)
        quad = torch.sum(diff.pow(2) / vars_.view(1, 1, K, D), dim=-1)
        log_emit = -0.5 * (D * math.log(2 * math.pi) + log_det + quad)

        log_pi = F.log_softmax(self.init_logits, dim=-1)  # (K,)
        log_A = F.log_softmax(self.trans_logits, dim=-1)  # (K,K) rows sum to 1

        # Forward algorithm in log-space
        alpha = log_pi.view(1, K) + log_emit[:, 0, :]  # (B,K)
        for t in range(1, T):
            # alpha_new[k] = log_emit[t,k] + logsumexp_j alpha[j] + log_A[j,k]
            alpha = log_emit[:, t, :] + torch.logsumexp(alpha.unsqueeze(2) + log_A.view(1, K, K), dim=1)

        loglik = torch.logsumexp(alpha, dim=1)  # (B,)
        nll = -loglik.mean()
        return nll


class ADSNowcastLoss(nn.Module):
    """Semi-supervised loss: predict ADS from z_t on valid days."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, 1)

    def forward(
        self,
        z_seq: torch.Tensor,
        ads_targets: Optional[torch.Tensor] = None,
        ads_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if ads_targets is None or ads_mask is None:
            return torch.tensor(0.0, device=z_seq.device, requires_grad=True)

        pred = self.head(z_seq).squeeze(-1)
        valid = ads_mask > 0.5
        if valid.sum() == 0:
            return torch.tensor(0.0, device=z_seq.device, requires_grad=True)

        return F.mse_loss(pred[valid], ads_targets[valid])


class MacroForecastLoss(nn.Module):
    """Predict future macro changes from z_t.

    For each target, we predict k-step-ahead values using a separate
    linear head per target. Only active on days where the target is available.
    """

    def __init__(self, embed_dim: int, n_targets: int, horizon: int = 21):
        super().__init__()
        self.horizon = horizon
        self.heads = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(n_targets)])

    def forward(
        self,
        z_seq: torch.Tensor,
        macro_targets: Optional[torch.Tensor] = None,
        macro_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if macro_targets is None or macro_mask is None:
            return torch.tensor(0.0, device=z_seq.device, requires_grad=True)

        B, T, D = z_seq.shape
        n_tgt = macro_targets.shape[-1]
        k = self.horizon

        if T <= k:
            return torch.tensor(0.0, device=z_seq.device, requires_grad=True)

        total = torch.tensor(0.0, device=z_seq.device)
        n_valid = 0

        z_anchor = z_seq[:, :T - k, :]

        for i in range(min(n_tgt, len(self.heads))):
            future_val = macro_targets[:, k:, i]
            future_mask = macro_mask[:, k:, i]
            valid = future_mask > 0.5

            if valid.sum() == 0:
                continue

            pred = self.heads[i](z_anchor).squeeze(-1)
            total = total + F.mse_loss(pred[valid], future_val[valid])
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=z_seq.device, requires_grad=True)
        return total / n_valid


class CompositeLoss(nn.Module):
    """Weighted combination of all DSSDE training objectives."""

    def __init__(
        self,
        embed_dim: int,
        cpc_horizons: List[int] = (1, 5, 21, 63, 126),
        cpc_temperature: float = 0.07,
        cpc_n_negatives: int = 128,
        alpha_rec: float = 1.0,
        beta_smo: float = 0.01,
        beta_var: float = 0.0,
        use_vicreg: bool = False,
        vicreg_var_weight: float = 1.0,
        vicreg_cov_weight: float = 1.0,
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
        self.cpc = CPCLoss(embed_dim, cpc_horizons, cpc_temperature, cpc_n_negatives)
        self.rec = WeightedReconstructionLoss(feature_weights)
        self.smo = SmoothnessLoss()
        self.regime_smo = RegimeAwareSmoothnessLoss(transition_penalty=smooth_transition_penalty)
        self.var_loss = EmbeddingVarianceLoss()
        self.vicreg = VICRegLoss(gamma=vicreg_gamma)
        self.hmm = HMMPriorLoss(embed_dim=embed_dim, n_states=hmm_n_states)
        self.ads_loss = ADSNowcastLoss(embed_dim)
        self.macro_loss = MacroForecastLoss(embed_dim, n_macro_targets, macro_horizon)
        self.macro_contrast = MacroContrastiveLoss(
            temperature=regime_contrastive_temp,
            n_bins=macro_contrastive_n_bins,
        )

        self.alpha_rec = alpha_rec
        self.beta_smo = beta_smo
        self.beta_var = beta_var
        self.use_vicreg = use_vicreg
        self.vicreg_var_weight = vicreg_var_weight
        self.vicreg_cov_weight = vicreg_cov_weight
        self.use_hmm_prior = use_hmm_prior
        self.hmm_weight = hmm_weight
        self.gamma_vq = gamma_vq
        self.lambda_ads = lambda_ads
        self.lambda_macro = lambda_macro
        self.regime_aware_smooth = regime_aware_smooth
        self.regime_contrastive_weight = regime_contrastive_weight

    def forward(
        self,
        z_seq: torch.Tensor,
        x_hat: torch.Tensor,
        x_true: torch.Tensor,
        vq_loss: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        ads_targets: Optional[torch.Tensor] = None,
        ads_mask: Optional[torch.Tensor] = None,
        macro_targets: Optional[torch.Tensor] = None,
        macro_mask: Optional[torch.Tensor] = None,
        regime_indices: Optional[torch.Tensor] = None,
    ) -> dict:
        l_cpc = self.cpc(z_seq)
        l_rec = self.rec(x_hat, x_true, mask)
        l_ads = self.ads_loss(z_seq, ads_targets, ads_mask)
        l_macro = self.macro_loss(z_seq, macro_targets, macro_mask)

        if self.regime_aware_smooth and regime_indices is not None:
            l_smo = self.regime_smo(z_seq, regime_indices)
        else:
            l_smo = self.smo(z_seq)

        total = (
            l_cpc
            + self.alpha_rec * l_rec
            + self.beta_smo * l_smo
            + self.lambda_ads * l_ads
            + self.lambda_macro * l_macro
        )

        components = {
            "cpc": l_cpc,
            "rec": l_rec,
            "smo": l_smo,
            "ads": l_ads,
            "macro": l_macro,
        }

        if self.regime_contrastive_weight > 0 and ads_targets is not None and ads_mask is not None:
            l_rc = self.macro_contrast(z_seq, ads_targets, ads_mask)
            total = total + self.regime_contrastive_weight * l_rc
            components["macro_contrast"] = l_rc

        if self.beta_var > 0:
            l_var = self.var_loss(z_seq)
            total = total + self.beta_var * l_var
            components["var"] = l_var

        if self.use_vicreg and (self.vicreg_var_weight > 0 or self.vicreg_cov_weight > 0):
            vic = self.vicreg(z_seq)
            if self.vicreg_var_weight > 0:
                total = total + self.vicreg_var_weight * vic["vicreg_var"]
                components["vicreg_var"] = vic["vicreg_var"]
            if self.vicreg_cov_weight > 0:
                total = total + self.vicreg_cov_weight * vic["vicreg_cov"]
                components["vicreg_cov"] = vic["vicreg_cov"]

        if self.use_hmm_prior and self.hmm_weight > 0:
            l_hmm = self.hmm(z_seq)
            total = total + self.hmm_weight * l_hmm
            components["hmm"] = l_hmm

        if vq_loss is not None:
            total = total + self.gamma_vq * vq_loss
            components["vq"] = vq_loss

        components["total"] = total
        return components
