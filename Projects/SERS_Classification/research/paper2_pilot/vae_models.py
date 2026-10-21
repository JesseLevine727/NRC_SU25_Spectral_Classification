#!/usr/bin/env python
"""Phase 3: small PyTorch models for the Paper 2 pilot (VAE ladder).

Four fixed architectures (no hyperparameter search):
  1. VAE                     8 latent dims, KL weight 1
  2. beta-VAE                same encoder/decoder, KL weight 4
  3. CVAE                    chemical (4) + substrate (4) [+ laser (2)] blocks,
                             one classification head per block, decoder sees latents only
  4. Physics-structured VAE  additive decoder with non-negative A_a / B_s,
                             a positive analyte x substrate gain G, a smooth laser
                             response, an overall gain and an offset

The 1-D conv encoder and conv decoder are shared by the ladder.  The physics
decoder's spectral templates (A_a, B_s) are the only parameters excluded from
the ~50k weight budget; they are reported separately.

All preprocessing is done outside this module.  Inputs are min-max scaled
spectra in [0, 1] on the block's common axis.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Encoder / decoder
# --------------------------------------------------------------------------- #
class ConvEncoder(nn.Module):
    """Small 1-D convolutional encoder shared by all four models.

    Four stride-4 convolutions followed by adaptive average pooling to 8
    positions and a linear layer to a 64-D feature vector.  This keeps the
    encoder at 26,480 trainable weights for either block length.
    """

    def __init__(self, in_len: int, feat_dim: int = 64):
        super().__init__()
        self.in_len = int(in_len)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=15, stride=4, padding=7),
            nn.GELU(),
            nn.Conv1d(8, 16, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Linear(32 * 8, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1))
        return self.fc(h.flatten(1))


class ConvDecoder(nn.Module):
    """Transposed-convolution decoder, interpolated to the exact input length."""

    def __init__(self, latent_dim: int, out_len: int):
        super().__init__()
        self.out_len = int(out_len)
        self.fc = nn.Linear(latent_dim, 16 * 8)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(16, 16, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.ConvTranspose1d(16, 8, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.ConvTranspose1d(8, 1, kernel_size=8, stride=4, padding=2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 16, 8)
        h = self.net(h)
        h = F.interpolate(h, size=self.out_len, mode="linear", align_corners=False)
        return h.squeeze(1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def inv_softplus(x: np.ndarray) -> np.ndarray:
    """Inverse of softplus, clipped to a numerically safe range."""
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-4, 30.0)
    return np.log(np.expm1(x)).astype(np.float32)


def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q||N(0,I)), summed over latent dims, averaged over the batch."""
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 30
    warmup_epochs: int = 50
    grad_clip: float = 5.0
    val_fraction: float = 0.10
    seed: int = 0


@dataclass
class FitResult:
    model: nn.Module
    best_epoch: int
    best_val: float
    epochs_run: int
    history: dict = field(default_factory=dict)
    failure: Optional[str] = None


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> dict:
    """Return trainable parameter counts, splitting physics templates out."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    template = 0
    if getattr(model, "kind", "") == "physics":
        template = sum(p.numel() for p in [model.A_raw, model.B_raw] if p.requires_grad)
    return {"trainable_total": int(total),
            "trainable_excl_physics_templates": int(total - template),
            "physics_templates": int(template)}


# --------------------------------------------------------------------------- #
# 1 / 2. VAE and beta-VAE
# --------------------------------------------------------------------------- #
class VAE(nn.Module):
    kind = "vae"

    def __init__(self, in_len: int, latent_dim: int = 8, beta: float = 1.0):
        super().__init__()
        self.in_len = int(in_len)
        self.latent_dim = int(latent_dim)
        self.beta = float(beta)
        self.encoder = ConvEncoder(in_len)
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.decoder = ConvDecoder(latent_dim, in_len)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return {"recon": self.decoder(z), "mu": mu, "logvar": logvar,
                "kl": kl_normal(mu, logvar)}

    @torch.no_grad()
    def latent_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)[0]


class BetaVAE(VAE):
    def __init__(self, in_len: int, latent_dim: int = 8, beta: float = 4.0):
        super().__init__(in_len, latent_dim=latent_dim, beta=beta)


# --------------------------------------------------------------------------- #
# 3. Conditional VAE with label-partitioned latents
# --------------------------------------------------------------------------- #
class CVAE(nn.Module):
    kind = "cvae"

    def __init__(self, in_len: int, dims=(4, 4), n_classes=(3, 4),
                 use_laser: bool = False, beta: float = 1.0):
        super().__init__()
        self.in_len = int(in_len)
        self.dims = tuple(int(d) for d in dims)
        self.n_classes = tuple(int(n) for n in n_classes)
        self.use_laser = bool(use_laser)
        self.beta = float(beta)
        self.block_names = ["chem", "sub"] + (["laser"] if use_laser else [])
        self.encoder = ConvEncoder(in_len)
        self.mu_heads = nn.ModuleList([nn.Linear(64, d) for d in self.dims])
        self.logvar_heads = nn.ModuleList([nn.Linear(64, d) for d in self.dims])
        self.cls_heads = nn.ModuleList([nn.Linear(d, n)
                                        for d, n in zip(self.dims, self.n_classes)])
        self.decoder = ConvDecoder(sum(self.dims), in_len)

    def encode_blocks(self, x: torch.Tensor):
        h = self.encoder(x)
        mus = [head(h) for head in self.mu_heads]
        logvars = [head(h) for head in self.logvar_heads]
        return mus, logvars

    def forward(self, x: torch.Tensor, y_chem=None, y_sub=None, y_laser=None):
        mus, logvars = self.encode_blocks(x)
        zs = []
        for mu, logvar in zip(mus, logvars):
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
            zs.append(z)
        recon = self.decoder(torch.cat(zs, dim=1))
        logits = [head(z) for head, z in zip(self.cls_heads, zs)]
        kl = sum(kl_normal(mu, lv) for mu, lv in zip(mus, logvars))
        ce = torch.zeros((), device=x.device)
        labels = [y_chem, y_sub] + ([y_laser] if self.use_laser else [])
        for lg, y in zip(logits, labels):
            if y is not None:
                ce = ce + F.cross_entropy(lg, y)
        return {"recon": recon, "mus": mus, "logvars": logvars,
                "logits": logits, "kl": kl, "ce": ce}

    @torch.no_grad()
    def latent_blocks(self, x: torch.Tensor):
        mus, _ = self.encode_blocks(x)
        return {name: mu for name, mu in zip(self.block_names, mus)}

    @torch.no_grad()
    def decode_blocks(self, blocks: dict) -> torch.Tensor:
        z = torch.cat([blocks[name] for name in self.block_names], dim=1)
        return self.decoder(z)


# --------------------------------------------------------------------------- #
# 4. Physics-structured VAE with an additive decoder
# --------------------------------------------------------------------------- #
class PhysicsVAE(nn.Module):
    """Additive physics decoder.

    recon = gain * r_laser(shift) * ( G_eff * A_mix + B_mix ) + offset
    A_mix = sum_a p_a A_a,  B_mix = sum_s q_s B_s,
    G_eff = sum_a sum_s p_a G[a,s] q_s.

    A_a and B_s are softplus-transformed learned spectra.  G is a softplus-
    transformed positive analyte x substrate matrix.  r_laser is a positive
    smooth polynomial response (degree <= 7, i.e. at most 8 coefficients);
    in block A it is fixed to 1.
    """

    kind = "physics"

    def __init__(self, in_len: int, n_analytes: int, n_substrates: int,
                 n_lasers: int = 1, use_laser: bool = False,
                 init_A: Optional[np.ndarray] = None,
                 init_B: Optional[np.ndarray] = None,
                 laser_poly: int = 8):
        super().__init__()
        self.in_len = int(in_len)
        self.n_analytes = int(n_analytes)
        self.n_substrates = int(n_substrates)
        self.n_lasers = int(n_lasers)
        self.use_laser = bool(use_laser)
        self.laser_poly = int(laser_poly)

        self.encoder = ConvEncoder(in_len)
        self.p_head = nn.Linear(64, self.n_analytes)
        self.q_head = nn.Linear(64, self.n_substrates)
        self.gain_head = nn.Linear(64, 1)
        self.offset_head = nn.Linear(64, 1)

        if init_A is None:
            init_A = np.full((self.n_analytes, in_len), 0.5, dtype=np.float32)
        if init_B is None:
            init_B = np.full((self.n_substrates, in_len), 0.5, dtype=np.float32)
        self.A_raw = nn.Parameter(torch.tensor(inv_softplus(init_A), dtype=torch.float32))
        self.B_raw = nn.Parameter(torch.tensor(inv_softplus(init_B), dtype=torch.float32))
        self.G_raw = nn.Parameter(torch.tensor(
            inv_softplus(np.full((self.n_analytes, self.n_substrates), 0.7)),
            dtype=torch.float32))

        t = torch.linspace(-1.0, 1.0, self.in_len)
        self.register_buffer("poly_t", torch.stack([t ** k for k in range(self.laser_poly)], 0))
        if self.use_laser:
            coef = torch.zeros(self.n_lasers, self.laser_poly)
            coef[:, 0] = float(inv_softplus(np.array([1.0]))[0])
            self.r_coef = nn.Parameter(coef)
        else:
            self.register_buffer("r_coef", torch.zeros(0, self.laser_poly))

    # -- physics pieces ----------------------------------------------------- #
    def analyte_spectra(self) -> torch.Tensor:
        return F.softplus(self.A_raw)

    def substrate_spectra(self) -> torch.Tensor:
        return F.softplus(self.B_raw)

    def gain_matrix(self) -> torch.Tensor:
        return F.softplus(self.G_raw)

    def laser_response(self, laser_idx: torch.Tensor) -> torch.Tensor:
        if not self.use_laser:
            return torch.ones((laser_idx.shape[0], self.in_len),
                              device=laser_idx.device)
        coef = self.r_coef[laser_idx.long()]                    # (B, K)
        poly = self.poly_t.unsqueeze(0)                         # (1, K, L)
        val = (coef.unsqueeze(-1) * poly).sum(dim=1)            # (B, L)
        return F.softplus(val) + 1e-3

    # -- forward ------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, laser_idx: torch.Tensor,
                y_chem=None, y_sub=None):
        h = self.encoder(x)
        p = F.softmax(self.p_head(h), dim=1)
        q = F.softmax(self.q_head(h), dim=1)
        gain = F.softplus(self.gain_head(h)).squeeze(1) + 1e-3
        offset = self.offset_head(h).squeeze(1)

        A = self.analyte_spectra()
        B = self.substrate_spectra()
        G = self.gain_matrix()
        A_mix = p @ A
        B_mix = q @ B
        G_eff = torch.einsum("ba,as,bs->b", p, G, q)
        r = self.laser_response(laser_idx)
        recon = gain.unsqueeze(1) * r * (G_eff.unsqueeze(1) * A_mix + B_mix) \
            + offset.unsqueeze(1)

        ce = torch.zeros((), device=x.device)
        if y_chem is not None:
            ce = ce + F.cross_entropy(self.p_head(h), y_chem)
        if y_sub is not None:
            ce = ce + F.cross_entropy(self.q_head(h), y_sub)
        return {"recon": recon, "p": p, "q": q, "gain": gain, "offset": offset,
                "ce": ce, "logits_p": self.p_head(h), "logits_q": self.q_head(h)}

    # -- inference helpers -------------------------------------------------- #
    @torch.no_grad()
    def probabilities(self, x: torch.Tensor, laser_idx: torch.Tensor):
        h = self.encoder(x)
        p = F.softmax(self.p_head(h), dim=1)
        q = F.softmax(self.q_head(h), dim=1)
        gain = F.softplus(self.gain_head(h)).squeeze(1) + 1e-3
        offset = self.offset_head(h).squeeze(1)
        return p, q, gain, offset

    @torch.no_grad()
    def decode_mixture(self, p: torch.Tensor, q: torch.Tensor, laser_idx: torch.Tensor,
                       gain: Optional[torch.Tensor] = None,
                       offset: Optional[torch.Tensor] = None,
                       G_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode from supplied p/q (and optional gain/offset)."""
        A = self.analyte_spectra()
        B = self.substrate_spectra()
        G = self.gain_matrix() if G_override is None else G_override
        A_mix = p @ A
        B_mix = q @ B
        if G_override is None:
            G_eff = torch.einsum("ba,as,bs->b", p, G, q)
        else:
            # G_override is (B, n_substrates): the analyte's mean gain per substrate.
            G_eff = (q * G_override).sum(dim=1)
        if gain is None:
            gain = torch.ones(p.shape[0], device=p.device)
        if offset is None:
            offset = torch.zeros(p.shape[0], device=p.device)
        r = self.laser_response(laser_idx)
        return gain.unsqueeze(1) * r * (G_eff.unsqueeze(1) * A_mix + B_mix) \
            + offset.unsqueeze(1)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_model(model: nn.Module,
                X: torch.Tensor,
                y_chem: torch.Tensor,
                y_sub: torch.Tensor,
                y_laser: torch.Tensor,
                train_idx: torch.Tensor,
                cfg: TrainConfig,
                device: torch.device,
                val_idx: Optional[torch.Tensor] = None) -> FitResult:
    """Train one model; early stopping on validation reconstruction MSE.

    Validation indices are drawn only from training maps by the caller.  If
    val_idx is None, a random ``val_fraction`` of ``train_idx`` is used.
    """
    _set_seed(cfg.seed)
    model = model.to(device)
    if val_idx is None:
        n_val = max(1, int(round(cfg.val_fraction * len(train_idx))))
        perm = np.random.RandomState(cfg.seed).permutation(len(train_idx))
        val_idx = train_idx[perm[:n_val]]
        train_idx = train_idx[perm[n_val:]]
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    bad = 0
    history = {"train_loss": [], "val_recon": []}
    failure = None

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        beta_eff = model.beta * min(1.0, epoch / max(1, cfg.warmup_epochs)) \
            if getattr(model, "kind", "") in ("vae", "cvae") else 0.0
        perm = torch.randperm(len(train_idx), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(train_idx), cfg.batch_size):
            idx = train_idx[perm[start:start + cfg.batch_size]]
            xb = X[idx]
            opt.zero_grad(set_to_none=True)
            if model.kind == "physics":
                out = model(xb, y_laser[idx], y_chem=y_chem[idx], y_sub=y_sub[idx])
                loss = F.mse_loss(out["recon"], xb) + out["ce"]
            elif model.kind == "cvae":
                out = model(xb, y_chem=y_chem[idx], y_sub=y_sub[idx],
                            y_laser=y_laser[idx] if model.use_laser else None)
                loss = F.mse_loss(out["recon"], xb) + beta_eff * out["kl"] + out["ce"]
            else:
                out = model(xb)
                loss = F.mse_loss(out["recon"], xb) + beta_eff * out["kl"]
            if not torch.isfinite(loss):
                failure = f"non-finite training loss at epoch {epoch}"
                break
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        if failure:
            break
        history["train_loss"].append(epoch_loss / max(1, n_batches))

        model.eval()
        with torch.no_grad():
            val_recon = 0.0
            val_n = 0
            for start in range(0, len(val_idx), cfg.batch_size):
                idx = val_idx[start:start + cfg.batch_size]
                xb = X[idx]
                if model.kind == "physics":
                    out = model(xb, y_laser[idx])
                else:
                    out = model(xb)
                val_recon += float(F.mse_loss(out["recon"], xb, reduction="sum"))
                val_n += xb.numel()
            val = val_recon / max(1, val_n)
        history["val_recon"].append(val)
        if not np.isfinite(val):
            failure = f"non-finite validation loss at epoch {epoch}"
            break
        if val < best_val - 1e-7:
            best_val = val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_epoch > 0:
        model.load_state_dict(best_state)
    model.eval()
    epochs_run = len(history["val_recon"])
    return FitResult(model=model, best_epoch=best_epoch, best_val=best_val,
                     epochs_run=epochs_run, history=history, failure=failure)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def vae_diagnostics(model: nn.Module, X: torch.Tensor, batch_size: int = 256) -> dict:
    """Active units, per-dimension KL and latent variances for VAE/CVAE."""
    model.eval()
    with torch.no_grad():
        if model.kind == "vae":
            mus, logvars = [], []
            for start in range(0, len(X), batch_size):
                mu, logvar = model.encode(X[start:start + batch_size])
                mus.append(mu)
                logvars.append(logvar)
            blocks = {"full": (torch.cat(mus, 0), torch.cat(logvars, 0))}
        elif model.kind == "cvae":
            per_mu = [[] for _ in model.dims]
            per_lv = [[] for _ in model.dims]
            for start in range(0, len(X), batch_size):
                mus, logvars = model.encode_blocks(X[start:start + batch_size])
                for j, (mu, lv) in enumerate(zip(mus, logvars)):
                    per_mu[j].append(mu)
                    per_lv[j].append(lv)
            blocks = {name: (torch.cat(per_mu[j], 0), torch.cat(per_lv[j], 0))
                      for j, name in enumerate(model.block_names)}
        else:
            return {}

    out = {}
    for name, (mu, logvar) in blocks.items():
        var = mu.var(dim=0, unbiased=False)
        kl_dim = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)
        out[name] = {
            "active_units": int((var > 0.01).sum().item()),
            "n_dims": int(mu.shape[1]),
            "latent_var": [float(v) for v in var.cpu()],
            "kl_per_dim": [float(v) for v in kl_dim.cpu()],
            "kl_total": float(kl_dim.sum().item()),
        }
    return out


@torch.no_grad()
def physics_diagnostics(model: PhysicsVAE, X: torch.Tensor,
                        y_laser: torch.Tensor) -> dict:
    p, q, gain, offset = model.probabilities(X, y_laser)
    A = model.analyte_spectra()
    B = model.substrate_spectra()
    G = model.gain_matrix()
    p_pred = p.argmax(dim=1)
    q_pred = q.argmax(dim=1)
    return {
        "mean_max_p": float(p.max(dim=1).values.mean().item()),
        "mean_max_q": float(q.max(dim=1).values.mean().item()),
        "n_p_classes": int(torch.unique(p_pred).numel()),
        "n_q_classes": int(torch.unique(q_pred).numel()),
        "p_class_counts": torch.bincount(p_pred, minlength=p.shape[1]).cpu().tolist(),
        "q_class_counts": torch.bincount(q_pred, minlength=q.shape[1]).cpu().tolist(),
        "mean_gain": float(gain.mean().item()),
        "offset_std": float(offset.std().item()),
        "A_min": float(A.min().item()),
        "B_min": float(B.min().item()),
        "G_min": float(G.min().item()),
        "G_max": float(G.max().item()),
        "G_range": float((G.max() - G.min()).item()),
    }
