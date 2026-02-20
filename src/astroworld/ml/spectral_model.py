"""
SSTF: Siamese Spectral-Temporal Fusion Network for Planet 9 detection.

Processes 5-channel spectral cubes (optical, W1, W2, uncertainty,
temporal gradient) through a shared CNN encoder with cross-attention
fusion between epochs and dual output heads:
  - Motion classifier: binary detection of moving point source
  - Temperature regressor: blackbody temperature from W1/W2 ratio

MC Dropout enables Bayesian uncertainty estimation at inference time.

Architecture
------------
Epoch1 cube (5, 64, 64) ──┐
                           ├─→ Shared SpectralEncoder → f1 (B, 256)
Epoch2 cube (5, 64, 64) ──┘                             f2 (B, 256)
                                                         ↓
                        CrossAttentionFusion(f1, f2) → (B, 1024)
                                                         ↓
                        ┌─ motion_head → (B, 1) logit   [BCE loss]
                        └─ temp_head   → (B, 1) log10(K) [MSE loss]

Design choices:
  - Custom lightweight CNN (~1.2M params total) — no torchvision dependency
  - 5×5 first kernel captures cross-band spectral correlations
  - Pooling adapted for 64×64 patches (4, 2, 2) vs original 512×512 (4, 4, 2)
  - Cross-attention replaces simple |f1-f2| difference for richer fusion
  - Temperature predicted in log10(K) space for numerical stability (range 0-5)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Convolutional building block (same pattern as model.py)
# ---------------------------------------------------------------------------

def _conv_block(
    in_ch: int,
    out_ch: int,
    kernel: int = 3,
    padding: int = 1,
    pool_size: int = 2,
) -> nn.Sequential:
    """Conv2d -> BatchNorm -> ReLU -> MaxPool."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool_size),
    )


# ---------------------------------------------------------------------------
# SpectralEncoder — 5-channel CNN backbone
# ---------------------------------------------------------------------------

class SpectralEncoder(nn.Module):
    """CNN backbone for 5-channel spectral cubes.

    Architecture (for 64×64 input)
    ---------
    Block 1: Conv(5->32, 5×5) + BN + ReLU + MaxPool(4)    [64→16]
    Block 2: Conv(32->64, 3×3) + BN + ReLU + MaxPool(2)   [16→8]
    Block 3: Conv(64->128, 3×3) + BN + ReLU + MaxPool(2)  [8→4]
    Block 4: Conv(128->256, 3×3) + BN + ReLU + AdaptiveAvgPool(1) → (B, 256)

    Wider than original P9DetectorCNN (32/64/128/256 vs 16/32/64/128)
    to handle the additional spectral information.

    The 5×5 first kernel is critical: it captures cross-band correlations
    (optical-W1-W2) at the PSF scale (~2.5 px FWHM).

    Parameters
    ----------
    in_channels : Number of input channels (default 5)
    feature_dim : Output feature vector dimension (default 256)
    """

    def __init__(self, in_channels: int = 5, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        self.features = nn.Sequential(
            # Block 1: 5×5 kernel captures cross-band PSF-scale features
            # 64→16 (or smaller patches scale proportionally)
            _conv_block(in_channels, 32, kernel=5, padding=2, pool_size=4),
            # Block 2: 16→8
            _conv_block(32, 64, kernel=3, padding=1, pool_size=2),
            # Block 3: 8→4
            _conv_block(64, 128, kernel=3, padding=1, pool_size=2),
            # Block 4: global average pooling → (B, 256, 1, 1)
            nn.Conv2d(128, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, 5, H, W) spectral cube tensor

        Returns
        -------
        (B, feature_dim) feature vector
        """
        return self.flatten(self.features(x))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# CrossAttentionFusion — bidirectional cross-attention between epochs
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention fusion between two epoch features.

    Instead of simple concatenation + absolute difference (original Siamese),
    this module uses cross-attention to learn which spectral features from
    one epoch are most relevant for detecting changes in the other.

    For each direction:
      Q from epoch A, K/V from epoch B  →  "what in B explains A?"

    Output: cat(f1, f2, attn_12, attn_21) → (B, 4 * embed_dim)

    Parameters
    ----------
    embed_dim : Feature dimension from encoder (default 256)
    num_heads : Number of attention heads (default 4)
    dropout : Dropout probability in attention (default 0.1)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_attn_1to2 = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=False,
        )
        self.cross_attn_2to1 = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=False,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse two epoch feature vectors via cross-attention.

        Parameters
        ----------
        f1, f2 : (B, embed_dim) feature vectors from shared encoder

        Returns
        -------
        (B, 4 * embed_dim) fused representation
        """
        # MultiheadAttention expects (seq_len, batch, embed_dim)
        # Our "sequence" is length 1 (global feature vector)
        f1_seq = f1.unsqueeze(0)  # (1, B, D)
        f2_seq = f2.unsqueeze(0)  # (1, B, D)

        # Cross-attention: what in epoch2 is relevant for epoch1?
        attn_12, _ = self.cross_attn_1to2(
            query=f1_seq, key=f2_seq, value=f2_seq,
        )  # (1, B, D)
        attn_12 = self.norm1(attn_12.squeeze(0))  # (B, D)

        # Cross-attention: what in epoch1 is relevant for epoch2?
        attn_21, _ = self.cross_attn_2to1(
            query=f2_seq, key=f1_seq, value=f1_seq,
        )  # (1, B, D)
        attn_21 = self.norm2(attn_21.squeeze(0))  # (B, D)

        # Concatenate all representations
        combined = torch.cat([f1, f2, attn_12, attn_21], dim=1)  # (B, 4*D)
        return combined

    @property
    def output_dim(self) -> int:
        """Output feature dimension (4 * embed_dim)."""
        return 4 * self.cross_attn_1to2.embed_dim


# ---------------------------------------------------------------------------
# SpectralSiameseNet — main SSTF model
# ---------------------------------------------------------------------------

class SpectralSiameseNet(nn.Module):
    """Siamese Spectral-Temporal Fusion Network.

    Dual-head architecture with MC Dropout for Bayesian uncertainty:
      - Motion head: binary classifier (moving source detected?)
      - Temperature head: regressor (blackbody temperature in log10 K)

    Parameters
    ----------
    encoder : Shared encoder module (default: SpectralEncoder)
    feature_dim : Encoder output dimension (default 256)
    num_heads : Cross-attention heads (default 4)
    dropout : MC Dropout rate for inference uncertainty (default 0.2)
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        feature_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = SpectralEncoder(in_channels=5, feature_dim=feature_dim)

        self.flatten = nn.Flatten()
        self.cross_attn = CrossAttentionFusion(
            feature_dim, num_heads, dropout=dropout,
        )

        combined_dim = 4 * feature_dim  # 1024

        # Motion classification head (binary)
        self.motion_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # Temperature regression head (log10 Kelvin)
        self.temp_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self._dropout_rate = dropout

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with two epoch spectral cubes.

        Parameters
        ----------
        x1, x2 : (B, 5, H, W) spectral cube tensors

        Returns
        -------
        motion_logits : (B, 1) raw logits for binary motion detection
        temp_log10k : (B, 1) predicted temperature in log10(Kelvin)
        """
        f1 = self.encoder(x1)  # (B, feature_dim)
        f2 = self.encoder(x2)  # (B, feature_dim)

        # Cross-attention fusion
        fused = self.cross_attn(f1, f2)  # (B, 4 * feature_dim)

        # Dual heads
        motion_logits = self.motion_head(fused)  # (B, 1)
        temp_log10k = self.temp_head(fused)       # (B, 1)

        return motion_logits, temp_log10k

    def predict_proba(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """Return motion detection probabilities.

        Compatible with P9Searcher interface (single output).

        Parameters
        ----------
        x1, x2 : (B, 5, H, W) spectral cubes

        Returns
        -------
        (B, 1) probabilities in [0, 1]
        """
        motion_logits, _ = self.forward(x1, x2)
        return torch.sigmoid(motion_logits)

    def predict_with_uncertainty(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        n_samples: int = 10,
    ) -> dict[str, torch.Tensor]:
        """MC Dropout inference for Bayesian uncertainty estimation.

        Runs ``n_samples`` forward passes with dropout active, then
        computes mean and standard deviation of predictions.

        Parameters
        ----------
        x1, x2 : (B, 5, H, W) spectral cubes
        n_samples : Number of MC samples (default 10)

        Returns
        -------
        Dict with keys:
            'motion_prob_mean' : (B,) mean detection probability
            'motion_prob_std'  : (B,) epistemic uncertainty
            'temp_mean'        : (B,) mean temperature (Kelvin)
            'temp_std'         : (B,) temperature uncertainty (Kelvin)
        """
        was_training = self.training
        self.train()  # Enable dropout

        motion_probs = []
        temp_values = []

        with torch.no_grad():
            for _ in range(n_samples):
                motion_logits, temp_log10k = self.forward(x1, x2)
                probs = torch.sigmoid(motion_logits).squeeze(-1)  # (B,)
                temps = 10.0 ** temp_log10k.squeeze(-1)            # (B,) Kelvin
                motion_probs.append(probs)
                temp_values.append(temps)

        # Restore original mode
        if not was_training:
            self.eval()

        # Stack and compute statistics
        motion_stack = torch.stack(motion_probs, dim=0)  # (n_samples, B)
        temp_stack = torch.stack(temp_values, dim=0)       # (n_samples, B)

        return {
            "motion_prob_mean": motion_stack.mean(dim=0),
            "motion_prob_std": motion_stack.std(dim=0),
            "temp_mean": temp_stack.mean(dim=0),
            "temp_std": temp_stack.std(dim=0),
        }

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_encoder(self) -> nn.Module:
        """Return the shared encoder for transfer learning."""
        return self.encoder


# ---------------------------------------------------------------------------
# PlanckFilter — physical validation
# ---------------------------------------------------------------------------

class PlanckFilter:
    """Physical classification filter based on blackbody temperature.

    Uses the temperature estimated from WISE W1/W2 flux ratio to classify
    detection candidates into physically meaningful categories.

    Classification thresholds:
      - T < 50 K  →  'p9_candidate' (P9 at ~480 AU has T ~ 2-5 K)
      - 50 < T < 500 K  →  'cold_object' (could be cold brown dwarf, TNO)
      - T > 500 K  →  'warm_reject' (asteroid, star, artifact)
      - High uncertainty  →  'uncertain'
    """

    P9_TEMP_MAX = 50.0        # K — upper limit for P9 candidate
    BROWN_DWARF_MIN = 500.0   # K — typical minimum for brown dwarfs

    @staticmethod
    def classify(temp_k: float, temp_std_k: float = 0.0) -> str:
        """Classify a candidate based on estimated temperature.

        Parameters
        ----------
        temp_k : float
            Estimated blackbody temperature in Kelvin.
        temp_std_k : float
            Temperature uncertainty (1-sigma) in Kelvin.

        Returns
        -------
        Classification string:
            'p9_candidate', 'cold_object', 'warm_reject', or 'uncertain'
        """
        # High relative uncertainty → cannot classify
        if temp_k > 0 and temp_std_k > 0.5 * temp_k:
            return "uncertain"

        # Apply 2-sigma margin for cold classification
        effective_temp = temp_k + 2.0 * temp_std_k

        if effective_temp < PlanckFilter.P9_TEMP_MAX:
            return "p9_candidate"
        elif temp_k < PlanckFilter.BROWN_DWARF_MIN:
            return "cold_object"
        else:
            return "warm_reject"

    @staticmethod
    def expected_p9_temperature(distance_au: float) -> float:
        """Expected equilibrium temperature for a body at given distance.

        Uses the solar radiative equilibrium formula:
            T = T_sun * sqrt(R_sun / (2 * d))
        which simplifies to approximately:
            T ≈ 278 * d^(-0.5) K  (for low-albedo body)

        Parameters
        ----------
        distance_au : float
            Heliocentric distance in AU.

        Returns
        -------
        Equilibrium temperature in Kelvin.
        """
        if distance_au <= 0:
            return 0.0
        # T ~ 278 K * d^(-1/2) for a perfectly absorbing sphere
        return 278.0 / (distance_au ** 0.5)
