"""
CNN architectures for Planet 9 detection in FITS sky survey images.

Two models:
  - P9DetectorCNN: single-image binary classifier (is there a faint
    point source in this 512x512 image?)
  - P9SiameseDetector: multi-epoch Siamese network (is there a source
    that moved consistently between two epoch images?)

The single-image detector is a baseline.  At S/N < 1 (mag > 22),
detection from a single frame is physically impossible; the Siamese
network exploits correlated motion (~8 px/month) between epoch pairs
to beat the background noise floor.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Convolutional building block
# ---------------------------------------------------------------------------

def _conv_block(
    in_ch: int,
    out_ch: int,
    kernel: int = 3,
    padding: int = 1,
    pool_size: int = 4,
) -> nn.Sequential:
    """Conv2d -> BatchNorm -> ReLU -> MaxPool."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool_size),
    )


# ---------------------------------------------------------------------------
# P9DetectorCNN — Single-Image Baseline
# ---------------------------------------------------------------------------

class P9DetectorCNN(nn.Module):
    """Binary classifier for faint point source detection.

    Architecture
    ------------
    Four convolutional blocks with aggressive MaxPool(4) downsample
    the 512x512 input to a 128-dim feature vector, followed by
    a dropout + linear head producing a single logit.

    Block 1: Conv(1->16, 5x5) + BN + ReLU + MaxPool(4)  => 128x128
    Block 2: Conv(16->32, 3x3) + BN + ReLU + MaxPool(4)  => 32x32
    Block 3: Conv(32->64, 3x3) + BN + ReLU + MaxPool(2)  => 16x16
    Block 4: Conv(64->128, 3x3) + BN + ReLU + AdaptiveAvgPool(1) => 128

    Design rationale:
    - First 5x5 kernel matches typical PSF size (~2.5 px FWHM)
    - Aggressive pooling: source is ~25 pixels out of 262,144
    - AdaptiveAvgPool avoids large FC layers, reduces overfitting
    - ~160K parameters: appropriate for datasets of 100-10,000 images

    Parameters
    ----------
    in_channels : Number of input channels (1 for single-band FITS)
    dropout : Dropout probability before the final linear layer
    """

    def __init__(self, in_channels: int = 1, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 5x5 kernel to match PSF scale
            # 512->128 (or 32->8 for tests)
            _conv_block(in_channels, 16, kernel=5, padding=2, pool_size=4),
            # Block 2: 128->32 (or 8->2)
            _conv_block(16, 32, kernel=3, padding=1, pool_size=4),
            # Block 3: 32->16 (or 2->1), pool_size=2 for small-image compat
            _conv_block(32, 64, kernel=3, padding=1, pool_size=2),
            # Block 4 with adaptive global pooling -> (B, 128, 1, 1)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, 1, H, W) float32 input images

        Returns
        -------
        (B, 1) raw logits (pass through sigmoid for probabilities)
        """
        features = self.features(x)
        return self.head(features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return detection probabilities (sigmoid of logits).

        Parameters
        ----------
        x : (B, 1, H, W) input images

        Returns
        -------
        (B, 1) probabilities in [0, 1]
        """
        return torch.sigmoid(self.forward(x))

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_encoder(self) -> nn.Module:
        """Return the convolutional backbone (without classification head).

        Useful for initializing the Siamese network encoder from a
        pre-trained single-image detector.

        Returns
        -------
        nn.Sequential containing the feature extraction layers
        """
        return self.features


# ---------------------------------------------------------------------------
# P9SiameseDetector — Multi-Epoch Network
# ---------------------------------------------------------------------------

class P9SiameseDetector(nn.Module):
    """Multi-epoch Siamese network for detecting faint moving sources.

    Applies a shared CNN encoder to two epoch images, then combines
    the feature vectors via concatenation + absolute difference for
    binary classification.

    Architecture
    ------------
    Epoch1 -> encoder -> f1 (B, feature_dim)
    Epoch2 -> encoder -> f2 (B, feature_dim)
    Combined: cat(f1, f2, |f1-f2|) -> (B, 3*feature_dim)
    Head: Linear -> ReLU -> Dropout -> Linear -> (B, 1) logit

    The ``|f1 - f2|`` difference vector explicitly encodes inter-epoch
    change.  If P9 moves ~8 pixels between epochs, the difference
    captures a consistent spatial shift signature that random noise
    fluctuations do not produce.

    Parameters
    ----------
    encoder : Pre-built encoder module (default: P9DetectorCNN backbone)
    feature_dim : Dimension of each encoder output (default 128)
    dropout : Dropout probability in the combination head
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        feature_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            # Build default encoder from P9DetectorCNN backbone
            self.encoder = P9DetectorCNN(in_channels=1).get_encoder()

        self.flatten = nn.Flatten()

        # Combination head: f1 + f2 + |f1-f2| = 3 * feature_dim
        combined_dim = 3 * feature_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with two epoch images.

        Parameters
        ----------
        x1 : (B, 1, H, W) epoch 1 images
        x2 : (B, 1, H, W) epoch 2 images

        Returns
        -------
        (B, 1) raw logits
        """
        f1 = self.flatten(self.encoder(x1))  # (B, feature_dim)
        f2 = self.flatten(self.encoder(x2))  # (B, feature_dim)

        # Concatenate features + absolute difference
        diff = torch.abs(f1 - f2)
        combined = torch.cat([f1, f2, diff], dim=1)  # (B, 3*feature_dim)

        return self.head(combined)

    def predict_proba(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """Return detection probabilities."""
        return torch.sigmoid(self.forward(x1, x2))

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_encoder(self) -> nn.Module:
        """Return the shared encoder for transfer learning."""
        return self.encoder
