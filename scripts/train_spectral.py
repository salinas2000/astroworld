#!/usr/bin/env python
"""
Train the SSTF (Siamese Spectral-Temporal Fusion) model.

Usage
-----
    uv run python scripts/train_spectral.py \\
        --data-dir data/training/spectral \\
        --epochs 100 --lr 5e-4 --batch-size 16 \\
        --checkpoint-dir checkpoints/spectral

The data directory must contain a ``training_catalog.csv`` with the
multi-survey schema (optical_path, w1_path, w2_path, temp_kelvin, etc.)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from astroworld.ml.spectral_model import SpectralSiameseNet
from astroworld.ml.spectral_dataset import MultiSurveyDataset
from astroworld.ml.trainer import (
    TrainingConfig,
    TrainingResult,
    EpochMetrics,
    compute_metrics,
    compute_roc_auc,
)


# ---------------------------------------------------------------------------
# SpectralTrainer — dual-head training loop
# ---------------------------------------------------------------------------

class SpectralTrainer:
    """Training loop for SpectralSiameseNet with dual-head loss.

    Combined loss:
        L = alpha * BCE(motion_logits, labels) + beta * MSE(temp_pred, temp_true) * mask

    Temperature loss is masked: only applied when source_present=True
    (negatives have NaN temperature and are excluded).

    Parameters
    ----------
    model : SpectralSiameseNet
    config : TrainingConfig with standard hyperparameters
    alpha : Weight for motion classification loss (default 1.0)
    beta : Weight for temperature regression loss (default 0.1)
    """

    def __init__(
        self,
        model: SpectralSiameseNet,
        config: TrainingConfig,
        alpha: float = 1.0,
        beta: float = 0.1,
    ):
        self.model = model
        self.config = config
        self.alpha = alpha
        self.beta = beta
        self.device = self._select_device()
        self.model.to(self.device)

    def train(
        self,
        train_dataset: MultiSurveyDataset,
        val_dataset: MultiSurveyDataset,
    ) -> TrainingResult:
        """Run the full training loop.

        Parameters
        ----------
        train_dataset, val_dataset : MultiSurveyDataset instances

        Returns
        -------
        TrainingResult with epoch-by-epoch metrics
        """
        cfg = self.config
        result = TrainingResult(config=cfg)
        t0 = time.time()

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            shuffle=True, num_workers=cfg.num_workers, drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.batch_size,
            shuffle=False, num_workers=cfg.num_workers, drop_last=False,
        )

        # Loss functions
        pos_weight = self._compute_pos_weight(train_dataset)
        motion_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=self.device)
        )
        temp_criterion = nn.MSELoss(reduction="none")

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # Scheduler
        scheduler = self._build_scheduler(optimizer, len(train_loader))

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, cfg.epochs + 1):
            # --- Train ---
            train_loss = self._train_epoch(
                train_loader, motion_criterion, temp_criterion,
                optimizer, scheduler,
            )

            # --- Validate ---
            metrics = self._validate(
                epoch, val_loader, motion_criterion, temp_criterion,
            )
            metrics.train_loss = train_loss

            current_lr = optimizer.param_groups[0]["lr"]
            metrics.learning_rate = current_lr

            if cfg.scheduler == "plateau" and scheduler is not None:
                scheduler.step(metrics.val_loss)

            result.epoch_metrics.append(metrics)

            # Checkpoint
            is_best = metrics.val_loss < best_val_loss - cfg.min_delta
            marker = ""
            if is_best:
                best_val_loss = metrics.val_loss
                patience_counter = 0
                result.best_epoch = epoch
                result.best_val_loss = metrics.val_loss
                result.best_val_f1 = metrics.val_f1
                if cfg.checkpoint_dir is not None:
                    result.model_state_path = self._save_checkpoint(
                        epoch, metrics.val_loss, optimizer,
                    )
                marker = " *best*"
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch:3d}/{cfg.epochs} "
                f"| train={train_loss:.4f} "
                f"| val={metrics.val_loss:.4f} "
                f"| acc={metrics.val_accuracy:.3f} "
                f"| F1={metrics.val_f1:.3f} "
                f"| AUC={metrics.val_roc_auc:.3f} "
                f"| lr={current_lr:.1e}{marker}"
            )

            if patience_counter >= cfg.patience:
                print(
                    f"  Early stopping (patience={cfg.patience}, "
                    f"no improvement for {patience_counter} epochs)"
                )
                break

        result.training_time_seconds = time.time() - t0
        return result

    def _train_epoch(
        self,
        loader: DataLoader,
        motion_criterion: nn.Module,
        temp_criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
    ) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            cube1, cube2, motion_labels, temp_labels = batch
            cube1 = cube1.to(self.device)
            cube2 = cube2.to(self.device)
            motion_labels = motion_labels.float().unsqueeze(-1).to(self.device)
            temp_labels = temp_labels.float().unsqueeze(-1).to(self.device)

            motion_logits, temp_pred = self.model(cube1, cube2)

            # Combined loss
            loss = self._compute_combined_loss(
                motion_logits, motion_labels, temp_pred, temp_labels,
                motion_criterion, temp_criterion,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.config.scheduler == "onecycle" and scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(
        self,
        epoch: int,
        loader: DataLoader,
        motion_criterion: nn.Module,
        temp_criterion: nn.Module,
    ) -> EpochMetrics:
        """Run validation and compute all metrics."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_labels = []
        all_logits = []

        for batch in loader:
            cube1, cube2, motion_labels, temp_labels = batch
            cube1 = cube1.to(self.device)
            cube2 = cube2.to(self.device)
            motion_labels_t = motion_labels.float().unsqueeze(-1).to(self.device)
            temp_labels_t = temp_labels.float().unsqueeze(-1).to(self.device)

            motion_logits, temp_pred = self.model(cube1, cube2)

            loss = self._compute_combined_loss(
                motion_logits, motion_labels_t, temp_pred, temp_labels_t,
                motion_criterion, temp_criterion,
            )
            total_loss += loss.item()
            n_batches += 1

            all_labels.append(motion_labels.numpy().flatten())
            all_logits.append(motion_logits.cpu().numpy().flatten())

        val_loss = total_loss / max(n_batches, 1)
        labels_np = np.concatenate(all_labels)
        logits_np = np.concatenate(all_logits)

        metrics_dict = compute_metrics(labels_np, logits_np)
        auc = compute_roc_auc(labels_np, logits_np)

        return EpochMetrics(
            epoch=epoch,
            train_loss=0.0,
            val_loss=val_loss,
            val_accuracy=metrics_dict["accuracy"],
            val_precision=metrics_dict["precision"],
            val_recall=metrics_dict["recall"],
            val_f1=metrics_dict["f1"],
            val_roc_auc=auc,
            learning_rate=0.0,
        )

    def _compute_combined_loss(
        self,
        motion_logits: torch.Tensor,
        motion_labels: torch.Tensor,
        temp_pred: torch.Tensor,
        temp_labels: torch.Tensor,
        motion_criterion: nn.Module,
        temp_criterion: nn.Module,
    ) -> torch.Tensor:
        """Compute alpha * BCE + beta * MSE(temp) * mask."""
        # Motion loss (always)
        motion_loss = motion_criterion(motion_logits, motion_labels)

        # Temperature loss (only for positive examples with valid temp)
        # NaN labels → mask out
        temp_mask = ~torch.isnan(temp_labels)
        if temp_mask.any():
            # Convert temp to log10(K) space
            temp_log10_true = torch.log10(
                torch.clamp(temp_labels[temp_mask], min=1.0)
            )
            temp_log10_pred = temp_pred[temp_mask]
            temp_loss = temp_criterion(temp_log10_pred, temp_log10_true).mean()
        else:
            temp_loss = torch.tensor(0.0, device=motion_logits.device)

        return self.alpha * motion_loss + self.beta * temp_loss

    def _compute_pos_weight(self, dataset: MultiSurveyDataset) -> float:
        """Compute class weight for imbalanced data."""
        n_pos = 0
        n_neg = 0
        for i in range(len(dataset)):
            _, _, label, _ = dataset[i]
            if label > 0.5:
                n_pos += 1
            else:
                n_neg += 1
        if n_pos == 0:
            return 1.0
        return max(0.1, n_neg / n_pos)

    def _build_scheduler(self, optimizer, steps_per_epoch):
        cfg = self.config
        if cfg.scheduler == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=cfg.learning_rate,
                steps_per_epoch=steps_per_epoch, epochs=cfg.epochs,
                pct_start=0.3,
            )
        elif cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5,
            )
        return None

    def _select_device(self) -> torch.device:
        if self.config.device != "auto":
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        optimizer: torch.optim.Optimizer,
    ) -> Path:
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        path = ckpt_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "model_type": "SpectralSiameseNet",
            },
            path,
        )
        return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train SSTF (Siamese Spectral-Temporal Fusion) model"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory with training_catalog.csv (multi-survey format)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Weight for motion loss")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Weight for temperature loss")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("checkpoints/spectral"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()

    catalog_path = args.data_dir / "training_catalog.csv"
    if not catalog_path.exists():
        print(f"ERROR: Catalog not found: {catalog_path}")
        sys.exit(1)

    print("=" * 60)
    print("  SSTF Training — Siamese Spectral-Temporal Fusion")
    print("=" * 60)
    print(f"  Data:       {args.data_dir}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  LR:         {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Loss:       {args.alpha}*BCE + {args.beta}*MSE(temp)")
    print(f"  Features:   {args.feature_dim}-dim, {args.num_heads} heads")
    print(f"  Dropout:    {args.dropout}")
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Device:     {args.device}")
    print("=" * 60)

    # Build model
    model = SpectralSiameseNet(
        feature_dim=args.feature_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    print(f"\n  Model parameters: {model.num_parameters:,}")

    # Build datasets
    full_dataset = MultiSurveyDataset(
        catalog_path, do_normalize=True, augment=False,
    )
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_val

    indices = np.arange(n_total)
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = MultiSurveyDataset(
        catalog_path, do_normalize=True, augment=True, indices=train_idx,
    )
    val_dataset = MultiSurveyDataset(
        catalog_path, do_normalize=True, augment=False, indices=val_idx,
    )

    print(f"  Train: {len(train_dataset)} pairs")
    print(f"  Val:   {len(val_dataset)} pairs")

    # Train
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer = SpectralTrainer(
        model, config, alpha=args.alpha, beta=args.beta,
    )
    result = trainer.train(train_dataset, val_dataset)

    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(result.summary())


if __name__ == "__main__":
    main()
