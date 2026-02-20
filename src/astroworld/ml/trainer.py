"""
Training loop for Planet 9 CNN detectors.

Handles:
  - Automatic device selection (CUDA > MPS > CPU)
  - Class-balanced binary cross-entropy loss
  - Per-epoch metrics (accuracy, precision, recall, F1, ROC-AUC)
  - Checkpoint saving / loading (best model by validation loss)
  - Early stopping with configurable patience
  - Learning rate scheduling (OneCycleLR or ReduceLROnPlateau)

All metrics are computed with pure NumPy (no sklearn dependency).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    patience: int = 10
    min_delta: float = 1e-4
    scheduler: str = "onecycle"  # "onecycle" or "plateau"
    pos_weight: float | None = None  # auto-computed from dataset if None
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    checkpoint_dir: Path | None = None
    num_workers: int = 0


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1: float
    val_roc_auc: float
    learning_rate: float


@dataclass
class TrainingResult:
    """Complete training history and final state."""

    config: TrainingConfig
    epoch_metrics: list[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_f1: float = 0.0
    model_state_path: Path | None = None
    training_time_seconds: float = 0.0

    def summary(self) -> str:
        """Return a formatted training summary string."""
        lines = [
            f"  Best epoch:  {self.best_epoch}",
            f"  Val loss:    {self.best_val_loss:.4f}",
            f"  Val F1:      {self.best_val_f1:.4f}",
        ]
        if self.model_state_path:
            lines.append(f"  Checkpoint:  {self.model_state_path}")
        lines.append(f"  Time:        {self.training_time_seconds:.1f} seconds")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pure-numpy metrics (no sklearn)
# ---------------------------------------------------------------------------

def compute_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary classification metrics from numpy arrays.

    Parameters
    ----------
    labels : 1D array of true labels (0 or 1)
    logits : 1D array of raw logits (sigmoid applied internally)
    threshold : Classification threshold on probabilities

    Returns
    -------
    Dict with keys: accuracy, precision, recall, f1
    """
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    preds = (probs >= threshold).astype(int)

    labels = labels.astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy = float((preds == labels).mean())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC via trapezoidal integration (no sklearn).

    Parameters
    ----------
    labels : 1D array of true labels (0 or 1)
    scores : 1D array of prediction scores (higher = more positive)

    Returns
    -------
    Area under the ROC curve (0.0 to 1.0)
    """
    labels = labels.astype(int)

    # Handle degenerate cases
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5  # undefined, return chance level

    # Sort by descending score
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    # Compute TPR and FPR at each threshold
    tp_cumsum = np.cumsum(sorted_labels)
    fp_cumsum = np.cumsum(1 - sorted_labels)

    tpr = tp_cumsum / n_pos
    fpr = fp_cumsum / n_neg

    # Prepend (0, 0) origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    # Trapezoidal integration (np.trapezoid in NumPy 2.x, np.trapz in 1.x)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trapz(tpr, fpr))
    return max(0.0, min(1.0, auc))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Training loop with metrics, checkpointing, and early stopping.

    Parameters
    ----------
    model : P9DetectorCNN or P9SiameseDetector
    config : TrainingConfig with hyperparameters
    """

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = self._select_device()
        self.model.to(self.device)

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
    ) -> TrainingResult:
        """Run the full training loop.

        Parameters
        ----------
        train_dataset : Training data (FITSDataset or MultiEpochDataset)
        val_dataset : Validation data (same type)

        Returns
        -------
        TrainingResult with epoch-by-epoch metrics
        """
        cfg = self.config
        result = TrainingResult(config=cfg)
        t0 = time.time()

        # Reproducibility
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False,
        )

        # Loss with class weighting
        pos_weight = self._compute_pos_weight(train_dataset)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=self.device)
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # Scheduler
        scheduler = self._build_scheduler(optimizer, len(train_loader))

        # Early stopping state
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, cfg.epochs + 1):
            # --- Train ---
            train_loss = self._train_epoch(train_loader, criterion, optimizer, scheduler)

            # --- Validate ---
            metrics = self._validate(epoch, val_loader, criterion)
            metrics.train_loss = train_loss

            # Get current LR
            current_lr = optimizer.param_groups[0]["lr"]
            metrics.learning_rate = current_lr

            # ReduceLROnPlateau step (OneCycleLR steps per batch)
            if cfg.scheduler == "plateau" and scheduler is not None:
                scheduler.step(metrics.val_loss)

            result.epoch_metrics.append(metrics)

            # Checkpoint best model
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
                        epoch, metrics.val_loss, optimizer
                    )
                marker = " *best*"
            else:
                patience_counter += 1

            # Print progress
            print(
                f"Epoch {epoch:3d}/{cfg.epochs} "
                f"| train={train_loss:.4f} "
                f"| val={metrics.val_loss:.4f} "
                f"| acc={metrics.val_accuracy:.3f} "
                f"| F1={metrics.val_f1:.3f} "
                f"| AUC={metrics.val_roc_auc:.3f} "
                f"| lr={current_lr:.1e}{marker}"
            )

            # Early stopping
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
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
    ) -> float:
        """Run a single training epoch.

        Returns
        -------
        Mean training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            logits = self._forward_batch(batch)
            labels = self._extract_labels(batch).to(self.device)

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # OneCycleLR steps per batch
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
        criterion: nn.Module,
    ) -> EpochMetrics:
        """Run validation and compute all metrics.

        Returns
        -------
        EpochMetrics with loss, accuracy, precision, recall, F1, AUC
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_labels = []
        all_logits = []

        for batch in loader:
            logits = self._forward_batch(batch)
            labels = self._extract_labels(batch).to(self.device)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            all_labels.append(labels.cpu().numpy().flatten())
            all_logits.append(logits.cpu().numpy().flatten())

        val_loss = total_loss / max(n_batches, 1)
        labels_np = np.concatenate(all_labels)
        logits_np = np.concatenate(all_logits)

        # Compute metrics
        metrics_dict = compute_metrics(labels_np, logits_np)
        auc = compute_roc_auc(labels_np, logits_np)

        return EpochMetrics(
            epoch=epoch,
            train_loss=0.0,  # filled by caller
            val_loss=val_loss,
            val_accuracy=metrics_dict["accuracy"],
            val_precision=metrics_dict["precision"],
            val_recall=metrics_dict["recall"],
            val_f1=metrics_dict["f1"],
            val_roc_auc=auc,
            learning_rate=0.0,  # filled by caller
        )

    def _forward_batch(self, batch) -> torch.Tensor:
        """Forward pass for a batch (handles both single and Siamese)."""
        if len(batch) == 3:
            # Siamese: (image1, image2, label)
            x1 = batch[0].to(self.device)
            x2 = batch[1].to(self.device)
            return self.model(x1, x2)
        else:
            # Single: (image, label)
            x = batch[0].to(self.device)
            return self.model(x)

    def _extract_labels(self, batch) -> torch.Tensor:
        """Extract labels from batch (last element)."""
        labels = batch[-1]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)
        return labels.float().unsqueeze(-1)  # (B,) -> (B, 1)

    def _compute_pos_weight(self, dataset) -> float:
        """Compute positive class weight from dataset labels."""
        if self.config.pos_weight is not None:
            return self.config.pos_weight

        # Count positives and negatives
        n_pos = 0
        n_neg = 0
        for i in range(len(dataset)):
            sample = dataset[i]
            label = sample[-1]  # last element is label
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label > 0.5:
                n_pos += 1
            else:
                n_neg += 1

        if n_pos == 0:
            return 1.0
        return max(0.1, n_neg / n_pos)

    def _build_scheduler(self, optimizer, steps_per_epoch):
        """Build learning rate scheduler."""
        cfg = self.config
        if cfg.scheduler == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.learning_rate,
                steps_per_epoch=steps_per_epoch,
                epochs=cfg.epochs,
                pct_start=0.3,
            )
        elif cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
        return None

    def _select_device(self) -> torch.device:
        """Auto-select the best available compute device."""
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
        """Save model checkpoint.

        Returns
        -------
        Path to the saved checkpoint file
        """
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        path = ckpt_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: Path) -> int:
        """Load model from checkpoint.

        Parameters
        ----------
        path : Path to checkpoint file

        Returns
        -------
        Epoch number from the checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint["epoch"]
