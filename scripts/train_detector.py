#!/usr/bin/env python3
"""
Train the Planet 9 CNN detector on synthetic FITS training data.

Usage:
  # Train single-image detector (default)
  uv run python scripts/train_detector.py --data-dir data/training/dss2_red

  # Custom hyperparameters
  uv run python scripts/train_detector.py --data-dir data/training/dss2_red \
      --epochs 100 --lr 0.0005 --batch-size 32

  # Train Siamese multi-epoch detector
  uv run python scripts/train_detector.py --data-dir data/training/dss2_red \
      --model siamese

  # Resume from checkpoint
  uv run python scripts/train_detector.py --data-dir data/training/dss2_red \
      --resume checkpoints/best_model.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Train Planet 9 CNN detector on synthetic FITS data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Training data directory (with training_catalog.csv)",
    )

    # Model
    parser.add_argument(
        "--model", choices=["single", "siamese"], default="single",
        help="Model architecture (default: single)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5,
        help="Dropout probability (default: 0.5)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument(
        "--scheduler", choices=["onecycle", "plateau"], default="onecycle",
    )

    # Infrastructure
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints"),
    )
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)

    args = parser.parse_args()

    # ---- Check PyTorch ----
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed.")
        print("Install with: uv add --optional ml 'torch>=2.0'")
        sys.exit(1)

    from astroworld.ml.model import P9DetectorCNN, P9SiameseDetector
    from astroworld.ml.dataset import FITSDataset, MultiEpochDataset, split_train_val
    from astroworld.ml.trainer import Trainer, TrainingConfig

    # ---- Validate data directory ----
    catalog_path = args.data_dir / "training_catalog.csv"
    if not catalog_path.exists():
        print(f"ERROR: Catalog not found: {catalog_path}")
        print("Generate training data first with scripts/generate_training_data.py")
        sys.exit(1)

    catalog = pd.read_csv(catalog_path)

    # ---- Header ----
    print("=" * 60)
    print("  ASTROWORLD - PLANET 9 CNN DETECTOR TRAINING")
    print("=" * 60)
    print()

    cuda_str = "True" if torch.cuda.is_available() else "False"
    print(f"  PyTorch:  {torch.__version__} (CUDA: {cuda_str})")

    # ---- Build model ----
    if args.model == "single":
        model = P9DetectorCNN(dropout=args.dropout)
        model_name = "P9DetectorCNN"
    else:
        model = P9SiameseDetector(dropout=args.dropout)
        model_name = "P9SiameseDetector"

    print(f"  Model:    {model_name} ({model.num_parameters:,} parameters)")
    print()

    # ---- Prepare datasets ----
    n_pos = int(catalog["source_present"].sum())
    n_neg = len(catalog) - n_pos

    if args.model == "siamese":
        # For Siamese, split at the PAIR level (not row level)
        # First build the dataset to count pairs, then split pair indices
        full_ds = MultiEpochDataset(catalog_path, augment=False)
        n_pairs = len(full_ds)
        rng = np.random.default_rng(args.seed)
        pair_indices = np.arange(n_pairs)
        rng.shuffle(pair_indices)
        n_val = max(1, int(n_pairs * args.val_fraction))
        val_idx_arr = pair_indices[:n_val]
        train_idx_arr = pair_indices[n_val:]

        train_ds = MultiEpochDataset(
            catalog_path, augment=True, indices=train_idx_arr,
        )
        val_ds = MultiEpochDataset(
            catalog_path, augment=False, indices=val_idx_arr,
        )
        print(f"  Data:     {catalog_path}")
        print(f"  Pairs:    {n_pairs} total -> {len(train_ds)} train, {len(val_ds)} val")
    else:
        train_idx_arr, val_idx_arr = split_train_val(
            catalog, val_fraction=args.val_fraction, seed=args.seed
        )
        train_ds = FITSDataset(
            catalog_path, augment=True, indices=train_idx_arr,
        )
        val_ds = FITSDataset(
            catalog_path, augment=False, indices=val_idx_arr,
        )
        train_pos = sum(1 for i in range(len(train_ds)) if train_ds[i][1] > 0.5)
        train_neg = len(train_ds) - train_pos
        val_pos = sum(1 for i in range(len(val_ds)) if val_ds[i][1] > 0.5)
        val_neg = len(val_ds) - val_pos

        print(f"  Data:     {catalog_path}")
        print(f"  Total:    {len(catalog)} examples ({n_pos} pos, {n_neg} neg)")
        print(f"  Train:    {len(train_ds)} ({train_pos} pos, {train_neg} neg)")
        print(f"  Val:      {len(val_ds)} ({val_pos} pos, {val_neg} neg)")

    print()
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch_size}")
    print(f"  LR:       {args.lr:.1e} ({args.scheduler})")
    print(f"  Patience: {args.patience}")
    print("  " + "-" * 56)
    print()

    # ---- Build config and trainer ----
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        patience=args.patience,
        scheduler=args.scheduler,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
    )

    trainer = Trainer(model, config)

    # Resume from checkpoint
    if args.resume is not None:
        epoch = trainer.load_checkpoint(args.resume)
        print(f"  Resumed from checkpoint: {args.resume} (epoch {epoch})")
        print()

    print(f"  Device:   {trainer.device}")
    print()

    # ---- Train ----
    result = trainer.train(train_ds, val_ds)

    # ---- Summary ----
    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(result.summary())
    print()

    # ---- Physics reality check ----
    if result.best_val_f1 < 0.55 and args.model == "single":
        print("  NOTE: Low F1 score is expected for single-image detection")
        print("  of very faint sources (mag > 22, S/N << 1).")
        print("  Try the Siamese detector: --model siamese")
        print()


if __name__ == "__main__":
    main()
