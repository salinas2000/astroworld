"""
Post-processing for P9 candidate detections.

Functions:
  - non_maximum_suppression: Grid-based NMS to merge overlapping detections
  - detections_to_candidates: NMS + pixel-to-RA/Dec coordinate conversion
  - generate_report: CSV summary of all candidates
  - plot_heatmap: Probability overlay on FITS image
  - save_diagnostic_cutouts: Per-candidate epoch1/epoch2/diff PNG cutouts
"""

from pathlib import Path

import numpy as np
import pandas as pd

from astroworld.ml.inference import Candidate, PatchDetection, SearchResult


# ---------------------------------------------------------------------------
# Non-Maximum Suppression
# ---------------------------------------------------------------------------

def non_maximum_suppression(
    detections: list[PatchDetection],
    grid_shape: tuple[int, int],
    suppress_radius: int = 1,
) -> list[PatchDetection]:
    """Suppress overlapping detections, keeping highest probability.

    Operates on the patch grid: for each surviving detection, all
    neighboring patches within ``suppress_radius`` grid cells are
    suppressed.

    Parameters
    ----------
    detections : Patch detections to filter
    grid_shape : (n_rows, n_cols) of the full patch grid
    suppress_radius : Suppression radius in grid cells (default 1)

    Returns
    -------
    Surviving detections after suppression (sorted by probability)
    """
    if not detections:
        return []

    # Sort by descending probability
    sorted_dets = sorted(
        detections, key=lambda d: d.probability, reverse=True
    )

    suppressed: set[tuple[int, int]] = set()
    survivors: list[PatchDetection] = []

    for det in sorted_dets:
        key = (det.patch_row, det.patch_col)
        if key in suppressed:
            continue

        survivors.append(det)

        # Suppress neighbors
        for dr in range(-suppress_radius, suppress_radius + 1):
            for dc in range(-suppress_radius, suppress_radius + 1):
                nr = det.patch_row + dr
                nc = det.patch_col + dc
                if 0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1]:
                    suppressed.add((nr, nc))

    return survivors


# ---------------------------------------------------------------------------
# Coordinate conversion + candidate grouping
# ---------------------------------------------------------------------------

def detections_to_candidates(
    detections: list[PatchDetection],
    header: dict | None,
    grid_shape: tuple[int, int],
    suppress_radius: int = 1,
) -> list[Candidate]:
    """Apply NMS and convert surviving detections to sky candidates.

    Parameters
    ----------
    detections : Raw patch detections (above threshold)
    header : FITS header dict for WCS (None → pixel coords only)
    grid_shape : (n_rows, n_cols) of the patch grid
    suppress_radius : NMS radius in grid cells

    Returns
    -------
    List of Candidate objects with RA/Dec coordinates
    """
    survivors = non_maximum_suppression(detections, grid_shape, suppress_radius)

    candidates = []
    for det in survivors:
        # Convert pixel to RA/Dec if WCS available
        ra_deg = float("nan")
        dec_deg = float("nan")

        if header is not None:
            try:
                from astroworld.imaging.synthetic_injection import pixel_to_radec
                ra_deg, dec_deg = pixel_to_radec(
                    det.pixel_x, det.pixel_y, header
                )
            except Exception:
                pass  # Keep NaN if WCS conversion fails

        # Count nearby patches that also triggered
        merged = [
            (d.patch_row, d.patch_col)
            for d in detections
            if abs(d.patch_row - det.patch_row) <= suppress_radius
            and abs(d.patch_col - det.patch_col) <= suppress_radius
        ]

        candidates.append(Candidate(
            pixel_x=det.pixel_x,
            pixel_y=det.pixel_y,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            probability=det.probability,
            n_patches=len(merged),
            patch_indices=merged,
        ))

    return candidates


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    results: list[SearchResult],
    output_path: Path | str,
) -> pd.DataFrame:
    """Generate CSV detection report from multiple search results.

    Parameters
    ----------
    results : List of SearchResult objects (with candidates filled)
    output_path : Path to write the CSV

    Returns
    -------
    DataFrame with all candidates
    """
    records = []
    for result in results:
        for cand in result.candidates:
            records.append({
                "image_1": result.image_path_1,
                "image_2": result.image_path_2,
                "pixel_x": cand.pixel_x,
                "pixel_y": cand.pixel_y,
                "ra_deg": cand.ra_deg,
                "dec_deg": cand.dec_deg,
                "probability": cand.probability,
                "n_patches": cand.n_patches,
            })

    df = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_heatmap(
    result: SearchResult,
    image: np.ndarray,
    output_path: Path | str,
    title: str = "",
) -> None:
    """Overlay probability heatmap on FITS image.

    Creates a diagnostic plot with:
    - Grayscale image background (sigma-clipped for contrast)
    - Semi-transparent probability heatmap overlay
    - Red circles on candidates above threshold

    Parameters
    ----------
    result : SearchResult with heatmap and candidates
    image : 2D numpy array of the full FITS image
    output_path : Path to save PNG
    title : Optional plot title
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Display image with sigma-clipped contrast
    vmin, vmax = _sigma_clip_limits(image)
    ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    # Overlay heatmap
    n_rows, n_cols = result.grid_shape
    stride = result.stride
    ps = result.patch_size

    # Build full-resolution heatmap
    h, w = image.shape
    heat_full = np.zeros((h, w), dtype=np.float32)
    heat_count = np.zeros((h, w), dtype=np.float32)

    _, _, y_origins, x_origins = _recompute_origins(
        h, w, ps, stride
    )

    for row in range(n_rows):
        for col in range(n_cols):
            y0 = y_origins[row]
            x0 = x_origins[col]
            prob = result.heatmap[row, col]
            heat_full[y0:y0 + ps, x0:x0 + ps] += prob
            heat_count[y0:y0 + ps, x0:x0 + ps] += 1.0

    # Average overlapping regions
    mask = heat_count > 0
    heat_full[mask] /= heat_count[mask]

    ax.imshow(
        heat_full, origin="lower", cmap="hot", alpha=0.4,
        vmin=0.0, vmax=1.0,
    )

    # Mark candidates
    for cand in result.candidates:
        circle = plt.Circle(
            (cand.pixel_x, cand.pixel_y), radius=ps / 2,
            fill=False, color="lime", linewidth=2,
        )
        ax.add_patch(circle)
        ax.text(
            cand.pixel_x + ps / 2 + 5, cand.pixel_y,
            f"p={cand.probability:.2f}",
            color="lime", fontsize=9, fontweight="bold",
        )

    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title(
            f"P9 Search Heatmap | {len(result.candidates)} candidates "
            f"(threshold={result.threshold:.2f})",
            fontsize=12,
        )

    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_diagnostic_cutouts(
    result: SearchResult,
    image_1: np.ndarray,
    image_2: np.ndarray,
    output_dir: Path | str,
    pair_label: str = "pair",
) -> list[Path]:
    """Save 64×64 cutouts for each candidate above threshold.

    For each candidate, saves:
    - Epoch 1 cutout
    - Epoch 2 cutout
    - Difference image (epoch2 - epoch1)

    Parameters
    ----------
    result : SearchResult with candidates
    image_1, image_2 : Full FITS images (2D numpy arrays)
    output_dir : Directory for cutout PNGs
    pair_label : Prefix for filenames

    Returns
    -------
    List of saved file paths
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    ps = result.patch_size
    h, w = image_1.shape

    for i, cand in enumerate(result.candidates):
        # Extract cutout centered on candidate
        cx = int(round(cand.pixel_x))
        cy = int(round(cand.pixel_y))

        y0 = max(0, cy - ps // 2)
        x0 = max(0, cx - ps // 2)
        y1 = min(h, y0 + ps)
        x1 = min(w, x0 + ps)

        cut1 = image_1[y0:y1, x0:x1].astype(np.float32)
        cut2 = image_2[y0:y1, x0:x1].astype(np.float32)
        diff = cut2 - cut1

        # Plot 3-panel: epoch1, epoch2, difference
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        vmin1, vmax1 = _sigma_clip_limits(cut1)
        axes[0].imshow(cut1, origin="lower", cmap="gray", vmin=vmin1, vmax=vmax1)
        axes[0].set_title(f"Epoch 1")

        vmin2, vmax2 = _sigma_clip_limits(cut2)
        axes[1].imshow(cut2, origin="lower", cmap="gray", vmin=vmin2, vmax=vmax2)
        axes[1].set_title(f"Epoch 2")

        vd = max(abs(diff.min()), abs(diff.max()), 1.0)
        axes[2].imshow(diff, origin="lower", cmap="RdBu_r", vmin=-vd, vmax=vd)
        axes[2].set_title(f"Difference (E2 - E1)")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(
            f"Candidate {i + 1} | p={cand.probability:.3f} | "
            f"px=({cand.pixel_x:.0f}, {cand.pixel_y:.0f})",
            fontsize=11,
        )

        path = output_dir / f"{pair_label}_cand_{i + 1:03d}.png"
        fig.savefig(str(path), dpi=100, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigma_clip_limits(
    image: np.ndarray,
    n_sigma: float = 3.0,
) -> tuple[float, float]:
    """Compute display limits using sigma-clipping.

    Returns (vmin, vmax) for matplotlib imshow.
    """
    flat = image.flatten()
    mask = np.ones(len(flat), dtype=bool)

    for _ in range(3):
        masked = flat[mask]
        if len(masked) == 0:
            break
        mean = masked.mean()
        std = masked.std()
        if std < 1e-10:
            break
        mask = np.abs(flat - mean) < n_sigma * std

    masked = flat[mask]
    if len(masked) == 0:
        return float(image.min()), float(image.max())

    mean = masked.mean()
    std = max(masked.std(), 1e-6)
    return float(mean - n_sigma * std), float(mean + n_sigma * std)


def _recompute_origins(
    img_h: int,
    img_w: int,
    patch_size: int,
    stride: int,
) -> tuple[int, int, list[int], list[int]]:
    """Recompute grid origins (matches P9Searcher._compute_grid_shape)."""

    def _origins(img_size: int) -> list[int]:
        if img_size < patch_size:
            return [0]
        origins = list(range(0, img_size - patch_size + 1, stride))
        if origins[-1] + patch_size < img_size:
            origins.append(img_size - patch_size)
        return origins

    y_origins = _origins(img_h)
    x_origins = _origins(img_w)
    return len(y_origins), len(x_origins), y_origins, x_origins
