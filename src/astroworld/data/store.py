"""
Data storage utilities for ephemeris data.

Supports CSV for human readability and portability.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def save_ephemeris(
    data: pd.DataFrame,
    body_name: str,
    data_dir: Optional[Path] = None,
) -> Path:
    """Save ephemeris DataFrame to CSV. Returns the path to the saved file."""
    directory = data_dir or DATA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{body_name}.csv"
    data.to_csv(path, index=False)
    return path


def load_ephemeris(
    body_name: str,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load ephemeris DataFrame from CSV."""
    directory = data_dir or DATA_DIR
    path = directory / f"{body_name}.csv"
    return pd.read_csv(path)


def save_all(
    datasets: Dict[str, pd.DataFrame],
    data_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Save multiple bodies' ephemeris data."""
    paths = {}
    for name, df in datasets.items():
        paths[name] = save_ephemeris(df, name, data_dir)
    return paths


def load_all(
    body_names: list,
    data_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Load multiple bodies' ephemeris data."""
    return {name: load_ephemeris(name, data_dir) for name in body_names}
