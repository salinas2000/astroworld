"""
Download ephemeris data from JPL Horizons.

Usage:
    uv run python scripts/download_ephemeris.py
    uv run python scripts/download_ephemeris.py --bodies earth mars --start 2000-01-01 --stop 2020-01-01
    uv run python scripts/download_ephemeris.py --split-earth-moon
"""

import argparse

from astroworld.constants import BODIES, DEFAULT_BODIES, DEFAULT_BODIES_SPLIT
from astroworld.data.horizons import fetch_all_planets
from astroworld.data.store import save_all


def main():
    parser = argparse.ArgumentParser(description="Download ephemeris data from JPL Horizons")
    valid_bodies = [n for n in BODIES if n != "sun"]
    parser.add_argument(
        "--bodies",
        nargs="+",
        default=None,
        choices=valid_bodies,
        help="Bodies to download (default: all planets as EMB)",
    )
    parser.add_argument(
        "--split-earth-moon",
        action="store_true",
        help="Download Earth (399) and Moon (301) separately instead of EMB (3)",
    )
    parser.add_argument("--start", default="2000-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--stop", default="2020-01-01", help="Stop date (YYYY-MM-DD)")
    parser.add_argument("--step", default="1d", help="Time step (default: 1d)")
    args = parser.parse_args()

    if args.bodies is not None:
        bodies = args.bodies
    elif args.split_earth_moon:
        bodies = [b for b in DEFAULT_BODIES_SPLIT if b != "sun"]
    else:
        bodies = [b for b in DEFAULT_BODIES if b != "sun"]

    print(f"Downloading ephemeris for: {', '.join(bodies)}")
    print(f"Period: {args.start} to {args.stop}, step: {args.step}")

    datasets = fetch_all_planets(
        epoch_start=args.start,
        epoch_stop=args.stop,
        step=args.step,
        bodies=bodies,
    )

    paths = save_all(datasets)
    print("\nSaved files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
