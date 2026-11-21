#!/usr/bin/env python3
"""Helper script for setting up the MVTec Anomaly Detection dataset.

The MVTec AD dataset is released under a **CC BY-NC-SA 4.0** license and must
be downloaded manually from the official website.  This script guides the
user through the process and, if a local tarball is provided, extracts it
into the target directory.

Usage::

    # Print download instructions
    python scripts/download_dataset.py

    # Extract an already-downloaded tarball
    python scripts/download_dataset.py --tarball mvtec_anomaly_detection.tar.xz --output-dir ./data
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

OFFICIAL_URL = "https://www.mvtec.com/company/research/datasets/mvtec-ad"

EXPECTED_STRUCTURE = """\
<output-dir>/
└── mvtec_anomaly_detection/
    ├── bottle/
    │   ├── train/
    │   │   └── good/
    │   ├── test/
    │   │   ├── good/
    │   │   ├── broken_large/
    │   │   ├── broken_small/
    │   │   └── contamination/
    │   └── ground_truth/
    │       ├── broken_large/
    │       ├── broken_small/
    │       └── contamination/
    ├── hazelnut/
    │   ├── train/
    │   │   └── good/
    │   ├── test/
    │   │   ├── good/
    │   │   ├── crack/
    │   │   ├── cut/
    │   │   ├── hole/
    │   │   └── print/
    │   └── ground_truth/
    │       ├── crack/
    │       ├── cut/
    │       ├── hole/
    │       └── print/
    └── ... (other categories)
"""


def print_instructions() -> None:
    """Print step-by-step download instructions to stdout."""
    print("=" * 70)
    print("  MVTec Anomaly Detection (AD) Dataset — Download Instructions")
    print("=" * 70)
    print()
    print("The MVTec AD dataset is NOT bundled with AnomalyScope AI because")
    print("it is licensed under CC BY-NC-SA 4.0 and must be obtained directly")
    print("from the copyright holders.")
    print()
    print("Steps:")
    print(f"  1. Visit the official page:\n     {OFFICIAL_URL}")
    print("  2. Accept the license agreement and download the dataset archive")
    print("     (mvtec_anomaly_detection.tar.xz, approx. 4.9 GB).")
    print("  3. Run this script again with the --tarball flag to extract:")
    print()
    print("     python scripts/download_dataset.py \\")
    print("         --tarball /path/to/mvtec_anomaly_detection.tar.xz \\")
    print("         --output-dir ./data")
    print()
    print("Expected directory structure after extraction:")
    print(EXPECTED_STRUCTURE)


def extract_tarball(tarball_path: Path, output_dir: Path) -> None:
    """Extract the MVTec AD tarball into *output_dir*.

    Args:
        tarball_path: Path to the downloaded ``.tar.xz`` (or ``.tar.gz``)
            archive file.
        output_dir: Destination directory.  Created if it does not exist.

    Raises:
        FileNotFoundError: If the tarball does not exist.
        tarfile.TarError: If the archive cannot be read.
    """
    if not tarball_path.exists():
        raise FileNotFoundError(f"Tarball not found: {tarball_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {tarball_path} → {output_dir} ...")
    with tarfile.open(tarball_path, "r:*") as tar:
        # Filter out any absolute paths or parent-directory references for safety.
        members = []
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in member.name:
                print(f"  Skipping unsafe path: {member.name}")
                continue
            members.append(member)

        tar.extractall(path=output_dir, members=members)

    print(f"Done. Extracted {len(members)} entries.")
    print()
    print("Expected directory structure:")
    print(EXPECTED_STRUCTURE)


def main() -> None:
    """Parse CLI arguments and run the appropriate action."""
    parser = argparse.ArgumentParser(
        description="Download / extract the MVTec AD dataset for AnomalyScope AI.",
    )
    parser.add_argument(
        "--tarball",
        type=str,
        default=None,
        help="Path to the downloaded MVTec AD tarball (e.g. mvtec_anomaly_detection.tar.xz). "
        "If omitted, download instructions are printed instead.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to extract the dataset into (default: ./data).",
    )

    args = parser.parse_args()

    if args.tarball is None:
        print_instructions()
        sys.exit(0)

    extract_tarball(
        tarball_path=Path(args.tarball),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
