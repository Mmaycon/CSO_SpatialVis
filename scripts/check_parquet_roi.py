#!/usr/bin/env python3
"""Quick check that ROI columns exist in cell_metadata.parquet after saving from the app.

Usage (from repo root, same venv as backend):
  python scripts/check_parquet_roi.py ./data/toy_small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect cell_metadata.parquet for $ManualAnno:* columns.")
    p.add_argument("sample_dir", type=Path, help="Path to sample folder, e.g. ./data/toy_small")
    args = p.parse_args()
    meta_path = args.sample_dir / "cell_metadata.parquet"
    if not meta_path.exists():
        print(f"No file: {meta_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(meta_path)
    manual = [c for c in df.columns if str(c).startswith("$ManualAnno:")]
    print(meta_path)
    print("columns:", list(df.columns))
    print("$ManualAnno:* columns:", manual if manual else "(none yet — save an ROI while API + DB are running)")
    if manual:
        for col in manual:
            n = df[col].notna().sum()
            print(f"  {col}: {int(n)} non-null cells")


if __name__ == "__main__":
    main()
