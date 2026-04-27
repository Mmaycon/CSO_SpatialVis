#!/usr/bin/env python3
"""
Export a **full** 10x Xenium region output directory to CSO_SpatialVis sample assets.

Same file layout as ``xenium_subset_to_spatialvis.py``, but **no random subsampling** of cells:
every cell in the export pool is written (subject to the same cells∩matrix filter as the subset
script, unless ``--ignore-matrix-barcode-filter``).

The transcript tile ``transcripts/0_0_0.parquet`` may still be **row-capped** via
``--transcript-max`` for performance on huge runs; raise it if you need more points.

Dependencies:
  pip install -r scripts/requirements-xenium-pipeline.txt

Example:
  python scripts/xenium_to_spatialvis.py \\
    --xenium-dir /path/to/output-...__Region__... \\
    --out-dir  /path/to/CSO_SpatialVis/data/xenium_myregion_full
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable


def _load_run_export() -> Callable[[argparse.Namespace], int]:
    """Load ``run_export`` from ``xenium_subset_to_spatialvis.py`` next to this file or under ``scripts/``."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "xenium_subset_to_spatialvis.py",
        here / "scripts" / "xenium_subset_to_spatialvis.py",
        here.parent / "scripts" / "xenium_subset_to_spatialvis.py",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        name = "_xenium_subset_loaded"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            continue
        mod: Any = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "run_export", None)
        if callable(fn):
            return fn
        raise ImportError(
            f"Found {path} but it has no ``run_export()`` — update this file from the current "
            "CSO_SpatialVis repo (the subset script was refactored to share logic with the full export)."
        )
    raise ImportError(
        "Could not find xenium_subset_to_spatialvis.py. Expected it next to this script, or under "
        "scripts/xenium_subset_to_spatialvis.py from the repo root. Clone/pull CSO_SpatialVis and run "
        "``python scripts/xenium_to_spatialvis.py`` from the repository root."
    )


run_export = _load_run_export()


def parse_args_full() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert a full Xenium region to CSO_SpatialVis (all cells in pool; no random subset).",
    )
    ap.add_argument("--xenium-dir", type=Path, required=True, help="Xenium output region directory")
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output sample folder, e.g. data/xenium_myregion_full",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed when downsampling transcript rows to --transcript-max",
    )
    ap.add_argument(
        "--transcript-max",
        type=int,
        default=200_000,
        help="Max transcript rows in transcripts/0_0_0.parquet (tile); increase for denser tiles.",
    )
    ap.add_argument("--he-long-edge", type=int, default=1600, help="PNG long edge in pixels")
    ap.add_argument(
        "--skip-he-morphology",
        action="store_true",
        help="Do not try to read OME-TIFF; write a placeholder he.png in cropped bounds only",
    )
    ap.add_argument(
        "--ignore-matrix-barcode-filter",
        action="store_true",
        help="Include all cells from cells.parquet even if absent from feature matrix (zeros in expression-wide).",
    )
    args = ap.parse_args()
    args.export_all_cells = True
    args.n_cells = 0  # unused when export_all_cells is True
    return args


def main() -> int:
    return run_export(parse_args_full())


if __name__ == "__main__":
    raise SystemExit(main())
