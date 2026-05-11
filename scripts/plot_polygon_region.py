#!/usr/bin/env python3
"""
Load cell polygons in a microns bounding box and plot with matplotlib.

Supports:
  1) CSO_SpatialVis export: polygons_lod0.parquet (etc.) with geometry_json + min_* / max_*.
  2) Raw Xenium: cell_boundaries.parquet (vertex_x, vertex_y, cell_id) — two-pass chunked scan.

Examples:
  # Export folder + explicit bbox (µm)
  python scripts/plot_polygon_region.py \\
    --export-dir /path/to/data/xenium_subset_5k \\
    --min-x 100 --min-y 200 --max-x 400 --max-y 500

  # Raw boundaries (large file ok — chunked)
  python scripts/plot_polygon_region.py \\
    --boundaries /path/to/cell_boundaries.parquet \\
    --min-x 100 --min-y 200 --max-x 400 --max-y 500

  # Auto window: center of data, 250 µm square
  python scripts/plot_polygon_region.py --export-dir ... --window-um 250
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, shape


def _bbox_overlap(
    min_x: float, min_y: float, max_x: float, max_y: float,
    bx0: float, by0: float, bx1: float, by1: float,
) -> bool:
    return not (max_x < bx0 or min_x > bx1 or max_y < by0 or min_y > by1)


def load_from_export(parquet_path: Path, bbox: tuple[float, float, float, float]) -> list[tuple[str, Any]]:
    bx0, by0, bx1, by1 = bbox
    df = pq.read_table(parquet_path).to_pandas()
    if "geometry_json" not in df.columns:
        raise ValueError(f"{parquet_path} has no geometry_json (use --boundaries for raw Xenium?)")
    if all(c in df.columns for c in ("min_x", "max_x", "min_y", "max_y")):
        m = (
            (df["max_x"] >= bx0)
            & (df["min_x"] <= bx1)
            & (df["max_y"] >= by0)
            & (df["min_y"] <= by1)
        )
        df = df[m]
    else:
        # No bbox columns: parse and test (slower)
        keep = []
        for i, gj in enumerate(df["geometry_json"].astype(str)):
            g = shape(json.loads(gj))
            mn_x, mn_y, mx_x, mx_y = g.bounds
            if _bbox_overlap(mn_x, mn_y, mx_x, mx_y, bx0, by0, bx1, by1):
                keep.append(i)
        df = df.iloc[keep]
    out: list[tuple[str, Any]] = []
    for _, r in df.iterrows():
        cid = str(r["cell_id"])
        g = shape(json.loads(str(r["geometry_json"])))
        out.append((cid, g))
    return out


def load_from_raw_boundaries(path: Path, bbox: tuple[float, float, float, float], batch_rows: int) -> list[tuple[str, Polygon]]:
    bx0, by0, bx1, by1 = bbox
    cols = ["cell_id", "vertex_x", "vertex_y"]
    pf = pq.ParquetFile(path)
    needed: set[str] = set()

    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        t = batch.to_pandas()
        if t.empty:
            continue
        x = t["vertex_x"].to_numpy(dtype=np.float64, copy=False)
        y = t["vertex_y"].to_numpy(dtype=np.float64, copy=False)
        inside = (x >= bx0) & (x <= bx1) & (y >= by0) & (y <= by1)
        if not inside.any():
            continue
        cids = t["cell_id"].astype(str).to_numpy()
        needed.update(np.unique(cids[inside]).tolist())

    if not needed:
        return []

    by: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        t = batch.to_pandas()
        if t.empty:
            continue
        cids = t["cell_id"].astype(str).to_numpy()
        mask = np.isin(cids, list(needed))
        if not mask.any():
            continue
        t = t.loc[mask]
        for _, r in t.iterrows():
            by[str(r["cell_id"])].append((float(r["vertex_x"]), float(r["vertex_y"])))

    out: list[tuple[str, Polygon]] = []
    for cid, pts in by.items():
        if len(pts) < 3:
            continue
        poly = Polygon(pts)
        if not poly.is_valid:
            from shapely.validation import make_valid

            poly = make_valid(poly)
        if poly.geom_type == "Polygon":
            out.append((cid, poly))
        elif poly.geom_type == "MultiPolygon" and poly.geoms:
            out.append((cid, max(poly.geoms, key=lambda p: p.area)))
    return out


def _collect_patches(geoms: list[tuple[str, Any]], max_plot: int) -> list[MplPolygon]:
    patches: list[MplPolygon] = []
    for _, g in geoms[:max_plot] if max_plot > 0 else geoms:
        if g.geom_type == "Polygon":
            patches.append(MplPolygon(list(g.exterior.coords), closed=True))
        elif g.geom_type == "MultiPolygon":
            for p in g.geoms:
                patches.append(MplPolygon(list(p.exterior.coords), closed=True))
    return patches


def _infer_bbox_from_export_dir(export_dir: Path, window_um: float) -> tuple[float, float, float, float]:
    p = export_dir / "cells.parquet"
    if not p.exists():
        raise FileNotFoundError(f"No cells.parquet in {export_dir} (needed for --window-um)")
    cdf = pq.read_table(p, columns=["x", "y"]).to_pandas()
    cx = float(cdf["x"].mean())
    cy = float(cdf["y"].mean())
    h = window_um / 2.0
    return (cx - h, cy - h, cx + h, cy + h)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", type=Path, help="CSO sample folder (uses polygons_lod0.parquet + optional cells.parquet)")
    ap.add_argument(
        "--polygons-parquet",
        type=Path,
        help="Explicit path to polygons_lod*.parquet (overrides export-dir default)",
    )
    ap.add_argument("--boundaries", type=Path, help="Raw Xenium cell_boundaries.parquet")
    ap.add_argument("--lod", type=int, default=0, help="LOD index when using --export-dir (default 0)")
    ap.add_argument("--batch-rows", type=int, default=500_000, help="Chunk size for raw boundaries")
    ap.add_argument("--min-x", type=float, default=None)
    ap.add_argument("--min-y", type=float, default=None)
    ap.add_argument("--max-x", type=float, default=None)
    ap.add_argument("--max-y", type=float, default=None)
    ap.add_argument("--window-um", type=float, default=None, help="If set with --export-dir and no bbox: square side centered on cells mean")
    ap.add_argument("--max-cells", type=int, default=15_000, help="Cap polygons plotted (0 = no cap)")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Save PNG instead of interactive show")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    n_src = sum(x is not None for x in (args.export_dir, args.polygons_parquet, args.boundaries))
    if n_src != 1:
        print("Exactly one of --export-dir, --polygons-parquet, or --boundaries is required.", file=sys.stderr)
        return 1

    bbox: tuple[float, float, float, float] | None = None
    if args.min_x is not None and args.min_y is not None and args.max_x is not None and args.max_y is not None:
        bbox = (args.min_x, args.min_y, args.max_x, args.max_y)
    elif args.export_dir and args.window_um is not None:
        bbox = _infer_bbox_from_export_dir(args.export_dir.resolve(), float(args.window_um))
        print(f"Inferred bbox (centered window {args.window_um} µm): {bbox}")
    if bbox is None:
        print("Provide --min-x/--min-y/--max-x/--max-y or --export-dir with --window-um.", file=sys.stderr)
        return 1

    bx0, by0, bx1, by1 = bbox
    if bx1 <= bx0 or by1 <= by0:
        print("Invalid bbox: max must be greater than min.", file=sys.stderr)
        return 1

    if args.boundaries:
        geoms = load_from_raw_boundaries(args.boundaries.resolve(), bbox, args.batch_rows)
        source = str(args.boundaries)
    else:
        if args.polygons_parquet:
            pp = args.polygons_parquet.resolve()
        else:
            pp = args.export_dir.resolve() / f"polygons_lod{args.lod}.parquet"
        if not pp.is_file():
            print(f"Missing {pp}", file=sys.stderr)
            return 1
        geoms = load_from_export(pp, bbox)
        source = str(pp)

    print(f"Source: {source}")
    print(f"BBox µm: ({bx0:.2f}, {by0:.2f}) — ({bx1:.2f}, {by1:.2f})")
    print(f"Polygons in region: {len(geoms)}")

    patches = _collect_patches(geoms, args.max_cells)
    if args.max_cells and len(geoms) > args.max_cells:
        print(f"Plotting first {args.max_cells} of {len(geoms)} (--max-cells)")

    fig, ax = plt.subplots(figsize=(10, 10))
    if patches:
        col = PatchCollection(
            patches,
            facecolors="#4C72B0",
            edgecolors="#222222",
            linewidths=0.15,
            alpha=0.45,
        )
        ax.add_collection(col)
    ax.set_aspect("equal")
    pad = max(bx1 - bx0, by1 - by0) * 0.02
    ax.set_xlim(bx0 - pad, bx1 + pad)
    ax.set_ylim(by0 - pad, by1 + pad)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(f"Cell polygons ({len(geoms)} in bbox)")

    if args.output:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Wrote {args.output}")
    else:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
