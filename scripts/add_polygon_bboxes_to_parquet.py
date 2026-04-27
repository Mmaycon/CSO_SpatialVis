#!/usr/bin/env python3
"""
Add min_x, min_y, max_x, max_y to polygons_lod*.parquet for faster viewport filtering
server-side. Safe to re-run; skips files that already have the columns.

Usage:
  python scripts/add_polygon_bboxes_to_parquet.py /path/to/data/xenium_511Tumor_5k_full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import pandas as pd
    import pyarrow.parquet as pq
    from shapely.geometry import shape
except ModuleNotFoundError as e:  # pragma: no cover
    print(
        f"Missing dependency: {e.name!r}.\n"
        "  pip3 install pyarrow pandas shapely\n"
        "  or:  pip3 install -r backend/requirements.txt\n"
        "  or:  ./scripts/bootstrap_python_env.sh\n"
        "  then: .venv/bin/python scripts/add_polygon_bboxes_to_parquet.py …",
        file=sys.stderr,
    )
    raise SystemExit(1) from e


def bbox_of_geojson(g: dict) -> tuple[float, float, float, float] | None:
    try:
        s = shape(g)
    except Exception:
        return None
    b = s.bounds
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def process_file(p: Path) -> bool:
    need = {"min_x", "min_y", "max_x", "max_y"}
    schema = pq.read_schema(p)
    if need.issubset(set(schema.names)):
        print(f"  skip (already has bbox): {p.name}")
        return False

    df = pq.read_table(p, memory_map=True).to_pandas()
    bxs: list[tuple[float, float, float, float] | None] = []
    for _, r in df.iterrows():
        gj = r.get("geometry_json")
        g = json.loads(gj) if isinstance(gj, str) else gj
        bxs.append(bbox_of_geojson(g) if isinstance(g, dict) else None)
    nbad = sum(1 for b in bxs if b is None)
    if nbad:
        print(f"  warn: {nbad} rows without bbox: {p.name}", file=sys.stderr)
    df["min_x"] = [b[0] if b else 0.0 for b in bxs]
    df["min_y"] = [b[1] if b else 0.0 for b in bxs]
    df["max_x"] = [b[2] if b else 0.0 for b in bxs]
    df["max_y"] = [b[3] if b else 0.0 for b in bxs]
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(p)
    print(f"  updated: {p.name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir", type=Path, help="Sample folder containing polygons_lod*.parquet")
    args = ap.parse_args()
    d = args.sample_dir
    if not d.is_dir():
        print(f"Not a directory: {d}", file=sys.stderr)
        return 1
    n = 0
    for pat in ("polygons_lod0.parquet", "polygons_lod1.parquet", "polygons_lod2.parquet"):
        p = d / pat
        if p.exists() and process_file(p):
            n += 1
    if n == 0:
        print("No polygon files updated (missing or already had bbox).")
    else:
        print(f"Done. {n} file(s) written. Restart the API to clear polygon caches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
