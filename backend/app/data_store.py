"""
Load precomputed on-disk assets (parquet, JSON). Viewport-filtered reads only.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import TypeAdapter

from app.config import settings
from app.schemas import SampleManifest


@lru_cache(maxsize=4)
def _manifest_path(root: str) -> Path:
    return Path(root) / "manifest.json"


def load_manifest() -> SampleManifest:
    root = settings.data_root
    mp = _manifest_path(str(root))
    if not mp.exists():
        # dev fallback
        alt = Path(__file__).resolve().parents[2] / "data" / "manifest.json"
        if alt.exists():
            mp = alt
        else:
            raise FileNotFoundError(f"No manifest at {mp}")
    data = json.loads(mp.read_text())
    return TypeAdapter(SampleManifest).validate_python(data)


def sample_dir(sample_id: str) -> Path:
    root = settings.data_root
    d = root / sample_id
    if not d.exists():
        alt = Path(__file__).resolve().parents[2] / "data" / sample_id
        if alt.exists():
            return alt
    return d


_parquet_table_cache: dict[str, tuple[float, pa.Table]] = {}
_meta_index_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_polygon_frame_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_pathology_tags_cache: dict[str, tuple[float, pd.DataFrame]] = {}


def _cache_path_key(path: Path) -> str:
    return str(path.resolve())


def _read_cells_table_cached(cells_path: Path) -> pa.Table:
    key = _cache_path_key(cells_path)
    mtime = cells_path.stat().st_mtime
    hit = _parquet_table_cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    t = pq.read_table(cells_path, columns=["cell_id", "x", "y"], memory_map=True)
    _parquet_table_cache[key] = (mtime, t)
    return t


def _read_cell_metadata_index_cached(meta_path: Path) -> pd.DataFrame:
    key = _cache_path_key(meta_path)
    mtime = meta_path.stat().st_mtime
    hit = _meta_index_cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    meta_tbl = pq.read_table(meta_path, memory_map=True)
    meta_df = meta_tbl.to_pandas().set_index("cell_id")
    meta_df.index = meta_df.index.map(str)
    _meta_index_cache[key] = (mtime, meta_df)
    return meta_df


def _read_polygon_dataframe_cached(path: Path) -> pd.DataFrame:
    key = _cache_path_key(path)
    mtime = path.stat().st_mtime
    hit = _polygon_frame_cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    df = pq.read_table(path, memory_map=True).to_pandas()
    _polygon_frame_cache[key] = (mtime, df)
    return df


def _read_pathology_tags_df_cached(p: Path) -> pd.DataFrame:
    key = _cache_path_key(p)
    mtime = p.stat().st_mtime
    hit = _pathology_tags_cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    df = pq.read_table(p, memory_map=True).to_pandas()
    _pathology_tags_cache[key] = (mtime, df)
    return df


def read_cells_viewport(
    sample_id: str,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    max_cells: int,
) -> tuple[list[dict[str, Any]], int, bool]:
    cells_path = sample_dir(sample_id) / "cells.parquet"
    if not cells_path.exists():
        return [], 0, False

    table = _read_cells_table_cached(cells_path)
    x = table["x"]
    y = table["y"]
    # Use pc.and_ — ChunkedArray from parquet cannot be combined with Python &.
    mask = pc.and_(
        pc.and_(pc.greater_equal(x, min_x), pc.less_equal(x, max_x)),
        pc.and_(pc.greater_equal(y, min_y), pc.less_equal(y, max_y)),
    )
    filtered = table.filter(mask)
    total = int(filtered.num_rows)
    if total == 0:
        return [], 0, False

    truncated = total > max_cells
    if truncated:
        idx = np.sort(np.random.default_rng(42).choice(total, size=max_cells, replace=False))
        filtered = filtered.take(pa.array(idx, type=pa.uint32()))

    n = int(filtered.num_rows)
    cell_ids: list[str] = [str(c) for c in filtered["cell_id"].to_pylist()]
    xs = np.ascontiguousarray(np.asarray(filtered["x"].to_numpy(zero_copy_only=False), dtype=np.float64))
    ys = np.ascontiguousarray(np.asarray(filtered["y"].to_numpy(zero_copy_only=False), dtype=np.float64))

    meta_path = sample_dir(sample_id) / "cell_metadata.parquet"
    meta_df: pd.DataFrame | None = None
    if meta_path.exists():
        meta_df = _read_cell_metadata_index_cached(meta_path)

    rows: list[dict[str, Any]] = []
    if meta_df is not None:
        for i in range(n):
            cid = cell_ids[i]
            if cid in meta_df.index:
                md = meta_df.loc[cid].to_dict()
            else:
                md = {}
            rows.append({"cell_id": cid, "x": float(xs[i]), "y": float(ys[i]), "metadata": md})
    else:
        for i in range(n):
            rows.append(
                {
                    "cell_id": cell_ids[i],
                    "x": float(xs[i]),
                    "y": float(ys[i]),
                    "metadata": {},
                }
            )

    return rows, total, truncated


def read_polygons_viewport(
    sample_id: str,
    lod: int,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    max_features: int,
) -> tuple[list[dict[str, Any]], int, bool]:
    lod = max(0, min(2, lod))
    path = sample_dir(sample_id) / f"polygons_lod{lod}.parquet"
    if not path.exists():
        path = sample_dir(sample_id) / "polygons_lod0.parquet"
    if not path.exists():
        return [], 0, False

    df = _read_polygon_dataframe_cached(path)
    has_precomputed_bbox = {"min_x", "min_y", "max_x", "max_y"}.issubset(df.columns)
    if has_precomputed_bbox:
        m = (
            (df["max_x"] >= min_x)
            & (df["min_x"] <= max_x)
            & (df["max_y"] >= min_y)
            & (df["min_y"] <= max_y)
        )
        work = df.loc[m]
        total = int(len(work))
        truncated = total > max_features
        if total > 0 and truncated:
            work = work.sample(n=max_features, random_state=42)
    else:
        work = df
        total = 0
        truncated = False  # set after list built

    feats: list[dict[str, Any]] = []
    has_wkt = "wkt" in work.columns
    if has_wkt:
        from shapely import wkt as shapely_wkt

    if has_precomputed_bbox:
        for idx, r in work.iterrows():
            gj = r.get("geometry_json")
            if gj is None and has_wkt and r.get("wkt") is not None:
                g = shapely_wkt.loads(str(r["wkt"]))
                gj = json.loads(json.dumps(g.__geo_interface__))
            elif isinstance(gj, str):
                gj = json.loads(gj)
            if gj is None:
                continue
            pid = str(r.get("id", idx))
            props: dict[str, Any] = {}
            if "cell_id" in r and r.get("cell_id") is not None:
                props["cell_id"] = str(r["cell_id"])
            feats.append(
                {
                    "id": pid,
                    "geometry": gj,
                    "properties": props,
                }
            )
    else:
        for idx, r in work.iterrows():
            gj = r.get("geometry_json")
            if gj is None and has_wkt and r.get("wkt") is not None:
                g = shapely_wkt.loads(str(r["wkt"]))
                gj = json.loads(json.dumps(g.__geo_interface__))
            elif isinstance(gj, str):
                gj = json.loads(gj)

            if gj is None:
                continue
            xs, ys = _bbox_of_geometry(gj)
            if xs[1] < min_x or xs[0] > max_x or ys[1] < min_y or ys[0] > max_y:
                continue

            pid = str(r.get("id", idx))
            props: dict[str, Any] = {}
            if "cell_id" in r and r.get("cell_id") is not None:
                props["cell_id"] = str(r["cell_id"])
            feats.append(
                {
                    "id": pid,
                    "geometry": gj,
                    "properties": props,
                }
            )
        total = int(len(feats))
        truncated = total > max_features
        if total > max_features and max_features > 0:
            idx = np.sort(np.random.default_rng(42).choice(total, size=max_features, replace=False))
            feats = [feats[i] for i in idx]

    return feats, total, truncated


def _bbox_of_geometry(g: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    b = _try_bbox_geojson_fast(g)
    if b is not None:
        (minx, miny, maxx, maxy) = b
        return (minx, maxx), (miny, maxy)
    from shapely.geometry import shape

    s = shape(g)
    minx, miny, maxx, maxy = s.bounds
    return (minx, maxx), (miny, maxy)


def _try_bbox_geojson_fast(g: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Bounds without Shapely for common Polygon / MultiPolygon GeoJSON (large speedup vs shape())."""
    try:
        t = g.get("type")
        coords = g.get("coordinates")
        if t == "Polygon" and isinstance(coords, list) and coords:
            ring0 = coords[0]
            if not ring0 or not isinstance(ring0, list):
                return None
            xs = [float(p[0]) for p in ring0 if len(p) >= 2]
            ys = [float(p[1]) for p in ring0 if len(p) >= 2]
            if not xs:
                return None
            return (min(xs), min(ys), max(xs), max(ys))
        if t == "MultiPolygon" and isinstance(coords, list) and coords:
            minx = miny = float("inf")
            maxx = maxy = float("-inf")
            for poly in coords:
                if not poly or not isinstance(poly, list):
                    continue
                ring0 = poly[0]
                if not ring0:
                    continue
                for p in ring0:
                    if not isinstance(p, (list, tuple)) or len(p) < 2:
                        continue
                    x, y = float(p[0]), float(p[1])
                    minx, maxx = min(minx, x), max(maxx, x)
                    miny, maxy = min(miny, y), max(maxy, y)
            if not (minx < float("inf") and maxx > float("-inf")):
                return None
            return (minx, miny, maxx, maxy)
    except (TypeError, ValueError, KeyError):
        return None
    return None


def read_transcript_tile(sample_id: str, z: int, x: int, y: int) -> list[tuple[float, float, str]]:
    p = sample_dir(sample_id) / "transcripts" / f"{z}_{x}_{y}.parquet"
    if not p.exists():
        return []
    df = pq.read_table(p).to_pandas()
    out: list[tuple[float, float, str]] = []
    for _, r in df.iterrows():
        out.append((float(r["x"]), float(r["y"]), str(r.get("gene_id", ""))))
    return out


_gene_column_cache: dict[tuple[str, str], dict[str, float]] = {}


def read_gene_expression(sample_id: str, gene: str, cell_ids: list[str] | None) -> dict[str, float]:
    key = (sample_id, gene.lower())
    if key in _gene_column_cache and cell_ids is None:
        return dict(_gene_column_cache[key])

    expr_path = sample_dir(sample_id) / "expression-wide.parquet"
    if not expr_path.exists():
        return {}

    # Read only one column + cell_id for memory efficiency
    try:
        table = pq.read_table(expr_path, columns=["cell_id", gene])
    except Exception:
        # case-insensitive fallback: find column
        schema = pq.read_schema(expr_path)
        names = {n.lower(): n for n in schema.names}
        col = names.get(gene.lower())
        if col is None:
            return {}
        table = pq.read_table(expr_path, columns=["cell_id", col])

    df = table.to_pandas()
    expr_cols = [c for c in df.columns if str(c).lower() != "cell_id"]
    if len(expr_cols) != 1:
        return {}
    resolved_gene_col = expr_cols[0]

    if cell_ids is not None:
        sid = set(cell_ids)
        df = df[df["cell_id"].astype(str).isin(sid)]

    cids = df["cell_id"].astype(str)
    gcol = df[resolved_gene_col]
    mask = gcol.notna()
    cids2 = cids[mask]
    g2 = gcol[mask]
    vals = {str(c): float(v) for c, v in zip(cids2.tolist(), g2.tolist(), strict=True)}

    if cell_ids is None and len(vals) < 500_000:
        _gene_column_cache[key] = vals
        # bound cache size
        if len(_gene_column_cache) > settings.gene_cache_size:
            first = next(iter(_gene_column_cache))
            del _gene_column_cache[first]

    return vals


def bbox_polygon_for_cell_ids(sample_id: str, cell_ids: list[str]) -> dict[str, Any]:
    """Axis-aligned bbox around given cells as GeoJSON Polygon (for storing rect-like selections)."""
    cells_path = sample_dir(sample_id) / "cells.parquet"
    if not cells_path.exists():
        return {"type": "Polygon", "coordinates": []}

    tbl = pq.read_table(cells_path, columns=["cell_id", "x", "y"])
    df = tbl.to_pandas()
    want = set(cell_ids)
    df = df[df["cell_id"].astype(str).isin(want)]
    if df.empty:
        return {"type": "Polygon", "coordinates": []}

    pad = 2.0
    minx = float(df["x"].min() - pad)
    maxx = float(df["x"].max() + pad)
    miny = float(df["y"].min() - pad)
    maxy = float(df["y"].max() + pad)
    ring = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
    return {"type": "Polygon", "coordinates": [ring]}


def read_plot(sample_id: str, plot_type: str) -> dict[str, Any]:
    p = sample_dir(sample_id) / "plots" / f"{plot_type}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


MANUAL_ANNO_PREFIX = "$ManualAnno:"


def manual_anno_column_name(label: str) -> str:
    """Parquet column for manual ROI label; value stored is the label string for assigned cells."""
    return f"{MANUAL_ANNO_PREFIX}{label}"


def list_metadata_columns_ordered(sample_id: str) -> list[str]:
    """cell_metadata.parquet columns (excluding cell_id), manual ROI columns last."""
    mp = sample_dir(sample_id) / "cell_metadata.parquet"
    if not mp.exists():
        return []
    meta_cols = [c for c in pq.read_schema(mp).names if c != "cell_id"]
    manual = sorted([c for c in meta_cols if str(c).startswith(MANUAL_ANNO_PREFIX)])
    rest = sorted([c for c in meta_cols if not str(c).startswith(MANUAL_ANNO_PREFIX)])
    return rest + manual


def list_color_columns(sample_id: str) -> dict[str, Any]:
    """Metadata column names from cell_metadata.parquet + expression genes (for Color UI)."""
    return {
        "metadata_columns": list_metadata_columns_ordered(sample_id),
        "genes": list_expression_genes(sample_id),
    }


def list_expression_genes(sample_id: str) -> list[str]:
    expr_path = sample_dir(sample_id) / "expression-wide.parquet"
    if not expr_path.exists():
        return []
    schema = pq.read_schema(expr_path)
    # Keep all feature columns; skip only common id / spatial / sample key fields.
    skip_lower = {
        "cell_id",
        "cell-id",
        "cellid",
        "sample_id",
        "sample-id",
        "sampleid",
        "fov",
        "x",
        "y",
    }
    genes = sorted(
        [
            n
            for n in schema.names
            if str(n).strip() and str(n).strip().lower() not in skip_lower
        ],
        key=str.lower,
    )
    return genes


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def sync_sample_roi_parquet_from_tags(
    sample_id: str,
    tags: list[dict[str, Any]],
) -> None:
    """Rewrite cell_metadata pathology columns, $ManualAnno:{label} columns, and pathology_cell_tags."""

    root = sample_dir(sample_id)
    cells_path = root / "cells.parquet"
    if not cells_path.exists():
        return

    all_ids = pq.read_table(cells_path, columns=["cell_id"]).to_pandas()
    all_ids["cell_id"] = all_ids["cell_id"].astype(str)
    tag_map: dict[str, dict[str, Any]] = {}
    for t in tags:
        cid = str(t["cell_id"])
        tag_map[cid] = t

    meta_path = root / "cell_metadata.parquet"
    if meta_path.exists():
        meta = pq.read_table(meta_path).to_pandas()
    else:
        meta = all_ids.copy()
    meta["cell_id"] = meta["cell_id"].astype(str)

    want = set(all_ids["cell_id"].tolist())
    existing = set(meta["cell_id"].tolist())
    missing_ids = sorted(want - existing)
    if missing_ids:
        meta = pd.concat([meta, pd.DataFrame({"cell_id": missing_ids})], ignore_index=True)

    # Drop prior manual-annotation columns so schema matches current DB labels (same label name overwrites).
    manual_old = [c for c in meta.columns if str(c).startswith(MANUAL_ANNO_PREFIX)]
    if manual_old:
        meta = meta.drop(columns=manual_old, errors="ignore")

    for col in ("pathology_region", "pathology_region_source", "pathology_region_confidence"):
        if col not in meta.columns:
            meta[col] = pd.NA

    labels_in_use = sorted(
        {str(t.get("pathology_region")) for t in tags if t.get("pathology_region") not in (None, "")}
    )
    for L in labels_in_use:
        meta[manual_anno_column_name(L)] = pd.NA

    for cid in all_ids["cell_id"].tolist():
        idx = meta["cell_id"] == cid
        t = tag_map.get(cid)
        if t:
            meta.loc[idx, "pathology_region"] = t.get("pathology_region")
            meta.loc[idx, "pathology_region_source"] = t.get("pathology_region_source", "roi_assignment")
            meta.loc[idx, "pathology_region_confidence"] = float(t.get("pathology_region_confidence") or 1.0)
        else:
            meta.loc[idx, "pathology_region"] = pd.NA
            meta.loc[idx, "pathology_region_source"] = pd.NA
            meta.loc[idx, "pathology_region_confidence"] = pd.NA

        pr = None
        if t:
            pv = t.get("pathology_region")
            if pv is not None and str(pv).strip() != "":
                pr = str(pv)
        for L in labels_in_use:
            col = manual_anno_column_name(L)
            meta.loc[idx, col] = L if pr == L else pd.NA

    _atomic_write_parquet(meta, meta_path)

    tags_path = root / "pathology_cell_tags.parquet"
    if tags:
        tag_df = pd.DataFrame(tags)
        _atomic_write_parquet(tag_df, tags_path)
    elif tags_path.exists():
        tags_path.unlink()


def overlay_pathology_metadata(sample_id: str, cell_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Merge precomputed tags file with empty default."""
    p = sample_dir(sample_id) / "pathology_cell_tags.parquet"
    if not p.exists() or not cell_ids:
        return {c: {} for c in cell_ids}
    want = set(cell_ids)
    df = _read_pathology_tags_df_cached(p)
    df = df[df["cell_id"].astype(str).isin(want)]
    out: dict[str, dict[str, Any]] = {c: {} for c in cell_ids}
    for _, r in df.iterrows():
        cid = str(r["cell_id"])
        if cid not in out:
            continue
        out[cid] = {
            k: r[k]
            for k in df.columns
            if k != "cell_id" and pd.notna(r[k])
        }
    return out
