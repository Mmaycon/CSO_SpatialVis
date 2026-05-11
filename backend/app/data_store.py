"""
Load precomputed on-disk assets (parquet, JSON). Viewport-filtered reads only.
"""

from __future__ import annotations

import json
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


def _resolve_data_root() -> Path:
    root = Path(settings.data_root).resolve()
    if root.is_dir():
        return root
    alt = Path(__file__).resolve().parents[2] / "data"
    return alt.resolve()


def _bounds_from_cells_parquet(cells_path: Path, pad: float = 40.0) -> dict[str, float]:
    tbl = pq.read_table(cells_path, columns=["x", "y"])
    pdf = tbl.to_pandas()
    mn_x = float(pdf["x"].min())
    mx_x = float(pdf["x"].max())
    mn_y = float(pdf["y"].min())
    mx_y = float(pdf["y"].max())
    return {
        "min_x": mn_x - pad,
        "max_x": mx_x + pad,
        "min_y": mn_y - pad,
        "max_y": mx_y + pad,
    }


def _genes_available_count(sample_dir: Path) -> int:
    expr_path = sample_dir / "expression-wide.parquet"
    if not expr_path.exists():
        return 0
    schema = pq.read_schema(expr_path)
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
    return len([n for n in schema.names if str(n).strip() and str(n).strip().lower() not in skip_lower])


def _parse_sidecar_manifest(root: Path) -> tuple[str, list[str], set[str], dict[str, dict[str, Any]]]:
    mp = root / "manifest.json"
    version = "1.0"
    order: list[str] = []
    exclude: set[str] = set()
    overrides: dict[str, dict[str, Any]] = {}
    if not mp.exists():
        return version, order, exclude, overrides
    try:
        data = json.loads(mp.read_text())
    except json.JSONDecodeError:
        return version, order, exclude, overrides
    version = str(data.get("version") or "1.0")
    order = [str(x) for x in (data.get("sample_order") or data.get("preferred_sample_order") or [])]
    raw_ex = data.get("discovery_exclude") or data.get("exclude_sample_ids") or []
    exclude = {str(x) for x in raw_ex}
    for item in data.get("samples") or []:
        if isinstance(item, dict) and item.get("sample_id"):
            overrides[str(item["sample_id"])] = item
    return version, order, exclude, overrides


def persist_sample_image_translate_um(sample_id: str, dx_um: float, dy_um: float) -> None:
    """Write ``image.alignment.translate_um`` into ``data/manifest.json`` for a discovered sample.

    Creates ``manifest.json`` or a minimal ``samples[]`` entry when missing. Deep-merge-safe because
    :func:`_merge_manifest_override` merges ``alignment`` keys.
    """
    root = _resolve_data_root()
    sid = str(sample_id).strip()
    if not sid:
        raise ValueError("sample_id is required")
    sample_path = root / sid
    if not (sample_path / "cells.parquet").is_file():
        raise FileNotFoundError(f"No sample with cells.parquet under {sample_path}")

    mp = root / "manifest.json"
    if mp.is_file():
        try:
            data: Any = json.loads(mp.read_text())
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("version", "1.0")
    samples_raw = data.get("samples")
    samples: list[Any] = list(samples_raw) if isinstance(samples_raw, list) else []

    idx = -1
    for i, item in enumerate(samples):
        if isinstance(item, dict) and str(item.get("sample_id")) == sid:
            idx = i
            break

    if idx < 0:
        entry: dict[str, Any] = {"sample_id": sid}
        samples.append(entry)
        idx = len(samples) - 1
    else:
        entry = dict(samples[idx]) if isinstance(samples[idx], dict) else {"sample_id": sid}

    img = dict(entry.get("image") or {})
    al = dict(img.get("alignment") or {})
    al["translate_um"] = [float(dx_um), float(dy_um)]
    img["alignment"] = al
    entry["image"] = img
    samples[idx] = entry
    data["samples"] = samples

    mp.parent.mkdir(parents=True, exist_ok=True)
    tmp = mp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(mp)


def _discover_sample_directories(root: Path, exclude: set[str]) -> list[Path]:
    out: list[Path] = []
    if not root.is_dir():
        return out
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        if p.name in exclude:
            continue
        if not (p / "cells.parquet").is_file():
            continue
        out.append(p)
    return out


def _ensure_image_urls(
    image: dict[str, Any] | None,
    sid: str,
    bounds: dict[str, float],
    sample_dir: Path,
) -> dict[str, Any] | None:
    has_he = (sample_dir / "he.png").is_file()
    if image is None or image is False:
        image = None
    if image is None:
        if not has_he:
            return None
        image = {
            "type": "osd_simple",
            "url": f"/api/assets/he/{sid}",
            "alignment": {"mode": "identity", "bounds": dict(bounds)},
        }
    else:
        img = dict(image)
        img["type"] = img.get("type") or "osd_simple"
        img["url"] = f"/api/assets/he/{sid}"
        al = img.get("alignment")
        if not isinstance(al, dict):
            al = {}
        else:
            al = dict(al)
        al.setdefault("mode", "identity")
        alb = al.get("bounds")
        if not isinstance(alb, dict) or not alb:
            al["bounds"] = dict(bounds)
        img["alignment"] = al
        chans = img.get("channels")
        if isinstance(chans, list):
            fixed: list[dict[str, Any]] = []
            for c in chans:
                if not isinstance(c, dict):
                    continue
                cc = dict(c)
                rel = cc.get("relative_path")
                if isinstance(rel, str) and rel.strip():
                    rel_clean = rel.strip().lstrip("/")
                    cc["relative_path"] = rel_clean
                    cc["url"] = f"/api/assets/sample_files/{sid}/{rel_clean}"
                fixed.append(cc)
            img["channels"] = fixed
        image = img

    im_man = sample_dir / "images_manifest.json"
    existing_ch = image.get("channels") if isinstance(image, dict) else None
    if isinstance(existing_ch, list) and len(existing_ch) > 0:
        return image
    if not im_man.is_file():
        return image
    try:
        raw_im = json.loads(im_man.read_text())
    except json.JSONDecodeError:
        return image
    chans = raw_im.get("channels") or []
    if not chans:
        return image
    fixed_ch: list[dict[str, Any]] = []
    for c in chans:
        if not isinstance(c, dict):
            continue
        rel = c.get("relative_path")
        if not isinstance(rel, str) or not rel.strip():
            continue
        rel_clean = rel.strip().lstrip("/")
        fixed_ch.append(
            {
                "id": str(c.get("id", rel_clean)),
                "label": str(c.get("label", c.get("id", rel_clean))),
                "relative_path": rel_clean,
                "url": f"/api/assets/sample_files/{sid}/{rel_clean}",
                "default_visible": bool(c.get("default_visible", True)),
            }
        )
    if not fixed_ch:
        return image
    img = dict(image)
    img["channels"] = fixed_ch
    return img


def _apply_registration_image_metadata(
    sample_dir: Path,
    sid: str,
    image: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Ensure registration_channel_id and stable registration PNG URL for sidecar pipelines."""
    if image is None:
        return None
    img = dict(image)
    ref_png = sample_dir / "Images" / "registration_reference.png"
    if ref_png.is_file():
        img["registration_reference_url"] = f"/api/assets/sample_files/{sid}/Images/registration_reference.png"

    imp = sample_dir / "images_manifest.json"
    if imp.is_file():
        try:
            raw = json.loads(imp.read_text())
        except json.JSONDecodeError:
            raw = {}
        if isinstance(raw, dict):
            rid = raw.get("registration_channel_id")
            rl = raw.get("registration_channel_label")
            if isinstance(rid, str) and rid.strip():
                img["registration_channel_id"] = rid.strip()
            if isinstance(rl, str) and rl.strip():
                img["registration_channel_label"] = rl.strip()
            if raw.get("viewer_flip_y") is True:
                al = dict(img.get("alignment") or {})
                alb = al.get("bounds")
                if isinstance(alb, dict) and {"min_x", "min_y", "max_x", "max_y"}.issubset(alb):
                    al.setdefault("mode", "identity")
                    al["flip_y"] = True
                    img["alignment"] = al
            mf = raw.get("morphology_frame")
            if isinstance(mf, dict) and isinstance(mf.get("crop_microns"), dict):
                img["morphology_frame"] = dict(mf)

    chs = img.get("channels")
    if isinstance(chs, list) and chs:
        if not img.get("registration_channel_id"):
            dapi = next((c for c in chs if "dapi" in str(c.get("label", "")).lower()), None)
            pick = dapi if isinstance(dapi, dict) else chs[0]
            if isinstance(pick, dict) and pick.get("id") is not None:
                img["registration_channel_id"] = str(pick["id"])

    return img


def _merge_manifest_override(base: dict[str, Any], over: dict[str, Any], sid: str) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if k == "sample_id":
            continue
        if v is None:
            continue
        if k == "image" and isinstance(v, dict):
            bi = out.get("image")
            if not isinstance(bi, dict):
                out["image"] = dict(v)
            else:
                merged_img = {**bi, **v}
                ba, va = bi.get("alignment"), v.get("alignment")
                if isinstance(ba, dict) and isinstance(va, dict):
                    merged_img["alignment"] = {**ba, **va}
                out["image"] = merged_img
        else:
            out[k] = v
    out["sample_id"] = sid
    return out


def _build_sample_ref(sample_dir: Path, overrides: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    sid = sample_dir.name
    cells_path = sample_dir / "cells.parquet"
    if not cells_path.is_file():
        return None

    bounds_disk = _bounds_from_cells_parquet(cells_path)
    genes = _genes_available_count(sample_dir)

    entry_path = sample_dir / "sample_manifest_entry.json"
    if entry_path.is_file():
        try:
            raw = json.loads(entry_path.read_text())
        except json.JSONDecodeError:
            raw = {}
        if not isinstance(raw, dict):
            base = {}
        else:
            base = dict(raw)
            base["sample_id"] = sid
            b = base.get("bounds")
            if not isinstance(b, dict) or not {"min_x", "min_y", "max_x", "max_y"}.issubset(b):
                base["bounds"] = bounds_disk
            base.setdefault("coordinate_unit", "microns")
            base.setdefault("layers", ["centroids", "segmentation", "transcripts", "he"])
            base.setdefault("title", sid)
            base.setdefault("description", "")
            base["genes_available"] = max(int(base.get("genes_available") or 0), genes)
    else:
        base = {
            "sample_id": sid,
            "title": sid,
            "description": "",
            "coordinate_unit": "microns",
            "bounds": bounds_disk,
            "layers": ["centroids", "segmentation", "transcripts", "he"],
            "genes_available": genes,
            "image": None,
        }

    ov = overrides.get(sid)
    if ov:
        base = _merge_manifest_override(base, ov, sid)

    base["bounds"] = base.get("bounds") or bounds_disk
    base["genes_available"] = max(int(base.get("genes_available") or 0), genes)
    base["image"] = _ensure_image_urls(base.get("image"), sid, base["bounds"], sample_dir)
    base["image"] = _apply_registration_image_metadata(sample_dir, sid, base.get("image"))
    base.setdefault("coordinate_unit", "microns")
    base.setdefault("layers", ["centroids", "segmentation", "transcripts", "he"])
    base.setdefault("title", sid)
    base.setdefault("description", "")
    return base


def _sort_sample_refs(samples: list[dict[str, Any]], preferred_order: list[str]) -> list[dict[str, Any]]:
    idx = {sid: i for i, sid in enumerate(preferred_order)}

    def sort_key(s: dict[str, Any]) -> tuple[int, str]:
        sid = str(s.get("sample_id", ""))
        return (idx.get(sid, 10_000), sid)

    return sorted(samples, key=sort_key)


def load_manifest() -> SampleManifest:
    """Build manifest by scanning ``data/*/cells.parquet`` and merging optional ``manifest.json`` overrides."""
    root = _resolve_data_root()
    version, sample_order, exclude, overrides = _parse_sidecar_manifest(root)

    discovered_dirs = _discover_sample_directories(root, exclude)
    samples_raw: list[dict[str, Any]] = []
    for d in discovered_dirs:
        ref = _build_sample_ref(d, overrides)
        if ref:
            samples_raw.append(ref)

    samples_sorted = _sort_sample_refs(samples_raw, sample_order)
    payload = {"version": version, "samples": samples_sorted}
    return TypeAdapter(SampleManifest).validate_python(payload)


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
    max_cells: int | None,
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

    truncated = max_cells is not None and total > max_cells
    if truncated and max_cells is not None:
        idx = np.sort(np.random.default_rng(42).choice(total, size=max_cells, replace=False))
        filtered = filtered.take(pa.array(idx, type=pa.uint32()))

    pdf = filtered.select(["cell_id", "x", "y"]).to_pandas()
    pdf["cell_id"] = pdf["cell_id"].astype(str)

    meta_path = sample_dir(sample_id) / "cell_metadata.parquet"
    meta_df: pd.DataFrame | None = None
    if meta_path.exists():
        meta_df = _read_cell_metadata_index_cached(meta_path)

    if meta_df is not None:
        meta_aligned = meta_df.reindex(pdf["cell_id"].values)
        meta_records = meta_aligned.to_dict(orient="records")
        n = len(pdf)
        rows = [
            {
                "cell_id": pdf["cell_id"].iat[i],
                "x": float(pdf["x"].iat[i]),
                "y": float(pdf["y"].iat[i]),
                "metadata": {k: v for k, v in meta_records[i].items() if pd.notna(v)},
            }
            for i in range(n)
        ]
    else:
        rows = [
            {
                "cell_id": row.cell_id,
                "x": float(row.x),
                "y": float(row.y),
                "metadata": {},
            }
            for row in pdf.itertuples(index=False)
        ]

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


# Integer columns with at most this many distinct values are colored as categories (e.g. cluster bins).
_METADATA_INT_MAX_CATEGORICAL_UNIQ = 48


def _infer_metadata_column_kind(series: pd.Series, name: str) -> str:
    """Classify a cell_metadata column for coloring: viridis vs hashed categorical."""
    if str(name).startswith(MANUAL_ANNO_PREFIX):
        return "categorical"
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        nu = int(series.dropna().nunique())
        if nu <= _METADATA_INT_MAX_CATEGORICAL_UNIQ:
            return "categorical"
        return "numeric"
    coerced = pd.to_numeric(series, errors="coerce")
    nn = int(series.notna().sum())
    if nn == 0:
        return "categorical"
    finite = int(coerced.notna().sum())
    if finite / nn >= 0.9:
        return "numeric"
    return "categorical"


def infer_metadata_column_kinds(sample_id: str) -> dict[str, str]:
    """Map metadata column name -> ``numeric`` or ``categorical`` (from cell_metadata.parquet)."""
    mp = sample_dir(sample_id) / "cell_metadata.parquet"
    if not mp.exists():
        return {}
    df = pd.read_parquet(mp)
    out: dict[str, str] = {}
    for col in df.columns:
        if str(col) == "cell_id":
            continue
        out[str(col)] = _infer_metadata_column_kind(df[col], str(col))
    return out


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
        "metadata_column_kinds": infer_metadata_column_kinds(sample_id),
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


def invalidate_sample_parquet_caches(sample_id: str) -> None:
    """Clear process-local parquet caches so ROI writes are visible on the next API read."""
    root = sample_dir(sample_id)
    meta_path = root / "cell_metadata.parquet"
    tags_path = root / "pathology_cell_tags.parquet"
    _meta_index_cache.pop(_cache_path_key(meta_path), None)
    _pathology_tags_cache.pop(_cache_path_key(tags_path), None)


def sync_sample_roi_parquet_from_tags(
    sample_id: str,
    tags: list[dict[str, Any]],
) -> None:
    """Rewrite cell_metadata pathology columns, $ManualAnno:{label} columns, and pathology_cell_tags."""

    root = sample_dir(sample_id)
    cells_path = root / "cells.parquet"
    if not cells_path.exists():
        raise FileNotFoundError(f"cells.parquet not found for sample {sample_id!r} under {root}")

    all_ids = pq.read_table(cells_path, columns=["cell_id"]).to_pandas()
    all_ids["cell_id"] = all_ids["cell_id"].astype(str)
    want = set(all_ids["cell_id"].tolist())

    meta_path = root / "cell_metadata.parquet"
    if meta_path.exists():
        meta = pq.read_table(meta_path).to_pandas()
    else:
        meta = all_ids.copy()
    meta["cell_id"] = meta["cell_id"].astype(str)

    existing_ids = set(meta["cell_id"].tolist())
    missing_ids = sorted(want - existing_ids)
    if missing_ids:
        meta = pd.concat([meta, pd.DataFrame({"cell_id": missing_ids})], ignore_index=True)

    # Drop prior manual-annotation columns so schema matches current DB labels (same label name overwrites).
    manual_old = [c for c in meta.columns if str(c).startswith(MANUAL_ANNO_PREFIX)]
    if manual_old:
        meta = meta.drop(columns=manual_old, errors="ignore")

    # Parquet often loads pathology_region_confidence as float64; assigning tag-derived Series with
    # mixed floats + NA then raises TypeError/LossySetitemError. Use object columns like row-wise NA writes.
    for col in ("pathology_region", "pathology_region_source", "pathology_region_confidence"):
        if col not in meta.columns:
            meta[col] = pd.NA
        meta[col] = meta[col].astype(object)

    labels_in_use = sorted(
        {str(t.get("pathology_region")) for t in tags if t.get("pathology_region") not in (None, "")}
    )
    for L in labels_in_use:
        meta[manual_anno_column_name(L)] = pd.NA

    want_mask = meta["cell_id"].isin(want)
    orphan_mask = ~want_mask

    # Preserve pathology columns for rows not in cells.parquet (legacy rows); only canonical cells are rebuilt from tags.
    orphan_pr = meta.loc[orphan_mask, "pathology_region"].copy()
    orphan_src = meta.loc[orphan_mask, "pathology_region_source"].copy()
    orphan_conf = meta.loc[orphan_mask, "pathology_region_confidence"].copy()

    tag_map: dict[str, dict[str, Any]] = {}
    for t in tags:
        cid = str(t["cell_id"])
        tag_map[cid] = t

    cid_want = meta.loc[want_mask, "cell_id"]
    pr_map = {k: v.get("pathology_region") for k, v in tag_map.items()}
    src_map = {k: v.get("pathology_region_source", "roi_assignment") for k, v in tag_map.items()}
    conf_map = {k: float(v.get("pathology_region_confidence") or 1.0) for k, v in tag_map.items()}

    meta.loc[want_mask, "pathology_region"] = cid_want.map(pr_map).astype(object)
    meta.loc[want_mask, "pathology_region_source"] = cid_want.map(src_map).astype(object)
    meta.loc[want_mask, "pathology_region_confidence"] = cid_want.map(conf_map).astype(object)

    meta.loc[orphan_mask, "pathology_region"] = orphan_pr
    meta.loc[orphan_mask, "pathology_region_source"] = orphan_src
    meta.loc[orphan_mask, "pathology_region_confidence"] = orphan_conf

    pr_full = meta["pathology_region"]
    pr_str = pr_full.astype("string")
    for L in labels_in_use:
        col = manual_anno_column_name(L)
        match = want_mask & pr_str.notna() & (pr_str == str(L))
        meta.loc[match, col] = L

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
    tag_cols = [c for c in df.columns if c != "cell_id"]
    for r in df.itertuples(index=False):
        cid = str(getattr(r, "cell_id"))
        if cid not in out:
            continue
        out[cid] = {c: getattr(r, c) for c in tag_cols if pd.notna(getattr(r, c))}
    return out
