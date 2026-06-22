#!/usr/bin/env python3
"""
Export a CosMx ``*.sdata.zarr`` store to CSO_SpatialVis sample assets (Parquet + PNG).

Mirrors the layout produced by ``xenium_subset_to_spatialvis.py`` / ``xenium_to_spatialvis.py``:
``cells.parquet``, ``cell_metadata.parquet``, ``expression-wide.parquet``, ``polygons_lod*.parquet``,
``transcripts/0_0_0.parquet``, morphology PNGs under ``Images/``, ``he.png``, ``images_manifest.json``,
and ``sample_manifest_entry.json``.

Cell coordinates are written in **micrometers** (``global`` pixel coords × ``pixel_size_um`` from the
SpatialData store), matching the Xenium exporter convention.

Dependencies:
  pip install -r scripts/requirements-cosmx-pipeline.txt

Example (full GLP1 breast cancer run):
  python scripts/cosmx_to_spatialvis.py \\
    --sdata-zarr /path/to/cosmx.sdata.zarr \\
    --morphology-zarr /path/to/morphology.ome.zarr \\
    --out-dir ./data/GLP1_BrCa_cosmx \\
    --export-all-cells

Quick subset:
  python scripts/cosmx_to_spatialvis.py \\
    --sdata-zarr /path/to/cosmx.sdata.zarr \\
    --out-dir ./data/GLP1_BrCa_5k \\
    --n-cells 5000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from scipy.sparse import issparse
from shapely.affinity import scale as shapely_scale
from shapely.validation import make_valid
from spatialdata import read_zarr


def _load_xenium_helpers() -> Any:
    here = Path(__file__).resolve().parent
    path = here / "xenium_subset_to_spatialvis.py"
    if not path.is_file():
        raise ImportError(f"Missing shared helper module: {path}")
    spec = importlib.util.spec_from_file_location("_xenium_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_xh = _load_xenium_helpers()
_json_sanitize = _xh._json_sanitize
_write_poly_lods = _xh._write_poly_lods
_extract_morphology_crop_u8 = _xh._extract_morphology_crop_u8
_save_u8_crop_to_png = _xh._save_u8_crop_to_png
_crop_resize_plane_to_png = _xh._crop_resize_plane_to_png
_crop_microns_dict = _xh._crop_microns_dict
_normalize_fullplane_uint8 = _xh._normalize_fullplane_uint8
_synthetic_umap = _xh._synthetic_umap
_stable_channel_id = _xh._stable_channel_id
_slug_channel_filename = _xh._slug_channel_filename
_default_visible_channel = _xh._default_visible_channel
_pick_registration_channel_index = _xh._pick_registration_channel_index
_write_he_synthetic = _xh._write_he_synthetic


INSTANCE_KEY = "cell_uid"
GLOBAL_OBSM_KEY = "global"
DEFAULT_COSMX_CHANNEL_LABELS = ("PanCK", "G", "CD298_B2M", "CD45", "DNA")


def _normalize_plane_percentile_uint8(
    a: np.ndarray,
    *,
    low_pct: float = 1.0,
    high_pct: float = 99.5,
) -> np.ndarray:
    """Contrast-stretch a 2D plane to uint8 using nonzero percentiles.

    Min-max scaling crushes a full stitched slide to near-black because a few
    bright pixels dominate the range. Using percentiles over nonzero pixels
    yields Xenium-like contrast for sparse fluorescence (DNA/PanCK/etc.).
    """
    x = np.asarray(a, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(x.shape, dtype=np.uint8)
    vals = x[finite]
    nz = vals[vals > 0]
    sample = nz if nz.size else vals
    lo = float(np.percentile(sample, low_pct))
    hi = float(np.percentile(sample, high_pct))
    if hi <= lo:
        lo = float(sample.min())
        hi = float(sample.max())
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def _read_morphology_channel_labels(sdata_zarr: Path, n_ch: int) -> list[str]:
    """Best-effort channel labels from SpatialData or CosMx defaults."""
    labels: list[str] = []
    for rel in (
        sdata_zarr / "images" / "morphology" / ".zattrs",
        sdata_zarr.parent / "morphology.ome.zarr" / ".zattrs",
    ):
        if not rel.is_file():
            continue
        try:
            attrs = json.loads(rel.read_text())
        except json.JSONDecodeError:
            continue
        omero = attrs.get("omero") or (attrs.get("multiscales") or [{}])[0].get("metadata", {}).get("omero")
        if isinstance(omero, dict):
            labels = [str(c.get("label", "")).strip() for c in omero.get("channels") or []]
            labels = [x for x in labels if x]
        if labels:
            break
    if len(labels) < n_ch and n_ch == len(DEFAULT_COSMX_CHANNEL_LABELS):
        labels = list(DEFAULT_COSMX_CHANNEL_LABELS)
    if len(labels) < n_ch:
        labels.extend(f"channel_{i:02d}" for i in range(len(labels), n_ch))
    return labels[:n_ch]


def _read_root_attrs(sdata_zarr: Path) -> dict[str, Any]:
    p = sdata_zarr / ".zattrs"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def _resolve_morphology_zarr(sdata_zarr: Path, morphology_zarr: Path | None) -> Path:
    if morphology_zarr is not None:
        p = morphology_zarr.resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"--morphology-zarr not found: {p}")
        return p
    for cand in (
        sdata_zarr.parent / "morphology.ome.zarr",
        sdata_zarr / "images" / "morphology",
    ):
        if cand.is_dir():
            return cand.resolve()
    raise FileNotFoundError(
        "Could not find morphology OME-Zarr. Pass --morphology-zarr (recommended: morphology.ome.zarr "
        "with a multiscale pyramid)."
    )


def _load_ome_pyramid_meta(ome_zarr: Path) -> list[dict[str, Any]]:
    attrs = json.loads((ome_zarr / ".zattrs").read_text())
    datasets = attrs["multiscales"][0]["datasets"]
    omero = attrs.get("omero") or attrs["multiscales"][0].get("metadata", {}).get("omero") or {}
    channel_labels = [str(c.get("label", f"ch{i}")) for i, c in enumerate(omero.get("channels") or [])]
    out: list[dict[str, Any]] = []
    for ds in datasets:
        lev = int(ds["path"])
        sy = float(ds["coordinateTransformations"][0]["scale"][1])
        shape = json.loads((ome_zarr / ds["path"] / ".zarray").read_text())["shape"]
        out.append({"level": lev, "pixel_um": sy, "shape": shape, "path": ds["path"]})
    out.sort(key=lambda d: d["level"])
    if not channel_labels and out:
        n_ch = int(out[0]["shape"][0])
        channel_labels = [f"channel_{i:02d}" for i in range(n_ch)]
    return out


def _full_slide_bounds_um(root_attrs: dict[str, Any], pixel_um: float) -> dict[str, float]:
    """Micron bounds covering the full stitched morphology (origin at top-left)."""
    shape = root_attrs.get("mosaic_shape_yx") or root_attrs.get("image_shape_yx")
    if not shape or len(shape) < 2:
        raise ValueError("sdata .zattrs missing mosaic_shape_yx / image_shape_yx")
    h_px, w_px = float(shape[0]), float(shape[1])
    return {
        "min_x": 0.0,
        "min_y": 0.0,
        "max_x": w_px * pixel_um,
        "max_y": h_px * pixel_um,
    }


def _pick_ome_level(
    bounds_um: dict[str, float],
    levels: list[dict[str, Any]],
    pixel_um_full: float,
    morphology_max_long_edge: int,
    *,
    full_slide: bool = False,
) -> dict[str, Any]:
    cap = max(512, int(morphology_max_long_edge))
    if full_slide and levels:
        chosen = levels[-1]
        for lev in levels:
            _, h0, w0 = (int(x) for x in lev["shape"])
            chosen = lev
            if max(h0, w0) <= cap * 1.25:
                break
        return chosen

    span_um = max(
        float(bounds_um["max_x"] - bounds_um["min_x"]),
        float(bounds_um["max_y"] - bounds_um["min_y"]),
    )
    span_px_full = span_um / max(pixel_um_full, 1e-9)

    chosen = levels[0]
    for lev in levels:
        factor = lev["pixel_um"] / max(pixel_um_full, 1e-9)
        span_px = span_px_full / max(factor, 1e-9)
        chosen = lev
        if span_px <= cap * 1.25:
            break
    return chosen


def _sample_cell_ids(pool: list[str], n_cells: int, seed: int, export_all: bool) -> list[str]:
    if export_all:
        return list(pool)
    n_take = int(min(n_cells, len(pool)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=n_take, replace=False)
    return [pool[i] for i in sorted(idx)]


def _sanitize_meta_frame(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    for col in out.columns:
        if col == "cell_id":
            continue
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(str).replace({"nan": "", "None": "", "<NA>": ""}).astype(object)
        elif out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].astype(str).replace({"nan": "", "None": "", "<NA>": ""}).astype(object)
    return out


def _export_expression_wide(
    table,
    chosen_list: list[str],
    out_path: Path,
    *,
    batch_size: int = 4096,
) -> int:
    """Write log1p expression-wide.parquet in row batches from sparse AnnData X."""
    id_to_row = {str(v): i for i, v in enumerate(table.obs_names.astype(str))}
    valid_chosen = [cid for cid in chosen_list if cid in id_to_row]
    row_idx = [id_to_row[cid] for cid in valid_chosen]
    if len(valid_chosen) != len(chosen_list):
        missing = len(chosen_list) - len(valid_chosen)
        print(f"Warning: {missing} chosen cell_id(s) missing from table index.", file=sys.stderr)

    gene_names = [str(g) for g in table.var_names.tolist()]
    n_genes = len(gene_names)
    writer: pq.ParquetWriter | None = None
    n_written = 0

    for start in range(0, len(row_idx), batch_size):
        batch_rows = row_idx[start : start + batch_size]
        batch_ids = [valid_chosen[start + i] for i in range(len(batch_rows))]
        x_batch = table.X[batch_rows]
        if issparse(x_batch):
            arr = x_batch.toarray()
        else:
            arr = np.asarray(x_batch)
        arr = np.log1p(np.maximum(arr.astype(np.float32, copy=False), 0.0))
        data = {"cell_id": batch_ids}
        for j, gn in enumerate(gene_names):
            data[gn] = arr[:, j]
        df = pd.DataFrame(data)
        table_pa = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table_pa.schema, compression="zstd")
        writer.write_table(table_pa)
        n_written += len(batch_ids)
        print(f"  expression batch: {n_written:,} / {len(row_idx):,} cells", flush=True)

    if writer is not None:
        writer.close()
    elif n_genes == 0:
        pd.DataFrame({"cell_id": valid_chosen}).to_parquet(out_path, index=False)
    return n_genes


def _polygons_from_shapes(
    shapes_path: Path,
    chosen_set: set[str],
    pixel_um: float,
) -> dict[str, Any]:
    from shapely.geometry import Polygon

    gdf = gpd.read_parquet(shapes_path)
    if gdf.empty:
        return {}

    idx_name = gdf.index.name or INSTANCE_KEY
    polys: dict[str, Polygon] = {}
    for idx, geom in gdf.geometry.items():
        cid = str(idx)
        if idx_name and idx_name != "geometry":
            cid = str(idx)
        if cid not in chosen_set:
            continue
        if geom is None or geom.is_empty:
            continue
        g = geom
        if not g.is_valid:
            g = make_valid(g)
        g = shapely_scale(g, xfact=pixel_um, yfact=pixel_um, origin=(0.0, 0.0))
        if g.geom_type == "Polygon":
            polys[cid] = g
        elif g.geom_type == "MultiPolygon" and len(g.geoms) > 0:
            polys[cid] = max(g.geoms, key=lambda p: p.area)
    return polys


def _export_transcripts_tile(
    transcripts_parquet: Path,
    bounds_um: dict[str, float],
    pixel_um: float,
    out_tile: Path,
    *,
    max_rows: int,
    seed: int,
) -> None:
    if not transcripts_parquet.is_file():
        print(f"Warning: transcripts source not found: {transcripts_parquet}", file=sys.stderr)
        pd.DataFrame({"x": [], "y": [], "gene_id": []}).to_parquet(out_tile, index=False)
        return

    min_x_px = float(bounds_um["min_x"]) / pixel_um
    max_x_px = float(bounds_um["max_x"]) / pixel_um
    min_y_px = float(bounds_um["min_y"]) / pixel_um
    max_y_px = float(bounds_um["max_y"]) / pixel_um

    pf = pq.ParquetFile(transcripts_parquet)
    cols = pf.schema_arrow.names
    gene_col = "target" if "target" in cols else ("gene_id" if "gene_id" in cols else None)
    if gene_col is None:
        print(f"Warning: no gene column in {transcripts_parquet}; skipping transcripts.", file=sys.stderr)
        pd.DataFrame({"x": [], "y": [], "gene_id": []}).to_parquet(out_tile, index=False)
        return

    rng = np.random.default_rng(seed)
    kept_x: list[float] = []
    kept_y: list[float] = []
    kept_g: list[str] = []
    seen = 0

    for batch in pf.iter_batches(batch_size=500_000, columns=["x", "y", gene_col]):
        df = batch.to_pandas()
        m = (
            (df["x"] >= min_x_px)
            & (df["x"] <= max_x_px)
            & (df["y"] >= min_y_px)
            & (df["y"] <= max_y_px)
        )
        sub = df.loc[m]
        if sub.empty:
            continue
        seen += len(sub)
        if len(kept_x) + len(sub) > max_rows:
            take = max_rows - len(kept_x)
            if take <= 0:
                break
            sub = sub.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
        kept_x.extend((sub["x"].astype(float) * pixel_um).tolist())
        kept_y.extend((sub["y"].astype(float) * pixel_um).tolist())
        kept_g.extend(sub[gene_col].astype(str).tolist())
        if len(kept_x) >= max_rows:
            break

    out = pd.DataFrame({"x": kept_x, "y": kept_y, "gene_id": kept_g})
    out.to_parquet(out_tile, index=False)
    print(f"Transcripts tile: kept {len(out):,} rows (scanned ~{seen:,} in bounds).")


def _write_morphology_from_ome_zarr(
    ome_zarr: Path,
    bounds_um: dict[str, float],
    out_dir: Path,
    out_he_png: Path,
    sample_folder_name: str,
    *,
    sdata_zarr: Path,
    pixel_um_full: float,
    registration_channel_spec: str,
    he_long_edge: int,
    morphology_max_long_edge: int,
    morphology_flip_y: bool,
    full_slide: bool = True,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    levels = _load_ome_pyramid_meta(ome_zarr)
    if not levels:
        return False, [], {}

    lev = _pick_ome_level(
        bounds_um,
        levels,
        pixel_um_full,
        morphology_max_long_edge,
        full_slide=full_slide,
    )
    level_key = str(lev["level"])
    pixel_um = float(lev["pixel_um"])
    shape = lev["shape"]  # (c, y, x)
    n_ch = int(shape[0])
    h0, w0 = int(shape[1]), int(shape[2])

    labels = _read_morphology_channel_labels(sdata_zarr, n_ch)

    if full_slide:
        xa, ya, xb, yb = 0, 0, w0, h0
        print(
            f"Morphology: reading full slide OME-Zarr level {level_key} "
            f"shape ({n_ch}, {h0}, {w0}) pixel_um={pixel_um:.4f}",
            flush=True,
        )
    else:
        def um_to_ix_iy(x_um: float, y_um: float) -> tuple[int, int]:
            xi = int(np.clip(x_um / pixel_um, 0, w0 - 1))
            yi = int(np.clip(y_um / pixel_um, 0, h0 - 1))
            return xi, yi

        x0, y0 = um_to_ix_iy(bounds_um["min_x"], bounds_um["min_y"])
        x1, y1 = um_to_ix_iy(bounds_um["max_x"], bounds_um["max_y"])
        xa, xb = sorted([x0, x1])
        ya, yb = sorted([y0, y1])
        pad = 8
        xa = max(0, xa - pad)
        xb = min(w0, xb + pad)
        ya = max(0, ya - pad)
        yb = min(h0, yb + pad)
        if xb - xa < 2 or yb - ya < 2:
            return False, [], {}
        print(
            f"Morphology: reading OME-Zarr level {level_key} crop y=[{ya}:{yb}) x=[{xa}:{xb}) "
            f"({n_ch} channels, pixel_um={pixel_um:.4f})",
            flush=True,
        )

    zg = zarr.open(str(ome_zarr), mode="r")
    vol = np.asarray(zg[level_key][:, ya:yb, xa:xb])

    images_dir = out_dir / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)

    channel_planes: list[tuple[np.ndarray, str, int]] = []
    for i in range(n_ch):
        plane = vol[i]
        u8 = _normalize_plane_percentile_uint8(plane)
        channel_planes.append((u8, labels[i], i))

    used_ids: set[str] = set()
    channel_ids = [_stable_channel_id(lab, idx, used_ids) for _, lab, idx in channel_planes]
    provisional = [{"id": channel_ids[i], "label": labels[i]} for i in range(n_ch)]
    reg_idx = _pick_registration_channel_index(registration_channel_spec, provisional, labels)
    reg_idx = max(0, min(reg_idx, n_ch - 1))

    full_h, full_w = h0, w0
    geo = {
        "xa": int(xa),
        "ya": int(ya),
        "xb": int(xb),
        "yb": int(yb),
        "crop_w": int(xb - xa),
        "crop_h": int(yb - ya),
        "full_w": int(full_w),
        "full_h": int(full_h),
        "pixel_um": float(pixel_um),
    }

    ref_u8 = channel_planes[reg_idx][0]
    if full_slide:
        crop_ref = ref_u8 if not morphology_flip_y else np.ascontiguousarray(np.flipud(ref_u8))
        geo_ref = dict(geo)
    else:
        crop_ref, geo_ref = _extract_morphology_crop_u8(ref_u8, bounds_um, pixel_um, morphology_flip_y)
        if crop_ref is None:
            crop_ref = ref_u8 if not morphology_flip_y else np.ascontiguousarray(np.flipud(ref_u8))
            geo_ref = dict(geo)

    merged_ok = False
    png_meta_he: dict[str, Any] = {}
    if crop_ref is not None:
        merged_ok, png_meta_he = _save_u8_crop_to_png(
            crop_ref, he_long_edge, out_he_png, morphology_max_long_edge
        )

    morphology_frame = {
        "version": 1,
        "coordinate_unit": "micrometer",
        "convention_note": (
            "cells.parquet x,y are CosMx cell centroids in micrometers (global pixel coords × pixel_size_um). "
            "Morphology PNGs are cropped from the OME-NGFF pyramid to the cell bounds."
        ),
        "pixel_um": float(geo_ref.get("pixel_um", pixel_um)),
        "full_image_shape_px": [int(geo_ref.get("full_h", full_h)), int(geo_ref.get("full_w", full_w))],
        "crop_origin_ix": int(geo_ref.get("xa", xa)),
        "crop_origin_iy": int(geo_ref.get("ya", ya)),
        "crop_w": int(geo_ref.get("crop_w", xb - xa)),
        "crop_h": int(geo_ref.get("crop_h", yb - ya)),
        "crop_microns": _crop_microns_dict(geo_ref if geo_ref else geo),
        "morphology_flip_y_export": bool(morphology_flip_y),
        "segmentation_reference_channel_id": channel_ids[reg_idx],
        "segmentation_reference_channel_label": str(labels[reg_idx]),
        "segmentation_reference_channel_index": int(reg_idx),
        "export_micron_bounds": dict(bounds_um),
        "ome_zarr_level": int(lev["level"]),
    }

    manifest_channels: list[dict[str, Any]] = []
    for u8, lab, idx in channel_planes:
        fname = f"{idx:02d}_{_slug_channel_filename(lab, idx)}.png"
        rel_path = f"Images/{fname}"
        out_png = images_dir / fname
        if full_slide:
            plane = u8 if not morphology_flip_y else np.ascontiguousarray(np.flipud(u8))
            ok_plane, _ = _save_u8_crop_to_png(
                plane, he_long_edge, out_png, morphology_max_long_edge
            )
        else:
            ok_plane, _ = _crop_resize_plane_to_png(
                u8,
                bounds_um,
                pixel_um,
                he_long_edge,
                out_png,
                morphology_flip_y=morphology_flip_y,
                morphology_max_long_edge=morphology_max_long_edge,
            )
        if not ok_plane:
            continue
        cid = channel_ids[idx]
        manifest_channels.append(
            {
                "id": cid,
                "label": lab,
                "relative_path": rel_path,
                "url": f"/api/assets/sample_files/{sample_folder_name}/{rel_path}",
                "default_visible": _default_visible_channel(lab, idx, labels),
            }
        )

    reg_meta: dict[str, Any] = {}
    if manifest_channels:
        reg_ch_id = channel_ids[reg_idx]
        reg_entry = next((e for e in manifest_channels if e["id"] == reg_ch_id), manifest_channels[0])
        src_reg = images_dir / f"{reg_idx:02d}_{_slug_channel_filename(labels[reg_idx], reg_idx)}.png"
        dst_reg = images_dir / "registration_reference.png"
        if src_reg.is_file():
            shutil.copyfile(src_reg, dst_reg)
        im_man: dict[str, Any] = {
            "version": 3,
            "micron_bounds": bounds_um,
            "coordinate_note": (
                "cells.parquet uses CosMx micrometer coordinates on the same axes as morphology "
                "(see morphology_frame for pixel↔µm mapping)."
            ),
            "pixel_um": pixel_um,
            "ome_axes": "CYX",
            "pyramid_level": int(lev["level"]),
            "export_long_edge": he_long_edge,
            "morphology_max_long_edge": morphology_max_long_edge,
            "morphology_flip_y": bool(morphology_flip_y),
            "he_png_geometry": png_meta_he,
            "channels": [{k: v for k, v in ch.items() if k != "url"} for ch in manifest_channels],
            "registration_channel_spec": registration_channel_spec,
            "registration_channel_id": reg_entry["id"],
            "registration_channel_label": reg_entry["label"],
            "registration_reference_relative_path": "Images/registration_reference.png",
            "morphology_frame": _json_sanitize(morphology_frame),
        }
        (out_dir / "images_manifest.json").write_text(json.dumps(_json_sanitize(im_man), indent=2) + "\n")
        reg_meta = {
            "registration_channel_id": str(reg_entry["id"]),
            "registration_reference_url": (
                f"/api/assets/sample_files/{sample_folder_name}/Images/registration_reference.png"
            ),
        }

    if not merged_ok and manifest_channels:
        first_png = out_dir / Path(manifest_channels[0]["relative_path"])
        if first_png.is_file():
            shutil.copyfile(first_png, out_he_png)
            merged_ok = True

    return bool(merged_ok or manifest_channels), manifest_channels, reg_meta


def run_morphology_only(args: argparse.Namespace) -> int:
    """Re-export he.png + Images/ for an existing sample folder (skip cells/expression)."""
    sdata_path = args.sdata_zarr.resolve()
    out = args.out_dir.resolve()
    if not out.is_dir() or not (out / "cells.parquet").is_file():
        print(f"Expected existing sample with cells.parquet under {out}", file=sys.stderr)
        return 1

    root_attrs = _read_root_attrs(sdata_path)
    pixel_um = float(root_attrs.get("pixel_size_um") or args.pixel_size_um)
    cells = pd.read_parquet(out / "cells.parquet", columns=["x", "y"])
    pad = 40.0 * pixel_um
    bounds = {
        "min_x": float(cells["x"].min() - pad),
        "min_y": float(cells["y"].min() - pad),
        "max_x": float(cells["x"].max() + pad),
        "max_y": float(cells["y"].max() + pad),
    }
    full_slide = not bool(getattr(args, "morphology_crop_to_cells", False))
    morph_bounds = _full_slide_bounds_um(root_attrs, pixel_um) if full_slide else bounds

    reg_spec = str(args.registration_channel or "DNA").strip() or "DNA"
    if args.skip_he_morphology:
        print("--skip-he-morphology set; nothing to do.", file=sys.stderr)
        return 1

    ome_zarr = _resolve_morphology_zarr(sdata_path, args.morphology_zarr)
    ok, morph_chans, reg_meta = _write_morphology_from_ome_zarr(
        ome_zarr,
        morph_bounds,
        out,
        out / "he.png",
        out.name,
        sdata_zarr=sdata_path,
        pixel_um_full=pixel_um,
        registration_channel_spec=reg_spec,
        he_long_edge=int(args.he_long_edge),
        morphology_max_long_edge=int(args.morphology_max_long_edge),
        morphology_flip_y=bool(args.morphology_flip_y),
        full_slide=full_slide,
    )
    if not ok:
        print("Morphology export failed.", file=sys.stderr)
        return 1

    entry_path = out / "sample_manifest_entry.json"
    if entry_path.is_file():
        sample = json.loads(entry_path.read_text())
    else:
        n_genes = 0
        expr_path = out / "expression-wide.parquet"
        if expr_path.is_file():
            n_genes = max(0, len(pq.read_schema(expr_path).names) - 1)
        sample = {
            "sample_id": out.name,
            "title": out.name,
            "description": f"Exported from {sdata_path.name}",
            "coordinate_unit": "microns",
            "bounds": bounds,
            "layers": ["centroids", "segmentation", "transcripts", "he"],
            "genes_available": n_genes,
        }

    image_bounds = morph_bounds if full_slide else bounds
    image_obj: dict[str, Any] = {
        "type": "osd_simple",
        "url": f"/api/assets/he/{out.name}",
        "alignment": {"mode": "identity", "bounds": image_bounds},
    }
    if morph_chans:
        image_obj["channels"] = morph_chans
    if reg_meta.get("registration_channel_id"):
        image_obj["registration_channel_id"] = reg_meta["registration_channel_id"]
    if reg_meta.get("registration_reference_url"):
        image_obj["registration_reference_url"] = reg_meta["registration_reference_url"]
    sample["image"] = image_obj
    sample.setdefault("bounds", bounds)
    entry_path.write_text(json.dumps(sample, indent=2) + "\n")
    print("Updated morphology:", out / "he.png")
    print("Updated", entry_path)
    return 0


def run_export(args: argparse.Namespace) -> int:
    sdata_path = args.sdata_zarr.resolve()
    out = args.out_dir.resolve()
    if not sdata_path.is_dir():
        print(f"Not a directory: {sdata_path}", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    (out / "transcripts").mkdir(exist_ok=True)

    root_attrs = _read_root_attrs(sdata_path)
    pixel_um = float(root_attrs.get("pixel_size_um") or args.pixel_size_um)

    print(f"Loading SpatialData table from {sdata_path} …", flush=True)
    sdata = read_zarr(sdata_path)
    if "table" not in sdata.tables:
        print("SpatialData store has no tables/table.", file=sys.stderr)
        return 1
    table = sdata.tables["table"]

    if GLOBAL_OBSM_KEY not in table.obsm:
        print(f"table.obsm['{GLOBAL_OBSM_KEY}'] not found.", file=sys.stderr)
        return 1

    pool_ids = list(dict.fromkeys(table.obs_names.astype(str).tolist()))
    export_all = bool(getattr(args, "export_all_cells", False))
    if export_all:
        chosen_list = list(pool_ids)
        print(f"Export all cells: {len(chosen_list):,}")
    else:
        chosen_list = _sample_cell_ids(pool_ids, int(args.n_cells), int(args.seed), False)
        print(f"Export subset: {len(chosen_list):,} cells (seed={args.seed})")
    chosen_set = set(chosen_list)
    order_key = {cid: i for i, cid in enumerate(chosen_list)}

    global_xy = np.asarray(table.obsm[GLOBAL_OBSM_KEY], dtype=np.float64)
    id_to_row = {str(v): i for i, v in enumerate(table.obs_names.astype(str))}
    xs = []
    ys = []
    for cid in chosen_list:
        row = id_to_row.get(cid)
        if row is None:
            continue
        xs.append(float(global_xy[row, 0]) * pixel_um)
        ys.append(float(global_xy[row, 1]) * pixel_um)

    cells_out = pd.DataFrame({"cell_id": chosen_list, "x": xs, "y": ys})
    cells_out.to_parquet(out / "cells.parquet", index=False)
    cx_map = {str(r["cell_id"]): (float(r["x"]), float(r["y"])) for _, r in cells_out.iterrows()}

    print("Writing cell_metadata.parquet …", flush=True)
    obs = table.obs
    if hasattr(obs, "to_pandas"):
        obs = obs.to_pandas()
    obs = obs.copy()
    obs.index = obs.index.astype(str)
    if INSTANCE_KEY in obs.columns:
        obs = obs.set_index(INSTANCE_KEY, drop=False)
    meta_rows = []
    for cid in chosen_list:
        if cid not in obs.index:
            continue
        row = obs.loc[cid].copy()
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        rec = row.to_dict()
        rec["cell_id"] = str(cid)
        meta_rows.append(rec)
    meta = pd.DataFrame(meta_rows)
    meta = _sanitize_meta_frame(meta)
    if "cell_type" not in meta.columns:
        meta["cell_type"] = "unknown"
    meta.to_parquet(out / "cell_metadata.parquet", index=False)

    print("Writing expression-wide.parquet …", flush=True)
    n_genes = _export_expression_wide(table, chosen_list, out / "expression-wide.parquet")

    shapes_path = sdata_path / "shapes" / "cell_boundaries" / "shapes.parquet"
    try:
        if shapes_path.is_file():
            print("Building polygon LODs …", flush=True)
            polys = _polygons_from_shapes(shapes_path, chosen_set, pixel_um)
            _write_poly_lods(polys, cx_map, out)
        else:
            print(f"Warning: no shapes at {shapes_path}; square footprints.", file=sys.stderr)
            _write_poly_lods({}, cx_map, out)
    except Exception as exc:
        print(f"Warning: polygon export failed ({exc}); square footprints.", file=sys.stderr)
        _write_poly_lods({}, cx_map, out)

    pad = 40.0 * pixel_um
    bounds = {
        "min_x": float(cells_out["x"].min() - pad),
        "min_y": float(cells_out["y"].min() - pad),
        "max_x": float(cells_out["x"].max() + pad),
        "max_y": float(cells_out["y"].max() + pad),
    }
    full_slide = not bool(getattr(args, "morphology_crop_to_cells", False))
    morph_bounds = bounds
    if full_slide:
        try:
            morph_bounds = _full_slide_bounds_um(root_attrs, pixel_um)
            print(
                f"Morphology: full slide bounds "
                f"x=[{morph_bounds['min_x']:.1f}, {morph_bounds['max_x']:.1f}] "
                f"y=[{morph_bounds['min_y']:.1f}, {morph_bounds['max_y']:.1f}] µm",
                flush=True,
            )
        except ValueError as exc:
            print(f"Warning: {exc}; cropping morphology to cell bounds.", file=sys.stderr)
            full_slide = False
            morph_bounds = bounds

    if getattr(args, "export_transcripts", False):
        tx_src = args.transcripts_parquet
        if tx_src is None:
            tx_src = sdata_path.parent / "transcripts.parquet"
        print(f"Exporting transcript tile from {tx_src} …", flush=True)
        _export_transcripts_tile(
            Path(tx_src),
            bounds,
            pixel_um,
            out / "transcripts" / "0_0_0.parquet",
            max_rows=int(args.transcript_max),
            seed=int(args.seed),
        )
    else:
        print("Skipping transcript export (pass --export-transcripts to build the tile).")
        pd.DataFrame({"x": [], "y": [], "gene_id": []}).to_parquet(
            out / "transcripts" / "0_0_0.parquet", index=False
        )

    meta_for_plot = meta.copy()
    meta_for_plot["cell_id"] = meta_for_plot["cell_id"].astype(str)
    pts, ulabel = _synthetic_umap(chosen_list, None, meta_for_plot)
    col_for_comp = ulabel
    if col_for_comp is None and "novae_domains_20" in meta_for_plot.columns:
        col_for_comp = "novae_domains_20"
        for p in pts:
            r = meta_for_plot[meta_for_plot["cell_id"] == p["cell_id"]]
            if not r.empty:
                p["novae_domains_20"] = r["novae_domains_20"].iloc[0]
    if col_for_comp is None and "cluster" in meta_for_plot.columns:
        col_for_comp = "cluster"
    comp_labels: list[str] = []
    comp_counts: list[int] = []
    if col_for_comp and pts and all(col_for_comp in p for p in pts):
        uq: dict[Any, int] = defaultdict(int)
        for p in pts:
            v = p.get(col_for_comp)
            key = "NA" if v is None or (isinstance(v, float) and np.isnan(v)) else v
            uq[key] += 1
        comp_labels = [str(k) for k in uq]
        comp_counts = [int(uq[k]) for k in uq]
    comp = {"labels": comp_labels, "counts": comp_counts} if comp_labels else {"labels": [], "counts": []}
    plot_payload = _json_sanitize({"points": pts, "composition": comp})
    (out / "plots" / "umap.json").write_text(json.dumps(plot_payload, indent=2) + "\n")
    (out / "plots" / "composition.json").write_text(json.dumps(_json_sanitize(comp), indent=2) + "\n")

    tag_cols = [
        "cell_id",
        "pathology_region",
        "pathology_region_source",
        "pathology_region_confidence",
    ]
    pd.DataFrame(columns=tag_cols).to_parquet(out / "pathology_cell_tags.parquet", index=False)

    morph_chans: list[dict[str, Any]] = []
    reg_meta: dict[str, Any] = {}
    reg_spec = str(args.registration_channel or "DNA").strip() or "DNA"
    if not args.skip_he_morphology:
        try:
            ome_zarr = _resolve_morphology_zarr(sdata_path, args.morphology_zarr)
            ok, morph_chans, reg_meta = _write_morphology_from_ome_zarr(
                ome_zarr,
                morph_bounds,
                out,
                out / "he.png",
                out.name,
                sdata_zarr=sdata_path,
                pixel_um_full=pixel_um,
                registration_channel_spec=reg_spec,
                he_long_edge=int(args.he_long_edge),
                morphology_max_long_edge=int(args.morphology_max_long_edge),
                morphology_flip_y=bool(args.morphology_flip_y),
                full_slide=full_slide,
            )
            if not ok:
                print("Warning: morphology export produced no PNGs; placeholder he.png.", file=sys.stderr)
                _write_he_synthetic(bounds, out / "he.png", min(int(args.he_long_edge), 1420) or 1420)
        except Exception as exc:
            print(f"Warning: morphology export failed ({exc}); placeholder.", file=sys.stderr)
            _write_he_synthetic(bounds, out / "he.png", min(int(args.he_long_edge), 1420) or 1420)
    else:
        _write_he_synthetic(bounds, out / "he.png", min(int(args.he_long_edge), 1420) or 1420)

    n = len(chosen_list)
    image_bounds = morph_bounds if full_slide and morph_chans else bounds
    image_obj: dict[str, Any] = {
        "type": "osd_simple",
        "url": f"/api/assets/he/{out.name}",
        "alignment": {"mode": "identity", "bounds": image_bounds},
    }
    if morph_chans:
        image_obj["channels"] = morph_chans
    if reg_meta.get("registration_channel_id"):
        image_obj["registration_channel_id"] = reg_meta["registration_channel_id"]
    if reg_meta.get("registration_reference_url"):
        image_obj["registration_reference_url"] = reg_meta["registration_reference_url"]

    sample = {
        "sample_id": out.name,
        "title": f"CosMx GLP1 BrCa ({n:,} cells)" if export_all else f"CosMx subset ({n:,} cells)",
        "description": f"Exported from {sdata_path.name} for CSO_SpatialVis",
        "coordinate_unit": "microns",
        "bounds": bounds,
        "image": image_obj,
        "layers": ["centroids", "segmentation", "transcripts", "he"],
        "genes_available": n_genes,
    }
    (out / "sample_manifest_entry.json").write_text(json.dumps(sample, indent=2) + "\n")

    print("Done:", out)
    print("Wrote", out / "sample_manifest_entry.json")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert CosMx SpatialData Zarr to CSO_SpatialVis sample assets.")
    ap.add_argument("--sdata-zarr", type=Path, required=True, help="Path to cosmx.sdata.zarr")
    ap.add_argument(
        "--morphology-zarr",
        type=Path,
        default=None,
        help="OME-NGFF morphology pyramid (default: sibling morphology.ome.zarr)",
    )
    ap.add_argument(
        "--transcripts-parquet",
        type=Path,
        default=None,
        help="Cached transcripts Parquet (default: parent dir transcripts.parquet)",
    )
    ap.add_argument("--out-dir", type=Path, required=True, help="Output sample folder, e.g. data/GLP1_BrCa_cosmx")
    ap.add_argument("--n-cells", type=int, default=5000, help="Random subset size (ignored with --export-all-cells)")
    ap.add_argument("--export-all-cells", action="store_true", help="Export every cell in the table")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--transcript-max", type=int, default=200_000, help="Max transcript rows in 0_0_0.parquet")
    ap.add_argument("--export-transcripts", action="store_true", help="Build transcripts/0_0_0.parquet from cache")
    ap.add_argument("--he-long-edge", type=int, default=8192, help="Morphology PNG long edge (0 = native crop size)")
    ap.add_argument("--morphology-max-long-edge", type=int, default=8192)
    ap.add_argument(
        "--morphology-flip-y",
        action="store_true",
        help="Flip morphology vertically after crop (CosMx global coords usually do not need this)",
    )
    ap.add_argument("--skip-he-morphology", action="store_true", help="Skip OME-Zarr read; write placeholder he.png")
    ap.add_argument(
        "--registration-channel",
        type=str,
        default="DNA",
        help="Registration/reference channel (label/id substring or index; default DNA)",
    )
    ap.add_argument(
        "--pixel-size-um",
        type=float,
        default=0.1203,
        help="Fallback µm/px if missing from sdata .zattrs",
    )
    ap.add_argument(
        "--morphology-crop-to-cells",
        action="store_true",
        help="Crop morphology PNGs to cell bounds (default: export full stitched slide)",
    )
    ap.add_argument(
        "--morphology-only",
        action="store_true",
        help="Only (re)build he.png + Images/ for an existing --out-dir sample",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.morphology_only:
        return run_morphology_only(args)
    return run_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
