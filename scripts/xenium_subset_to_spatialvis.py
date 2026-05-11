#!/usr/bin/env python3
"""
Subsample a 10x Xenium region output directory and write CSO_SpatialVis sample assets
(same file layout as data/toy_small).

Dependencies (see scripts/requirements-xenium-pipeline.txt):
  pip install -r scripts/requirements-xenium-pipeline.txt

Example:
  python scripts/xenium_subset_to_spatialvis.py \\
    --xenium-dir /path/to/output-...__511Tumor__... \\
    --out-dir  /path/to/CSO_SpatialVis/data/xenium_511Tumor_5k \\
    --n-cells 5000

By default, random cells are drawn only from IDs that appear in both ``cells.*`` and the feature
matrix barcodes, so expression rows match centroids/metadata. Use
``--ignore-matrix-barcode-filter`` to sample from all cells in the table (missing matrix IDs get
zeros in expression-wide).

Optional: append the printed manifest sample JSON block to data/manifest.json (version 1.0, samples[]).

For **no** random subsampling (export every cell in the pool), use ``scripts/xenium_to_spatialvis.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from shapely.geometry import mapping, Polygon
from shapely.validation import make_valid

# --- optional heavy deps (import lazily with clear errors) ---


def _require_scipy_h5py():
    import h5py
    from scipy import sparse

    return h5py, sparse


def _require_tifffile():
    import tifffile

    return tifffile


def _find_file(root: Path, names: list[str]) -> Path | None:
    for n in names:
        p = root / n
        if p.exists():
            return p
    return None


def _read_cells(xenium: Path) -> pd.DataFrame:
    p = _find_file(
        xenium,
        [
            "cells.parquet",
            "cells.csv",
            "cells.csv.gz",
        ],
    )
    if p is None:
        raise FileNotFoundError(f"No cells file under {xenium} (expected cells.parquet or cells.csv[.gz])")
    if p.suffix == ".parquet" or p.name.endswith(".parquet"):
        df = pq.read_table(p).to_pandas()
    elif p.suffixes[-2:] == [".csv", ".gz"] or p.suffix == ".gz":
        df = pd.read_csv(p, compression="gzip" if p.suffix == ".gz" or str(p).endswith(".gz") else "infer")
    else:
        df = pd.read_csv(p)

    if "x_centroid" in df.columns and "y_centroid" in df.columns:
        out = df.rename(columns={"x_centroid": "x", "y_centroid": "y"})
    elif "x" in df.columns and "y" in df.columns:
        out = df
    else:
        raise ValueError(f"Unrecognized cell coordinate columns. Columns: {list(df.columns)}")

    if "cell_id" not in out.columns:
        raise ValueError("cells table must have cell_id")
    return out


def _sample_ids(cell_ids: list[str], n: int, seed: int) -> list[str]:
    """Uniform random subset of up to n distinct cell IDs, listed in the same order as the pool.

    (Pool order = first-seen order from `cell_ids` after de-duplication.) The subset of IDs is
    random; their relative order in the result follows the pool so rows stay aligned with tables.
    """
    rng = np.random.default_rng(seed)
    ids = list(dict.fromkeys(str(x) for x in cell_ids))
    n = min(int(n), len(ids))
    if n == 0:
        return []
    if n == len(ids):
        return list(ids)
    pick = np.sort(rng.choice(len(ids), size=n, replace=False))
    return [ids[int(i)] for i in pick]


def _read_feature_matrix_h5(
    h5_path: Path,
) -> tuple[Any, list[str], list[str], np.ndarray]:
    """Return (csc_matrix, barcode_list, feature_names, is_gene_mask)."""
    h5py, sparse = _require_scipy_h5py()
    f = h5py.File(h5_path, "r")
    try:
        g = f["matrix"]
        data = np.asarray(g["data"][:], dtype=np.float64)
        indices = np.asarray(g["indices"][:], dtype=np.int64)
        indptr = np.asarray(g["indptr"][:], dtype=np.int64)
        shape = tuple(int(x) for x in g["shape"][:])
        mat = sparse.csc_matrix((data, indices, indptr), shape=shape, dtype=np.float64)

        raw_b = g["barcodes"][:]
        barcodes = [
            b.decode() if isinstance(b, (bytes, bytearray)) else str(b) for b in raw_b
        ]

        fgrp = g["features"]
        names = fgrp["name"][:]
        names = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in names]
        if "feature_type" in fgrp:
            ftypes = fgrp["feature_type"][:]
            ftypes = [t.decode() if isinstance(t, (bytes, bytearray)) else str(t) for t in ftypes]
            gene_mask = np.array([str(t) == "Gene Expression" for t in ftypes], dtype=bool)
        else:
            gene_mask = np.ones(len(names), dtype=bool)
    finally:
        f.close()
    return mat, barcodes, names, gene_mask


def _read_feature_matrix_mex(mex_dir: Path) -> tuple[Any, list[str], list[str], np.ndarray]:
    h5py, sparse = _require_scipy_h5py()
    from scipy.io import mmread

    mpath = mex_dir / "matrix.mtx.gz"
    if not mpath.exists():
        mpath = mex_dir / "matrix.mtx"
    if not mpath.exists():
        raise FileNotFoundError(f"No MEX matrix in {mex_dir}")
    M = mmread(mpath).tocsr()

    fpath = mex_dir / "features.tsv.gz"
    if not fpath.exists():
        fpath = mex_dir / "features.tsv"
    feat = pd.read_csv(fpath, sep="\t", header=None)
    names = feat.iloc[:, 1].astype(str).tolist()
    if feat.shape[1] >= 3:
        ftypes = feat.iloc[:, 2].astype(str)
        gene_mask = (ftypes == "Gene Expression").to_numpy()
    else:
        gene_mask = np.ones(len(names), dtype=bool)

    bpath = mex_dir / "barcodes.tsv.gz"
    if not bpath.exists():
        bpath = mex_dir / "barcodes.tsv"
    bdf = pd.read_csv(bpath, sep="\t", header=None, names=["cell_id"])
    barcodes = bdf["cell_id"].astype(str).tolist()
    if M.shape[0] == len(barcodes) and M.shape[1] == len(names):
        mat = M.T.tocsc()
    elif M.shape[0] == len(names) and M.shape[1] == len(barcodes):
        mat = M.tocsc()
    else:
        raise ValueError(
            f"MEX shape {M.shape} vs barcodes {len(barcodes)} features {len(names)} — unexpected layout"
        )
    return mat, barcodes, names, gene_mask


def _find_feature_matrix_h5(xenium: Path) -> Path | None:
    p = _find_file(
        xenium,
        [
            "cell_feature_matrix.h5",
            "cell_feature_matrix/cell_feature_matrix.h5",
        ],
    )
    if p is None:
        p = xenium / "cell_feature_matrix" / "cell_feature_matrix.h5"
    return p if p.exists() else None


def _read_barcodes_h5_only(h5_path: Path) -> list[str]:
    h5py, _ = _require_scipy_h5py()
    f = h5py.File(h5_path, "r")
    try:
        raw = f["matrix"]["barcodes"][:]
        return [
            b.decode() if isinstance(b, (bytes, bytearray)) else str(b) for b in raw
        ]
    finally:
        f.close()


def _read_barcodes_mex_only(mex_dir: Path) -> list[str] | None:
    bpath = mex_dir / "barcodes.tsv.gz"
    if not bpath.exists():
        bpath = mex_dir / "barcodes.tsv"
    if not bpath.exists():
        return None
    bdf = pd.read_csv(bpath, sep="\t", header=None, names=["cell_id"])
    return bdf["cell_id"].astype(str).tolist()


def _matrix_barcodes_set(xenium: Path) -> set[str] | None:
    """Barcode set from feature matrix (H5 or MEX) if present, else None."""
    h5p = _find_feature_matrix_h5(xenium)
    if h5p is not None:
        return set(_read_barcodes_h5_only(h5p))
    mex_dir = xenium / "cell_feature_matrix"
    if (mex_dir / "matrix.mtx.gz").exists() or (mex_dir / "matrix.mtx").exists():
        b = _read_barcodes_mex_only(mex_dir)
        if b is not None:
            return set(b)
    return None


def _expression_wide_from_h5(
    h5_path: Path,
    cell_id_order: Sequence[str],
) -> pd.DataFrame:
    mat, barcodes, names, gene_mask = _read_feature_matrix_h5(h5_path)
    col = {b: j for j, b in enumerate(barcodes)}
    use_genes = [i for i, g in enumerate(names) if bool(gene_mask[i]) and str(g).strip()]

    j_idx = [col[c] for c in cell_id_order if c in col]
    ids = [c for c in cell_id_order if c in col]
    if not j_idx:
        return pd.DataFrame({"cell_id": list(cell_id_order)})

    sub = mat[use_genes, :][:, j_idx]
    d = sub.toarray()
    gnames = [str(names[i]) for i in use_genes]
    arr = d.T
    dfd: dict[str, Any] = {"cell_id": ids}
    for k, gn in enumerate(gnames):
        dfd[gn] = np.log1p(np.maximum(arr[:, k], 0.0))
    out = pd.DataFrame(dfd)
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def _expression_wide_from_mex(
    mex_dir: Path,
    cell_id_order: Sequence[str],
) -> pd.DataFrame:
    mat, barcodes, names, gene_mask = _read_feature_matrix_mex(mex_dir)
    return _mat_to_wide_df(mat, barcodes, names, gene_mask, cell_id_order)


def _mat_to_wide_df(mat, barcodes, names, gene_mask, cell_id_order: Sequence[str]):
    col = {b: j for j, b in enumerate(barcodes)}
    use_genes = [i for i, g in enumerate(names) if bool(gene_mask[i]) and str(g).strip()]

    j_idx = [col[c] for c in cell_id_order if c in col]
    ids = [c for c in cell_id_order if c in col]
    if not j_idx:
        return pd.DataFrame({"cell_id": list(cell_id_order)})

    sub = mat[use_genes, :][:, j_idx]
    d = sub.toarray()
    gnames = [str(names[i]) for i in use_genes]
    arr = d.T
    dfd: dict[str, Any] = {"cell_id": ids}
    for k, gn in enumerate(gnames):
        dfd[gn] = np.log1p(np.maximum(arr[:, k], 0.0))
    out = pd.DataFrame(dfd)
    return out.loc[:, ~out.columns.duplicated()]


def _read_boundaries(xenium: Path) -> pd.DataFrame:
    p = _find_file(
        xenium,
        [
            "cell_boundaries.parquet",
            "cell_boundaries.csv",
            "cell_boundaries.csv.gz",
        ],
    )
    if p is None:
        raise FileNotFoundError(f"No cell_boundaries* under {xenium}")
    if p.suffix == ".parquet" or p.name.endswith(".parquet"):
        return pq.read_table(p).to_pandas()
    if str(p).endswith(".gz"):
        return pd.read_csv(p, compression="gzip")
    return pd.read_csv(p)


def _boundaries_to_polygons(bdf: pd.DataFrame, cell_ids: set[str]) -> dict[str, Polygon]:
    need = {
        c
        for c in ("cell_id", "vertex_x", "vertex_y")
        if c in bdf.columns
    }
    if need != {"cell_id", "vertex_x", "vertex_y"}:
        raise ValueError("cell_boundaries need cell_id, vertex_x, vertex_y")

    sub = bdf[bdf["cell_id"].astype(str).isin(cell_ids)]
    by: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for _, r in sub.iterrows():
        by[str(r["cell_id"])].append((float(r["vertex_x"]), float(r["vertex_y"])))

    out: dict[str, Polygon] = {}
    for cid, pts in by.items():
        if len(pts) < 3:
            continue
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.geom_type == "Polygon":
            out[cid] = poly
        elif poly.geom_type == "MultiPolygon" and len(poly.geoms) > 0:
            out[cid] = max(poly.geoms, key=lambda p: p.area)
    return out


def _write_poly_lods(
    polys: dict[str, Polygon],
    cell_centroids: dict[str, tuple[float, float]],
    out_dir: Path,
) -> None:
    lod_tolerances = (0.0, 1.15, 3.2)
    for lod, tol in enumerate(lod_tolerances):
        rows = []
        for cid, p0 in polys.items():
            poly = p0
            if tol > 0 and not poly.is_empty:
                try:
                    sp = poly.simplify(tol, preserve_topology=True)
                    if not sp.is_empty and sp.area > 1e-9:
                        poly = sp
                except Exception:
                    pass
            gj = json.dumps(mapping(poly))
            bx = poly.bounds  # (minx, miny, maxx, maxy)
            rows.append(
                {
                    "id": f"cell_poly_{cid}",
                    "geometry_json": gj,
                    "cell_id": cid,
                    "min_x": bx[0],
                    "min_y": bx[1],
                    "max_x": bx[2],
                    "max_y": bx[3],
                }
            )
        for cid, (cx, cy) in cell_centroids.items():
            if cid in polys:
                continue
            r = 3.0
            poly = Polygon(
                [
                    (cx - r, cy - r),
                    (cx + r, cy - r),
                    (cx + r, cy + r),
                    (cx - r, cy + r),
                ]
            )
            bx = poly.bounds
            rows.append(
                {
                    "id": f"cell_poly_{cid}",
                    "geometry_json": json.dumps(mapping(poly)),
                    "cell_id": cid,
                    "min_x": bx[0],
                    "min_y": bx[1],
                    "max_x": bx[2],
                    "max_y": bx[3],
                }
            )
        pd.DataFrame(rows).to_parquet(out_dir / f"polygons_lod{lod}.parquet", index=False)


def _read_transcripts(xenium: Path) -> pd.DataFrame:
    p = _find_file(
        xenium,
        [
            "transcripts.parquet",
            "transcripts.csv",
            "transcripts.csv.gz",
        ],
    )
    if p is None:
        raise FileNotFoundError("No transcripts.parquet or transcripts.csv[.gz]")
    if p.suffix == ".parquet" or p.name.endswith(".parquet"):
        return pq.read_table(p).to_pandas()
    if str(p).endswith(".gz"):
        return pd.read_csv(p, compression="gzip")
    return pd.read_csv(p)


def _transcripts_to_tile(
    tdf: pd.DataFrame, wanted_ids: set[str], max_rows: int, seed: int
) -> pd.DataFrame:
    col_x = "x_location" if "x_location" in tdf.columns else "x"
    col_y = "y_location" if "y_location" in tdf.columns else "y"
    gcol = "feature_name" if "feature_name" in tdf.columns else "gene_id"
    tdf = tdf[tdf["cell_id"].astype(str).isin(wanted_ids)]
    if tdf.empty:
        return pd.DataFrame({"x": [], "y": [], "gene_id": []})
    if len(tdf) > max_rows:
        tdf = tdf.sample(n=max_rows, random_state=seed)
    return pd.DataFrame(
        {
            "x": tdf[col_x].astype(float).to_numpy(),
            "y": tdf[col_y].astype(float).to_numpy(),
            "gene_id": tdf[gcol].astype(str).to_numpy(),
        }
    )


def _find_experiment(xenium: Path) -> dict[str, Any] | None:
    p = xenium / "experiment.xenium"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _find_secondary_paths(xenium: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    a = xenium / "analysis"
    if not a.is_dir():
        return out
    for p in a.rglob("*.csv"):
        s = p.name.lower()
        if s == "clusters.csv" and "umap" not in str(p).lower():
            if "graphclust" in str(p).lower() or "clustering" in str(p) or "gene_expression" in str(
                p
            ).lower():
                out.setdefault("clusters", p)
        if "umap" in s or "projection" in s and "umap" in str(p.parent).lower():
            out.setdefault("umap", p)
    for p in a.rglob("*.csv"):
        if "umap" in p.parts and p.suffix == ".csv" and p.name not in out.values():
            out.setdefault("umap_any", p)
    return out


def _load_clusters_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_umap_csv(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    ucol = vcol = None
    for k in ("umap-1", "umap1", "u"):
        if k in cols:
            ucol = cols[k]
    for k in ("umap-2", "umap2", "v"):
        if k in cols:
            vcol = cols[k]
    if ucol and vcol:
        return df
    n = [c for c in df.columns if c != "Barcodes" and c != "barcode" and c != "cell_id"]
    if len(n) >= 2:
        ucol, vcol = n[0], n[1]
        return df
    return None


def _load_umap_points(
    xenium: Path, wanted: list[str]
) -> tuple[list[dict], str | None]:
    wset = set(wanted)
    paths = _find_secondary_paths(xenium)
    umap_p = paths.get("umap") or paths.get("umap_any")
    cluster_p = paths.get("clusters")
    cl_map: dict[str, Any] = {}
    if cluster_p and cluster_p.exists():
        cldf = _load_clusters_csv(cluster_p)
        idcol = "Barcode" if "Barcode" in cldf.columns else ("cell_id" if "cell_id" in cldf.columns else cldf.columns[0])
        ccol = "Cluster" if "Cluster" in cldf.columns else [c for c in cldf.columns if c != idcol][-1]
        for _, r in cldf.iterrows():
            cl_map[str(r[idcol])] = r[ccol]

    label_col = "cluster"
    if umap_p and umap_p.exists():
        udf = _load_umap_csv(umap_p)
        if udf is not None:
            cols = {c.lower(): c for c in udf.columns}
            idc = "cell_id" if "cell_id" in udf.columns else "Barcode" if "Barcode" in udf.columns else list(udf.columns)[0]
            ucol = next((c for c in udf.columns if c.lower() in ("umap-1", "umap1", "u")), udf.columns[1])
            vcol = next((c for c in udf.columns if c.lower() in ("umap-2", "umap2", "v")), udf.columns[2])
            pts: list[dict] = []
            for _, r in udf.iterrows():
                cid = str(r[idc])
                if cid not in wset:
                    continue
                lab = cl_map.get(cid)
                d = {
                    "cell_id": cid,
                    "u": float(r[ucol]),
                    "v": float(r[vcol]),
                }
                if lab is not None:
                    d[label_col] = lab
                pts.append(d)
            if pts:
                return pts, label_col
    return [], None


def _synthetic_umap(wanted: list[str], meta_label: str | None, meta_df: pd.DataFrame) -> tuple[list[dict], str | None]:
    rng = np.random.default_rng(42)
    col = "cluster" if "cluster" in meta_df.columns else None
    pts = []
    for i, cid in enumerate(wanted):
        th = 2 * math.pi * rng.random()
        rr = rng.gamma(2.0, 1.2)
        d = {"cell_id": cid, "u": float(rr * math.cos(th)), "v": float(rr * math.sin(th))}
        if col and cid in set(meta_df["cell_id"].astype(str)):
            row = meta_df[meta_df["cell_id"].astype(str) == cid]
            if not row.empty and col in row.columns:
                d[col] = row[col].iloc[0]
        pts.append(d)
    return pts, (col or meta_label)


def _json_sanitize(obj: Any) -> Any:
    """Make objects json-serializable (numpy scalars, pandas NA)."""
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if obj is None or (isinstance(obj, float) and np.isnan(obj)):
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def _pixel_um_from_experiment(exp: dict[str, Any] | None) -> float:
    if not exp:
        return 0.2125
    try:
        if "pixel_size" in exp:
            return float(exp["pixel_size"])
    except (TypeError, ValueError):
        pass
    return 0.2125


def _slug_channel_filename(label: str, idx: int) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip()).strip("_").lower()
    if not s:
        s = f"ch{idx:02d}"
    return s[:80]


def _stable_channel_id(label: str, idx: int, used: set[str]) -> str:
    base = _slug_channel_filename(label, idx)
    cid = base
    j = 0
    while cid in used:
        j += 1
        cid = f"{base}_{j}"
    used.add(cid)
    return cid


def _parse_ome_channel_names(tif_path: Path, n: int) -> list[str]:
    """Best-effort Channel Name=\"…\" from embedded OME XML."""
    out = [f"channel_{i:02d}" for i in range(max(0, n))]
    if n <= 0:
        return []
    try:
        from tifffile import TiffFile

        with TiffFile(str(tif_path)) as tf:
            desc = None
            for page in tf.pages[:16]:
                d = page.description
                if d and isinstance(d, str) and ("OME" in d or "Channel" in d):
                    desc = d
                    break
            if not desc:
                return out
        names = re.findall(r'Channel\s+[^>]*?Name="([^"]*)"', desc)
        if len(names) < n:
            names.extend(re.findall(r"Channel\s+[^>]*?Name='([^']*)'", desc))
        for i in range(min(n, len(names))):
            nm = str(names[i]).strip()
            if nm:
                out[i] = nm
    except Exception:
        pass
    return out


def _squeeze_extra_dims(vol: np.ndarray) -> np.ndarray:
    v = np.asarray(vol, dtype=np.float32)
    while v.ndim > 3 and v.shape[0] == 1:
        v = v[0]
    return v


def _volume_to_channel_planes(vol: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    """Legacy shape-only split when OME axes are unavailable (prefer ``_extract_morphology_planes``)."""
    v = _squeeze_extra_dims(vol)
    if v.ndim == 2:
        return [v], ["morphology"]
    if v.ndim != 3:
        raise ValueError(f"Unexpected morphology ndim={v.ndim} shape={getattr(v, 'shape', None)}")

    h, w = int(v.shape[-2]), int(v.shape[-1])
    # Channel-first CYX (OME multi-channel)
    if v.shape[0] <= 24 and v.shape[0] < min(h, w) // 4:
        planes = [v[i] for i in range(v.shape[0])]
        return planes, [f"channel_{i:02d}" for i in range(len(planes))]
    # YXC (RGB / rgba)
    if v.shape[-1] in (1, 3, 4) and v.shape[-1] <= 4:
        if v.shape[-1] == 1:
            return [v[:, :, 0]], ["morphology"]
        lbls = ["red", "green", "blue", "alpha"]
        planes = [v[:, :, k] for k in range(v.shape[-1])]
        return planes, lbls[: len(planes)]

    return [np.mean(v, axis=0)], ["morphology"]


def _squeeze_leading_ones(arr: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    """Drop leading length-1 dimensions and the matching prefix of ``axes``."""
    v = np.asarray(arr, dtype=np.float32)
    ax = axes.upper()
    while v.ndim >= 1 and len(ax) > 0 and v.shape[0] == 1:
        v = v[0]
        ax = ax[1:]
    return v, ax


def _transpose_to_suffix_yx(vol: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    """Ensure last two dims are Y, X."""
    ax = axes.upper()
    if len(ax) != vol.ndim:
        return vol, ax
    if not ax.endswith("YX"):
        if "Y" in ax and "X" in ax:
            iy, ix = ax.rindex("Y"), ax.rindex("X")
            if iy > ix:
                perm = [i for i in range(vol.ndim) if i not in (iy, ix)] + [iy, ix]
                ax_perm = "".join(ax[i] for i in perm)
                return np.transpose(vol, perm), ax_perm
    return vol, ax


def _z_project_along_axis(vol: np.ndarray, axis: int, how: str) -> np.ndarray:
    if how == "mean":
        return np.mean(vol, axis=axis, dtype=np.float32)
    if how == "max":
        return np.max(vol, axis=axis)
    if how == "middle":
        iz = int(vol.shape[axis] // 2)
        return np.take(vol, iz, axis=axis)
    raise ValueError(f"Unknown z_projection {how!r} (use max, mean, middle)")


def _extract_morphology_planes(
    raw_vol: np.ndarray,
    axes: str | None,
    z_projection: str,
    tif_path: Path,
) -> tuple[list[np.ndarray], list[str], dict[str, Any]]:
    """Use OME axis order to distinguish Z-stacks vs true multi-channel CYX data."""
    note: dict[str, Any] = {"z_projection": None, "axes_reported": axes}
    v = np.asarray(raw_vol, dtype=np.float32)

    if not axes or len(axes) != v.ndim:
        v = _squeeze_extra_dims(v)
        planes, stub = _volume_to_channel_planes(v)
        note["axes_fallback"] = True
        ome = _parse_ome_channel_names(tif_path, len(planes))
        return planes, _merge_labels_with_ome(stub, ome), note

    v, ax = _squeeze_leading_ones(v, axes)
    v, ax = _transpose_to_suffix_yx(v, ax)
    note["axes_used"] = ax

    if v.ndim == 2:
        nm = _parse_ome_channel_names(tif_path, 1)
        return [v], [nm[0] if nm else "morphology"], note

    if not ax.endswith("YX"):
        planes, stub = _volume_to_channel_planes(v)
        note["axes_fallback"] = True
        ome = _parse_ome_channel_names(tif_path, len(planes))
        return planes, _merge_labels_with_ome(stub, ome), note

    prefix = ax[:-2]
    if not prefix:
        nm = _parse_ome_channel_names(tif_path, 1)
        return [v], [nm[0] if nm else "morphology"], note

    # ZYX — single-channel Z-stack (common Xenium morphology_focus)
    if prefix == "Z":
        note["z_projection"] = z_projection
        out = _z_project_along_axis(v, 0, z_projection)
        nm = _parse_ome_channel_names(tif_path, 1)
        lab = next((n for n in nm if n and n != "channel_00"), None) or "dapi"
        return [out], [lab], note

    # CYX — multi-channel, no Z
    if prefix == "C":
        planes = [np.asarray(v[i], dtype=np.float32) for i in range(v.shape[0])]
        stub = [f"channel_{i:02d}" for i in range(len(planes))]
        ome = _parse_ome_channel_names(tif_path, len(planes))
        return planes, _merge_labels_with_ome(stub, ome), note

    # ZCYX — multiple channels, each with Z stack
    if prefix == "ZC":
        note["z_projection"] = z_projection
        n_z, n_c = int(v.shape[0]), int(v.shape[1])
        planes = [_z_project_along_axis(np.asarray(v[:, i, :, :], dtype=np.float32), 0, z_projection) for i in range(n_c)]
        stub = [f"channel_{i:02d}" for i in range(n_c)]
        ome = _parse_ome_channel_names(tif_path, len(planes))
        return planes, _merge_labels_with_ome(stub, ome), note

    # CZYX — channel, then Z
    if prefix == "CZ":
        note["z_projection"] = z_projection
        n_c = int(v.shape[0])
        planes = [_z_project_along_axis(np.asarray(v[i, :, :, :], dtype=np.float32), 0, z_projection) for i in range(n_c)]
        stub = [f"channel_{i:02d}" for i in range(n_c)]
        ome = _parse_ome_channel_names(tif_path, len(planes))
        return planes, _merge_labels_with_ome(stub, ome), note

    # Unknown prefix (e.g. multi-T); collapse non-YX dims by mean
    note["axes_fallback"] = prefix
    while v.ndim > 2:
        v = np.mean(v, axis=0, dtype=np.float32)
    return [v], ["morphology"], note


def _morphology_series_axes(tif_path: Path) -> str | None:
    tifffile = _require_tifffile()
    try:
        with tifffile.TiffFile(str(tif_path)) as tf:
            if not tf.series:
                return None
            ser = tf.series[0]
            ax = getattr(ser, "axes", None)
            return str(ax) if ax else None
    except Exception:
        return None


def _morphology_level_count(tif_path: Path) -> int:
    tifffile = _require_tifffile()
    try:
        with tifffile.TiffFile(str(tif_path)) as tf:
            if not tf.series:
                return 1
            ser = tf.series[0]
            levs = getattr(ser, "levels", None)
            if levs is not None:
                return max(1, len(levs))
    except Exception:
        pass
    return 1


def _imread_morphology_level(tif_path: Path, level: int) -> np.ndarray:
    tifffile = _require_tifffile()
    return np.asarray(tifffile.imread(str(tif_path), is_ome=True, level=int(level)), dtype=np.float32)


def _pick_morphology_level(
    tif_path: Path,
    morphology_level: int,
    long_edge: int,
    morphology_max_long_edge: int,
) -> int:
    """Choose pyramid level: explicit index, or coarsest acceptable then finest that fits heuristic."""
    nlev = _morphology_level_count(tif_path)
    max_idx = max(0, nlev - 1)
    if morphology_level >= 0:
        return int(min(morphology_level, max_idx))
    heuristic_edge = morphology_max_long_edge * 12 if long_edge <= 0 else max(int(long_edge) * 12, 1600 * 12)
    chosen = 0
    for level in range(max_idx, -1, -1):
        try:
            raw = _imread_morphology_level(tif_path, level)
            raw = _squeeze_extra_dims(np.asarray(raw, dtype=np.float32))
            if raw.ndim < 2:
                continue
            big = max(int(raw.shape[-1]), int(raw.shape[-2]))
            chosen = level
            if big < heuristic_edge:
                break
        except Exception:
            continue
    return int(chosen)


def _normalize_fullplane_uint8(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float32)
    x -= float(np.min(x))
    den = float(np.max(x) + 1e-9)
    x /= den
    return (x * 255.0).astype(np.uint8)


def _extract_morphology_crop_u8(
    u8_gray: np.ndarray,
    bounds: dict[str, float],
    pixel_um: float,
    morphology_flip_y: bool,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Crop full-plane morphology to micron ROI; same geometry as PNG export."""
    h0, w0 = int(u8_gray.shape[0]), int(u8_gray.shape[1])

    def um_to_ix_iy(x_um: float, y_um: float) -> tuple[int, int]:
        xi = int(np.clip((x_um / pixel_um), 0, w0 - 1))
        yi = int(np.clip((y_um / pixel_um), 0, h0 - 1))
        return xi, yi

    x0, y0 = um_to_ix_iy(bounds["min_x"], bounds["min_y"])
    x1, y1 = um_to_ix_iy(bounds["max_x"], bounds["max_y"])
    xa, xb = sorted([x0, x1])
    ya, yb = sorted([y0, y1])
    pad = 8
    xa = max(0, xa - pad)
    xb = min(w0, xb + pad)
    ya = max(0, ya - pad)
    yb = min(h0, yb + pad)
    if xb - xa < 2 or yb - ya < 2:
        return None, {}
    crop = u8_gray[ya:yb, xa:xb].copy()
    if crop.size == 0:
        return None, {}
    if morphology_flip_y:
        crop = np.ascontiguousarray(np.flipud(crop))
    ch, cw = int(crop.shape[0]), int(crop.shape[1])
    geo = {
        "xa": int(xa),
        "ya": int(ya),
        "xb": int(xb),
        "yb": int(yb),
        "crop_w": cw,
        "crop_h": ch,
        "full_w": w0,
        "full_h": h0,
        "pixel_um": float(pixel_um),
    }
    return crop, geo


def _save_u8_crop_to_png(
    crop: np.ndarray,
    long_edge: int,
    out_png: Path,
    morphology_max_long_edge: int,
) -> tuple[bool, dict[str, Any]]:
    """Resize (if needed) and save grayscale crop as RGB PNG."""
    from PIL import Image

    ch, cw = int(crop.shape[0]), int(crop.shape[1])
    cap = max(512, int(morphology_max_long_edge))
    capped = False
    if long_edge <= 0:
        new_w, new_h = cw, ch
    else:
        le = int(long_edge)
        if cw >= ch:
            new_w, new_h = le, max(1, int(round(le * ch / max(1, cw))))
        else:
            new_h, new_w = le, max(1, int(round(le * cw / max(1, ch))))

    m = max(new_w, new_h)
    if m > cap:
        capped = True
        if new_w >= new_h:
            new_w, new_h = cap, max(1, int(round(new_h * cap / max(1, new_w))))
        else:
            new_h, new_w = cap, max(1, int(round(new_w * cap / max(1, new_h))))

    im = Image.fromarray(crop, mode="L")
    if (new_w, new_h) != (cw, ch):
        im = im.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    im.convert("RGB").save(out_png, format="PNG", optimize=True, compress_level=9)
    meta = {
        "crop_w": cw,
        "crop_h": ch,
        "out_w": new_w,
        "out_h": new_h,
        "native_resize": bool(long_edge <= 0),
        "capped_to_max_long_edge": capped,
        "morphology_max_long_edge": cap,
    }
    return True, meta


def _crop_microns_dict(geo: dict[str, Any]) -> dict[str, float]:
    xa = float(geo["xa"])
    ya = float(geo["ya"])
    cw = float(geo["crop_w"])
    ch = float(geo["crop_h"])
    px = float(geo["pixel_um"])
    return {
        "min_x": xa * px,
        "min_y": ya * px,
        "max_x": (xa + cw) * px,
        "max_y": (ya + ch) * px,
    }


def _crop_resize_plane_to_png(
    u8_gray: np.ndarray,
    bounds: dict[str, float],
    pixel_um: float,
    long_edge: int,
    out_png: Path,
    *,
    morphology_flip_y: bool = False,
    morphology_max_long_edge: int = 8192,
) -> tuple[bool, dict[str, Any]]:
    """Crop to micron-aligned ROI (same mapping as cell coords), resize, save RGB PNG."""
    crop, geo = _extract_morphology_crop_u8(u8_gray, bounds, pixel_um, morphology_flip_y)
    if crop is None:
        return False, {}
    ok, meta = _save_u8_crop_to_png(crop, long_edge, out_png, morphology_max_long_edge)
    meta.update({k: geo[k] for k in ("xa", "ya", "crop_w", "crop_h", "pixel_um") if k in geo})
    return ok, meta


def _merge_labels_with_ome(stub: list[str], ome: list[str]) -> list[str]:
    out = []
    for i in range(len(stub)):
        o = ome[i] if i < len(ome) else stub[i]
        s = stub[i]
        if o != f"channel_{i:02d}":
            out.append(o)
        else:
            out.append(s)
    return out


def _default_visible_channel(label: str, idx: int, all_labels: list[str]) -> bool:
    if len(all_labels) <= 1:
        return True
    low = label.lower()
    if "dapi" in low:
        return True
    # If nothing named DAPI, show first channel only by default (often nuclear / reference).
    has_dapi_name = any("dapi" in str(x).lower() for x in all_labels)
    return idx == 0 and not has_dapi_name


def _pick_registration_channel_index(
    spec: str,
    manifest_channels: list[dict[str, Any]],
    labels: list[str],
) -> int:
    """Pick morphology channel index for ``Images/registration_reference.png``. Default ``dapi`` = first DAPI-like label/id."""
    if not manifest_channels:
        return 0
    s = (spec or "dapi").strip().lower()
    if not s:
        s = "dapi"
    if s.isdigit():
        i = int(s)
        if 0 <= i < len(manifest_channels):
            return i
    # exact id
    for i, ch in enumerate(manifest_channels):
        cid = str(ch.get("id", "")).strip().lower()
        if cid == s:
            return i
    # exact label
    for i, lab in enumerate(labels):
        if str(lab).strip().lower() == s:
            return i
    if s == "dapi":
        for i, lab in enumerate(labels):
            if "dapi" in str(lab).lower():
                return i
        for i, ch in enumerate(manifest_channels):
            if "dapi" in str(ch.get("id", "")).lower():
                return i
    # substring on label or id
    for i, lab in enumerate(labels):
        if s in str(lab).lower():
            return i
    for i, ch in enumerate(manifest_channels):
        if s in str(ch.get("id", "")).lower():
            return i
    return 0


_CH_FOCUS_STACK_RE = re.compile(r"^ch(\d{4})_(.+)\.ome\.tif$", re.I)


def _list_morphology_focus_channel_stacks(focus_dir: Path) -> list[Path]:
    """Ordered ``ch0000_*.ome.tif`` … files under ``morphology_focus/`` (per-channel Z-stacks)."""
    if not focus_dir.is_dir():
        return []
    rows: list[tuple[int, Path]] = []
    for p in focus_dir.iterdir():
        if not p.is_file():
            continue
        m = _CH_FOCUS_STACK_RE.match(p.name)
        if not m:
            continue
        rows.append((int(m.group(1)), p))
    rows.sort(key=lambda t: t[0])
    return [p for _, p in rows]


def _label_from_ch_focus_filename(path: Path) -> str:
    m = _CH_FOCUS_STACK_RE.match(path.name)
    if not m:
        return path.stem
    stem = m.group(2).strip()
    return stem.replace("_", " ") if stem else path.stem


def _resize_plane_float_to_hw(plane: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a single morphology plane to match the reference (multi-file stacks must align in pixels)."""
    from scipy.ndimage import zoom

    Ht, Wt = target_hw
    h, w = int(plane.shape[0]), int(plane.shape[1])
    if h == Ht and w == Wt:
        return np.asarray(plane, dtype=np.float32)
    zh = Ht / float(h)
    zw = Wt / float(w)
    out = zoom(np.asarray(plane, dtype=np.float32), (zh, zw), order=1)
    out = np.asarray(out, dtype=np.float32)
    out = out[:Ht, :Wt]
    if out.shape[0] < Ht or out.shape[1] < Wt:
        pad = np.zeros((Ht, Wt), dtype=np.float32)
        pad[: out.shape[0], : out.shape[1]] = out
        out = pad
    return out


def _resolve_morphology_input(xenium: Path) -> tuple[list[Path], Path | None]:
    """Prefer multiple ``ch????_*.ome.tif`` under ``morphology_focus/``; else legacy single TIFF."""
    focus = xenium / "morphology_focus"
    ch_stacks = _list_morphology_focus_channel_stacks(focus)
    if len(ch_stacks) >= 2:
        return ch_stacks, None

    for rel in (
        "morphology_focus/morphology_focus_0000.ome.tif",
        "morphology_mip.ome.tif",
        "morphology.ome.tif",
    ):
        cand = xenium / rel
        if cand.is_file():
            return [], cand

    if focus.is_dir():
        all_tifs = sorted(focus.glob("*.ome.tif"))
        if all_tifs:
            return [], all_tifs[0]

    return [], None


def _fuse_morphology_planes_to_2d(planes: list[np.ndarray], how: str) -> np.ndarray:
    """Collapse multiple same-size planes (Z slices often mis-tagged as CYX channels) to one image."""
    if not planes:
        raise ValueError("no planes")
    if len(planes) == 1:
        return np.asarray(planes[0], dtype=np.float32)
    stacked = np.stack([np.asarray(p, dtype=np.float32) for p in planes], axis=0)
    h = (how or "max").strip().lower()
    if h == "mean":
        return np.mean(stacked, axis=0, dtype=np.float32)
    if h == "middle":
        iz = int(stacked.shape[0] // 2)
        return stacked[iz]
    return np.max(stacked, axis=0)


def _load_multi_focus_planes(
    paths: list[Path],
    morphology_level: int,
    long_edge: int,
    morphology_max_long_edge: int,
    z_projection: str,
) -> tuple[list[np.ndarray], list[str], dict[str, Any], int]:
    """Load one 2D panel per ``chNNNN_*.ome.tif`` (Z-stack or mis-labeled axes → fused 2D)."""
    ref_path = paths[0]
    level = _pick_morphology_level(ref_path, morphology_level, long_edge, morphology_max_long_edge)
    planes_f: list[np.ndarray] = []
    labels: list[str] = []
    extr_note: dict[str, Any] = {}
    ref_hw: tuple[int, int] | None = None

    for i, p in enumerate(paths):
        try:
            raw_vol = _imread_morphology_level(p, level)
        except Exception as e:
            print(f"Warning: skip morphology file {p.name} ({e})", file=sys.stderr)
            continue
        if raw_vol is None or raw_vol.size == 0:
            continue
        axes_reported = _morphology_series_axes(p)
        try:
            planes, labs, note = _extract_morphology_planes(raw_vol, axes_reported, z_projection, p)
        except ValueError:
            print(f"Warning: could not parse planes from {p.name}", file=sys.stderr)
            continue
        if not planes:
            continue
        if len(planes) > 1:
            print(
                f"Note: {p.name} → {len(planes)} planes (often Z-focus slices mis-labeled as channels); "
                f"fusing with z_projection={z_projection!r}.",
                file=sys.stderr,
            )
        plane_f = _fuse_morphology_planes_to_2d(planes, z_projection)
        if ref_hw is None:
            ref_hw = (int(plane_f.shape[0]), int(plane_f.shape[1]))
        elif plane_f.shape[:2] != ref_hw:
            print(
                f"Warning: {p.name} shape {plane_f.shape[:2]} != reference {ref_hw}; resizing to match.",
                file=sys.stderr,
            )
            plane_f = _resize_plane_float_to_hw(plane_f, ref_hw)

        # Filename encodes the stain (``ch0001_atp1a1_...``); OME often repeats "DAPI" for every plane.
        lab = _label_from_ch_focus_filename(p)

        planes_f.append(plane_f)
        labels.append(lab)
        if i == 0:
            extr_note = dict(note)

    extr_note.setdefault("z_projection", z_projection)
    extr_note["axes_reported"] = "MULTI_FILE"
    extr_note["multi_file_sources"] = [p.name for p in paths]
    return planes_f, labels, extr_note, level


def _write_morphology_bundle(
    xenium: Path,
    bounds: dict[str, float],
    out_dir: Path,
    out_he_png: Path,
    long_edge: int,
    sample_folder_name: str,
    registration_channel_spec: str = "dapi",
    *,
    morphology_max_long_edge: int = 8192,
    morphology_level: int = -1,
    z_projection: str = "max",
    morphology_flip_y: bool = False,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    """Export morphology to ``Images/*.png`` (per channel) + ``he.png`` from segmentation reference channel crop."""
    exp = _find_experiment(xenium)
    pixel_um = _pixel_um_from_experiment(exp)

    multi_paths, single_tif = _resolve_morphology_input(xenium)
    axes_reported: str | None
    extr_note: dict[str, Any]
    level: int

    if multi_paths:
        print(
            f"Morphology: exporting {len(multi_paths)} channels from morphology_focus/ "
            f"({', '.join(p.name for p in multi_paths)})",
            file=sys.stderr,
        )
        planes_f, labels, extr_note, level = _load_multi_focus_planes(
            multi_paths,
            morphology_level,
            long_edge,
            morphology_max_long_edge,
            z_projection,
        )
        axes_reported = "MULTI_FILE"
        if not planes_f:
            return False, [], {}
    else:
        tif = single_tif
        if tif is None:
            return False, [], {}

        axes_reported = _morphology_series_axes(tif)
        level = _pick_morphology_level(tif, morphology_level, long_edge, morphology_max_long_edge)
        try:
            raw_vol = _imread_morphology_level(tif, level)
        except Exception:
            return False, [], {}

        if raw_vol is None or raw_vol.size == 0:
            return False, [], {}

        try:
            planes_f, labels, extr_note = _extract_morphology_planes(
                raw_vol,
                axes_reported,
                z_projection,
                tif,
            )
        except ValueError:
            return False, [], {}

    MAX_CH = 16
    if len(planes_f) > MAX_CH:
        print(f"Warning: truncating morphology channels from {len(planes_f)} to {MAX_CH}.", file=sys.stderr)
        planes_f = planes_f[:MAX_CH]
        labels = labels[:MAX_CH]

    images_dir = out_dir / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)

    channel_planes: list[tuple[np.ndarray, str, int]] = []
    for i, (plane_f, lab) in enumerate(zip(planes_f, labels, strict=True)):
        u8 = _normalize_fullplane_uint8(plane_f)
        channel_planes.append((u8, lab, i))

    used_for_ids: set[str] = set()
    channel_ids_ordered = [_stable_channel_id(lab, idx, used_for_ids) for _, lab, idx in channel_planes]

    provisional_manifest = [{"id": channel_ids_ordered[i], "label": labels[i]} for i in range(len(labels))]
    reg_labels_list = [str(lab) for lab in labels]
    reg_idx = _pick_registration_channel_index(registration_channel_spec, provisional_manifest, reg_labels_list)
    reg_idx = max(0, min(reg_idx, len(channel_planes) - 1))

    ref_u8 = channel_planes[reg_idx][0]
    crop_ref, geo_ref = _extract_morphology_crop_u8(ref_u8, bounds, pixel_um, morphology_flip_y)

    he_long_eff = int(long_edge)

    png_meta_he: dict[str, Any] = {}
    merged_ok = False
    morphology_frame: dict[str, Any] | None = None

    if crop_ref is not None and geo_ref:
        ok_he, png_meta_he = _save_u8_crop_to_png(crop_ref, he_long_eff, out_he_png, morphology_max_long_edge)
        merged_ok = bool(ok_he)

        seg_cid = channel_ids_ordered[reg_idx]
        morphology_frame = {
            "version": 1,
            "coordinate_unit": "micrometer",
            "convention_note": (
                "cells.parquet x,y are Xenium cell centroids in micrometers (same instrument axes as morphology). "
                "At pyramid_level on the OME-TIFF, column i maps to x_um = i * pixel_um and row j to "
                "y_um = j * pixel_um. crop_origin_ix/iy index into that level; crop_w/crop_h are native pixels "
                "after optional morphology_flip_y. he.png shares this crop with channel PNGs under Images/. "
                "alignment.translate_um in the app shifts morphology only for residual correction."
            ),
            "pixel_um": float(geo_ref["pixel_um"]),
            "full_image_shape_px": [int(geo_ref["full_h"]), int(geo_ref["full_w"])],
            "crop_origin_ix": int(geo_ref["xa"]),
            "crop_origin_iy": int(geo_ref["ya"]),
            "crop_w": int(geo_ref["crop_w"]),
            "crop_h": int(geo_ref["crop_h"]),
            "crop_microns": _crop_microns_dict(geo_ref),
            "morphology_flip_y_export": bool(morphology_flip_y),
            "segmentation_reference_channel_id": seg_cid,
            "segmentation_reference_channel_label": str(labels[reg_idx]),
            "segmentation_reference_channel_index": int(reg_idx),
            "export_micron_bounds": dict(bounds),
        }

    manifest_channels = []

    for u8, lab, idx in channel_planes:
        fname = f"{idx:02d}_{_slug_channel_filename(lab, idx)}.png"
        rel_path = f"Images/{fname}"
        out_png = images_dir / fname
        ok_plane, _ = _crop_resize_plane_to_png(
            u8,
            bounds,
            pixel_um,
            long_edge,
            out_png,
            morphology_flip_y=morphology_flip_y,
            morphology_max_long_edge=morphology_max_long_edge,
        )
        if not ok_plane:
            continue
        cid = channel_ids_ordered[idx]
        dv = _default_visible_channel(lab, idx, labels)
        entry = {
            "id": cid,
            "label": lab,
            "relative_path": rel_path.replace("\\", "/"),
            "url": f"/api/assets/sample_files/{sample_folder_name}/{rel_path.replace(chr(92), '/')}",
            "default_visible": dv,
        }
        manifest_channels.append(entry)

    reg_meta: dict[str, Any] = {}
    if manifest_channels:
        import shutil

        reg_plane_idx = reg_idx
        reg_ch_id = channel_ids_ordered[reg_plane_idx]
        reg_entry = next((e for e in manifest_channels if e["id"] == reg_ch_id), None)
        reg_ch = reg_entry if reg_entry is not None else manifest_channels[0]
        reg_fname = f"{reg_plane_idx:02d}_{_slug_channel_filename(labels[reg_plane_idx], reg_plane_idx)}.png"
        src_reg = images_dir / reg_fname
        dst_reg = images_dir / "registration_reference.png"
        if src_reg.is_file():
            shutil.copyfile(src_reg, dst_reg)
        im_man: dict[str, Any] = {
            "version": 3,
            "micron_bounds": bounds,
            "coordinate_note": (
                "cells.parquet uses Xenium micrometer coordinates on the same axes as morphology "
                "(see morphology_frame for pixel↔µm mapping)."
            ),
            "pixel_um": pixel_um,
            "ome_axes": axes_reported,
            "pyramid_level": level,
            "z_projection": extr_note.get("z_projection"),
            "export_long_edge": long_edge,
            "morphology_max_long_edge": morphology_max_long_edge,
            "morphology_flip_y": bool(morphology_flip_y),
            "he_png_geometry": png_meta_he,
            "channels": [{k: v for k, v in ch.items() if k != "url"} for ch in manifest_channels],
            "registration_channel_spec": registration_channel_spec,
            "registration_channel_id": reg_ch["id"],
            "registration_channel_label": reg_ch["label"],
            "registration_reference_relative_path": "Images/registration_reference.png",
        }
        if morphology_frame:
            im_man["morphology_frame"] = _json_sanitize(morphology_frame)
        (out_dir / "images_manifest.json").write_text(json.dumps(_json_sanitize(im_man), indent=2) + "\n")
        reg_meta = {
            "registration_channel_id": str(reg_ch["id"]),
            "registration_reference_url": f"/api/assets/sample_files/{sample_folder_name}/Images/registration_reference.png",
        }

    if not merged_ok and manifest_channels:
        import shutil

        first_png = out_dir / Path(manifest_channels[0]["relative_path"])
        if first_png.is_file():
            shutil.copyfile(first_png, out_he_png)
            merged_ok = True

    return bool(merged_ok or manifest_channels), manifest_channels, reg_meta


def _write_he_synthetic(bounds: dict[str, float], out_png: Path, long_edge: int) -> None:
    from PIL import Image, ImageDraw

    w = float(bounds["max_x"] - bounds["min_x"] + 1e-6)
    h = float(bounds["max_y"] - bounds["min_y"] + 1e-6)
    le = int(long_edge) if long_edge and int(long_edge) > 0 else 1600
    sx = int(max(1, le * (w / max(w, h))))
    sy = int(max(1, le * (h / max(w, h))))
    img = Image.new("RGB", (sx, sy), (235, 230, 225))
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, sx - 1, sy - 1], outline=(180, 170, 160))
    img.save(out_png, format="PNG", optimize=True, compress_level=9)


def run_export(args: argparse.Namespace) -> int:
    """Shared export logic for subset and full-run CLIs."""
    x = args.xenium_dir.resolve()
    out = args.out_dir.resolve()
    if not x.is_dir():
        print(f"Not a directory: {x}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    (out / "transcripts").mkdir(exist_ok=True)

    # ---- cells: one ordered ID list for every artifact (avoids set-order + matrix mismatch)
    print("Load cells…")
    cdf = _read_cells(x)
    pool_ids = list(dict.fromkeys(cdf["cell_id"].astype(str).tolist()))
    mb = _matrix_barcodes_set(x)
    if mb is not None and not args.ignore_matrix_barcode_filter:
        pool = [c for c in pool_ids if c in mb]
        n_overlap = len(pool)
        n_cells_tab = len(pool_ids)
        if n_cells_tab > 0 and n_overlap < n_cells_tab:
            print(
                f"Sampling pool: {n_overlap} cells in both cells table and feature matrix "
                f"(of {n_cells_tab} in table). Use --ignore-matrix-barcode-filter to include others.",
            )
        if not pool:
            print(
                f"No cell_id in common between cells table and feature matrix under {x}.",
                file=sys.stderr,
            )
            return 1
    else:
        pool = pool_ids
    export_all = bool(getattr(args, "export_all_cells", False))
    if export_all:
        chosen_list = list(pool)
        print(f"Export all cells in pool: {len(chosen_list)} (no random subset).")
    else:
        n_take = int(min(int(args.n_cells), len(pool)))
        chosen_list = _sample_ids(pool, n_take, args.seed)
    chosen_set = set(chosen_list)
    order_key = {cid: i for i, cid in enumerate(chosen_list)}
    sub_cells = cdf[cdf["cell_id"].astype(str).isin(chosen_set)].copy()
    sub_cells["_ord"] = sub_cells["cell_id"].astype(str).map(lambda s: order_key.get(s, -1))
    if (sub_cells["_ord"] < 0).any():
        print("Error: sample IDs missing from cells table (inconsistent data).", file=sys.stderr)
        return 1
    sub_cells = sub_cells.sort_values("_ord").drop(columns=["_ord"])
    sub_cells = sub_cells.rename(columns={c: c for c in sub_cells.columns})
    n = len(chosen_list)
    cells_out = pd.DataFrame(
        {
            "cell_id": sub_cells["cell_id"].astype(str),
            "x": sub_cells["x"].astype(float),
            "y": sub_cells["y"].astype(float),
        }
    )
    cells_out.to_parquet(out / "cells.parquet", index=False)
    cx_map = {str(r["cell_id"]): (float(r["x"]), float(r["y"])) for _, r in cells_out.iterrows()}

    # ---- feature matrix (column order = chosen_list)
    h5p = _find_feature_matrix_h5(x)
    mex_dir = x / "cell_feature_matrix"
    if h5p is not None:
        print("Load cell_feature_matrix.h5…")
        expr = _expression_wide_from_h5(h5p, chosen_list)
    elif (mex_dir / "matrix.mtx.gz").exists() or (mex_dir / "matrix.mtx").exists():
        print("Load cell_feature_matrix MEX…")
        expr = _expression_wide_from_mex(mex_dir, chosen_list)
    else:
        print("No cell_feature_matrix.h5 or MEX; expression-wide will be cell_id only.", file=sys.stderr)
        expr = pd.DataFrame({"cell_id": list(chosen_list)})

    order = cells_out["cell_id"].astype(str).tolist()
    expr = cells_out[["cell_id"]].merge(expr, on="cell_id", how="left")
    gene_cols = [c for c in expr.columns if c != "cell_id"]
    expr[gene_cols] = expr[gene_cols].fillna(0.0)
    expr = expr.set_index("cell_id").reindex(order).reset_index()
    expr.to_parquet(out / "expression-wide.parquet", index=False)

    # ---- metadata from cell summary (numeric + optional cluster)
    keep_meta = [c for c in sub_cells.columns if c not in ("x", "y") and c != "cell_id"]
    meta = sub_cells[["cell_id"] + keep_meta].copy() if keep_meta else sub_cells[["cell_id"]].copy()
    for c in meta.columns:
        if c != "cell_id" and not str(c).startswith("$ManualAnno:"):
            try:
                meta[c] = pd.to_numeric(meta[c], errors="ignore")
            except Exception:
                pass
    if "cluster" not in meta.columns and "total_counts" in meta.columns:
        try:
            meta["cluster"] = pd.cut(
                meta["total_counts"].astype(float), bins=12, labels=False, duplicates="drop"
            ).astype("Int64")
        except Exception:
            pass
    if "cell_type" not in meta.columns:
        meta["cell_type"] = "unknown"
    meta["cell_id"] = meta["cell_id"].astype(str)
    meta.to_parquet(out / "cell_metadata.parquet", index=False)

    # ---- boundaries
    try:
        print("Build polygons…")
        bdf = _read_boundaries(x)
        polys = _boundaries_to_polygons(bdf, chosen_set)
        _write_poly_lods(polys, cx_map, out)
    except FileNotFoundError as e:
        print(f"Warning: {e}; using square footprints.", file=sys.stderr)
        polys = {}
        _write_poly_lods({}, cx_map, out)

    # ---- transcripts (skipped by default; use --export-transcripts)
    if not getattr(args, "export_transcripts", False):
        print("Skipping transcript export (default; pass --export-transcripts to build the tile).")
        pd.DataFrame({"x": [], "y": [], "gene_id": []}).to_parquet(out / "transcripts" / "0_0_0.parquet", index=False)
    else:
        try:
            print("Prepare transcript tile…")
            tdf = _read_transcripts(x)
            tx = _transcripts_to_tile(tdf, chosen_set, args.transcript_max, args.seed)
            tx.to_parquet(out / "transcripts" / "0_0_0.parquet", index=False)
        except FileNotFoundError as e:
            print(f"Warning: {e}", file=sys.stderr)
            pd.DataFrame({"x": [], "y": [], "gene_id": []}).to_parquet(out / "transcripts" / "0_0_0.parquet", index=False)

    # ---- plots: UMAP + composition
    pts, ulabel = _load_umap_points(x, chosen_list)
    if not pts:
        print("UMAP: build synthetic projection (or from metadata).")
        pts, ulabel = _synthetic_umap(chosen_list, None, meta)
    if not pts and chosen_list:
        pts, ulabel = _synthetic_umap(chosen_list, None, meta)

    col_for_comp = ulabel
    if col_for_comp and pts and not all(col_for_comp in p for p in pts):
        col_for_comp = None
    if col_for_comp is None and "cluster" in meta.columns and meta["cluster"].nunique(dropna=True) > 0:
        col_for_comp = "cluster"
        for p in pts:
            cid = p["cell_id"]
            r = meta[meta["cell_id"] == cid]
            if not r.empty and pd.notna(r["cluster"].iloc[0]):
                p["cluster"] = r["cluster"].iloc[0]
    if col_for_comp is None and meta["cell_type"].nunique() > 1:
        for p in pts:
            cid = p["cell_id"]
            r = meta[meta["cell_id"] == cid]
            if not r.empty:
                p["cell_type"] = r["cell_type"].iloc[0]
        col_for_comp = "cell_type"

    comp_labels: list[str] = []
    comp_counts: list[int] = []
    if col_for_comp and pts and all(col_for_comp in p for p in pts):
        vals = [p.get(col_for_comp) for p in pts]
        uq: dict[Any, int] = {}
        for v in vals:
            if pd.isna(v):
                uq["NA"] = uq.get("NA", 0) + 1
            else:
                uq[v] = uq.get(v, 0) + 1
        comp_labels = [str(k) for k in uq]
        comp_counts = [int(uq[k]) for k in uq]
    comp = (
        {"labels": comp_labels, "counts": comp_counts}
        if comp_labels
        else {"labels": [], "counts": []}
    )
    plot_payload = _json_sanitize({"points": pts, "composition": comp})
    (out / "plots" / "umap.json").write_text(json.dumps(plot_payload, indent=2) + "\n")
    (out / "plots" / "composition.json").write_text(json.dumps(_json_sanitize(comp), indent=2) + "\n")

    # ---- pathology tag skeleton (app expects schema when ROI sync runs)
    tag_cols = [
        "cell_id",
        "pathology_region",
        "pathology_region_source",
        "pathology_region_confidence",
    ]
    empty_tags = pd.DataFrame(columns=tag_cols)
    empty_tags.to_parquet(out / "pathology_cell_tags.parquet", index=False)

    # ---- Morphology (DAPI / multi-channel → Images/, composite he.png)
    pad = 40.0
    bminx = float(cells_out["x"].min() - pad)
    bmaxx = float(cells_out["x"].max() + pad)
    bminy = float(cells_out["y"].min() - pad)
    bmaxy = float(cells_out["y"].max() + pad)
    bounds = {"min_x": bminx, "min_y": bminy, "max_x": bmaxx, "max_y": bmaxy}
    morph_chans: list[dict[str, Any]] = []
    reg_spec = str(getattr(args, "registration_channel", "dapi") or "dapi").strip() or "dapi"
    zproj = str(getattr(args, "morphology_z_projection", "max") or "max").strip().lower()
    morph_level = int(getattr(args, "morphology_level", -1))
    morph_cap = int(getattr(args, "morphology_max_long_edge", 8192))
    morph_flip = bool(getattr(args, "morphology_flip_y", False))
    if not args.skip_he_morphology:
        try:
            ok, morph_chans, reg_meta = _write_morphology_bundle(
                x,
                bounds,
                out,
                out / "he.png",
                args.he_long_edge,
                out.name,
                reg_spec,
                morphology_max_long_edge=morph_cap,
                morphology_level=morph_level,
                z_projection=zproj,
                morphology_flip_y=morph_flip,
            )
        except Exception as e:
            print(f"Warning: morphology export failed ({e}); placeholder.", file=sys.stderr)
            ok, morph_chans, reg_meta = False, [], {}
        if not ok:
            _write_he_synthetic(bounds, out / "he.png", min(args.he_long_edge, 1420))
            morph_chans = []
            reg_meta = {}
    else:
        _write_he_synthetic(bounds, out / "he.png", min(args.he_long_edge, 1420))
        reg_meta = {}

    n_genes = 0
    if expr.shape[1] > 1:
        n_genes = int(expr.shape[1] - 1)

    image_obj: dict[str, Any] = {
        "type": "osd_simple",
        "url": f"/api/assets/he/{out.name}",
        "alignment": {"mode": "identity", "bounds": bounds},
    }
    if morph_chans:
        image_obj["channels"] = morph_chans
    if reg_meta.get("registration_channel_id"):
        image_obj["registration_channel_id"] = reg_meta["registration_channel_id"]
    if reg_meta.get("registration_reference_url"):
        image_obj["registration_reference_url"] = reg_meta["registration_reference_url"]

    sample = {
        "sample_id": out.name,
        "title": (f"Xenium ({n} cells)" if export_all else f"Xenium subset ({n} cells)"),
        "description": (
            f"Exported from {x.name} for CSO_SpatialVis (all cells in pool)"
            if export_all
            else f"Subsampled from {x.name} for CSO_SpatialVis"
        ),
        "coordinate_unit": "microns",
        "bounds": bounds,
        "image": image_obj,
        "layers": ["centroids", "segmentation", "transcripts", "he"],
        "genes_available": n_genes,
    }
    (out / "sample_manifest_entry.json").write_text(json.dumps(sample, indent=2) + "\n")

    if morph_chans:
        print(
            f"Morphology: exported {len(morph_chans)} channel PNG(s) → {out / 'Images'} "
            f"(see images_manifest.json). Composite underlay: {out / 'he.png'}",
        )
        if reg_meta.get("registration_channel_id"):
            print(
                f"Registration reference: channel '{reg_meta['registration_channel_id']}' "
                f"→ {out / 'Images' / 'registration_reference.png'} "
                f"(use --registration-channel to change).",
            )

    print("Done:", out)
    print("Wrote", out / "sample_manifest_entry.json", "— merge into data/manifest.json if desired.")
    print("Manifest sample (copy into samples[]):")
    print(json.dumps(sample, indent=2))
    return 0


def parse_args_subset() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xenium-dir", type=Path, required=True, help="Xenium output region directory")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output sample folder, e.g. data/xenium_511Tumor_5k")
    ap.add_argument("--n-cells", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--transcript-max", type=int, default=200_000, help="Max transcript rows in transcripts/0_0_0.parquet")
    ap.add_argument(
        "--he-long-edge",
        type=int,
        default=1600,
        help="Morphology PNG long edge in pixels (0 = native resolution of ROI crop; capped by --morphology-max-long-edge)",
    )
    ap.add_argument(
        "--morphology-max-long-edge",
        type=int,
        default=8192,
        help="Hard cap on morphology PNG long edge (native or resized) for browser/GPU limits",
    )
    ap.add_argument(
        "--morphology-level",
        type=int,
        default=-1,
        help="OME-TIFF pyramid level (0 = finest/full-res; -1 = auto)",
    )
    ap.add_argument(
        "--morphology-z-projection",
        choices=("max", "mean", "middle"),
        default="max",
        help="How to collapse Z when morphology OME-TIFF is a Z-stack (e.g. Xenium DAPI focus planes)",
    )
    ap.add_argument(
        "--morphology-flip-y",
        action="store_true",
        help="Flip morphology vertically after micron crop (use if image y-axis opposes cell coordinates)",
    )
    ap.add_argument(
        "--skip-he-morphology",
        action="store_true",
        help="Do not try to read OME-TIFF; write a placeholder he.png in cropped bounds only",
    )
    ap.add_argument(
        "--export-transcripts",
        action="store_true",
        help="Read and export transcripts/0_0_0.parquet (default: skip; empty tile only)",
    )
    ap.add_argument(
        "--ignore-matrix-barcode-filter",
        action="store_true",
        help="Sample from all cells in cells.parquet, even if some IDs are missing from the "
        "feature matrix (expression shows zeros for missing). Default: when a matrix exists, only "
        "sample cells present in both the cells table and matrix so rows stay consistent.",
    )
    ap.add_argument(
        "--registration-channel",
        type=str,
        default="dapi",
        help="Morphology channel used for Images/registration_reference.png (sidecar registration pipeline). "
        "Match: id, label, substring, or 0-based index. Default: dapi (first channel whose name/id contains 'dapi').",
    )
    args = ap.parse_args()
    args.export_all_cells = False
    return args


def main() -> int:
    return run_export(parse_args_subset())


if __name__ == "__main__":
    raise SystemExit(main())
