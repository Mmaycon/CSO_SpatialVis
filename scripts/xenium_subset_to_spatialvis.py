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


def _write_he_morphology(
    xenium: Path, bounds: dict[str, float], out_png: Path, long_edge: int
) -> bool:
    tifffile = _require_tifffile()
    exp = _find_experiment(xenium)
    pixel_um = _pixel_um_from_experiment(exp)

    tif: Path | None = None
    for rel in (
        "morphology_focus/morphology_focus_0000.ome.tif",
        "morphology_mip.ome.tif",
        "morphology.ome.tif",
    ):
        cand = xenium / rel
        if cand.exists():
            tif = cand
            break
    if tif is None and (xenium / "morphology_focus").is_dir():
        first = sorted((xenium / "morphology_focus").glob("*.ome.tif"))
        tif = first[0] if first else None
    if tif is None:
        return False

    a: np.ndarray | None = None
    for level in (2, 1, 0):
        try:
            a = tifffile.imread(str(tif), is_ome=True, level=level)  # type: ignore[call-arg]
        except (TypeError, Exception):
            try:
                a = tifffile.imread(str(tif), is_ome=False, level=level)
            except Exception:
                a = None
        if a is not None and a.size:
            if max(a.shape[-2:]) < long_edge * 12:
                break
    if a is None or a.size == 0:
        return False

    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:
        if a.shape[0] <= 8 and a.shape[0] < min(a.shape[1], a.shape[2]):
            a = a[0]
        elif a.shape[-1] in (1, 3, 4):
            a = a[:, :, 0] if a.shape[-1] == 1 else a.mean(axis=2)
    if a.ndim != 2:
        return False

    a -= float(np.min(a))
    a /= float(np.max(a) + 1e-9)
    a = (a * 255.0).astype(np.uint8)

    h0, w0 = int(a.shape[0]), int(a.shape[1])

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
        return False
    crop = a[ya:yb, xa:xb]
    if crop.size == 0:
        return False
    from PIL import Image

    ch, cw = int(crop.shape[0]), int(crop.shape[1])
    if cw >= ch:
        new_w, new_h = long_edge, max(1, int(round(long_edge * ch / max(1, cw))))
    else:
        new_h, new_w = long_edge, max(1, int(round(long_edge * cw / max(1, ch))))
    im = Image.fromarray(crop, mode="L")
    im = im.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    im.convert("RGB").save(out_png, format="PNG", optimize=True, compress_level=9)
    return True


def _write_he_synthetic(bounds: dict[str, float], out_png: Path, long_edge: int) -> None:
    from PIL import Image, ImageDraw

    w = float(bounds["max_x"] - bounds["min_x"] + 1e-6)
    h = float(bounds["max_y"] - bounds["min_y"] + 1e-6)
    sx = int(max(1, long_edge * (w / max(w, h))))
    sy = int(max(1, long_edge * (h / max(w, h))))
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

    # ---- transcripts
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

    # ---- H&E
    pad = 40.0
    bminx = float(cells_out["x"].min() - pad)
    bmaxx = float(cells_out["x"].max() + pad)
    bminy = float(cells_out["y"].min() - pad)
    bmaxy = float(cells_out["y"].max() + pad)
    bounds = {"min_x": bminx, "min_y": bminy, "max_x": bmaxx, "max_y": bmaxy}
    if not args.skip_he_morphology:
        try:
            ok = _write_he_morphology(x, bounds, out / "he.png", args.he_long_edge)
        except Exception as e:
            print(f"Warning: H&E from morphology failed ({e}); placeholder.", file=sys.stderr)
            ok = False
        if not ok:
            _write_he_synthetic(bounds, out / "he.png", min(args.he_long_edge, 1420))
    else:
        _write_he_synthetic(bounds, out / "he.png", min(args.he_long_edge, 1420))

    n_genes = 0
    if expr.shape[1] > 1:
        n_genes = int(expr.shape[1] - 1)

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
        "image": {
            "type": "osd_simple",
            "url": f"/api/assets/he/{out.name}",
            "alignment": {"mode": "identity", "bounds": bounds},
        },
        "layers": ["centroids", "segmentation", "transcripts", "he"],
        "genes_available": n_genes,
    }
    (out / "sample_manifest_entry.json").write_text(json.dumps(sample, indent=2) + "\n")

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
    ap.add_argument("--he-long-edge", type=int, default=1600, help="PNG long edge in pixels")
    ap.add_argument(
        "--skip-he-morphology",
        action="store_true",
        help="Do not try to read OME-TIFF; write a placeholder he.png in cropped bounds only",
    )
    ap.add_argument(
        "--ignore-matrix-barcode-filter",
        action="store_true",
        help="Sample from all cells in cells.parquet, even if some IDs are missing from the "
        "feature matrix (expression shows zeros for missing). Default: when a matrix exists, only "
        "sample cells present in both the cells table and matrix so rows stay consistent.",
    )
    args = ap.parse_args()
    args.export_all_cells = False
    return args


def main() -> int:
    return run_export(parse_args_subset())


if __name__ == "__main__":
    raise SystemExit(main())
