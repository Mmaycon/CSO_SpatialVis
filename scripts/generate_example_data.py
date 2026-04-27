#!/usr/bin/env python3
"""
Generate a minimal precomputed sample for CSO_SpatialVis (no analysis — synthetic layout).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from shapely.geometry import mapping, Polygon


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for generated data (default: <repo>/data)",
    )
    p.add_argument(
        "--cells",
        type=int,
        default=12_000,
        help="Number of synthetic cells (default: 12000). Use ~500–2000 for a fast toy run.",
    )
    p.add_argument(
        "--sample-id",
        default="sample_demo",
        help="Sample folder name / manifest id (default: sample_demo).",
    )
    p.add_argument(
        "--toy",
        action="store_true",
        help="Shortcut: ~800 cells, sample id toy_small (good for quick UI checks).",
    )
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    data_root = args.out if args.out is not None else repo / "data"
    sample_id = "toy_small" if args.toy else args.sample_id
    n = 800 if args.toy else max(50, args.cells)
    out = data_root / sample_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    (out / "transcripts").mkdir(exist_ok=True)

    rng = np.random.default_rng(42)

    # Spatial layout: two curved bands + tighter cluster (visual interest)
    t = rng.uniform(0, 4 * math.pi, n)
    rad = rng.normal(180, 35, n)
    x = rad * np.cos(t) + rng.normal(0, 12, n) + 400
    y = rad * np.sin(t) * 0.65 + rng.normal(0, 18, n) + 350

    cluster_mask = (x - 620) ** 2 / 120**2 + (y - 280) ** 2 / 90**2 < 1
    x[cluster_mask] += rng.normal(0, 6, cluster_mask.sum())
    y[cluster_mask] += rng.normal(0, 6, cluster_mask.sum())

    cell_ids = np.array([f"C{i:06d}" for i in range(n)])

    types = np.where(cluster_mask, "immune_rich", "stromal")
    types = np.where((x < 480) & (y > 380), "necrosis_like", types)
    types = np.where((x > 720) & (y < 320), "tumor_core", types)

    df_cells = pd.DataFrame({"cell_id": cell_ids, "x": x, "y": y})
    df_cells.to_parquet(out / "cells.parquet", index=False)

    df_meta = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "cell_type": types,
            "cluster": rng.integers(0, 12, n),
            "niche": rng.choice(["outer", "inner", "interface"], n),
        }
    )
    df_meta.to_parquet(out / "cell_metadata.parquet", index=False)

    pad = 40.0
    min_x_b = float(x.min() - pad)
    max_x_b = float(x.max() + pad)
    min_y_b = float(y.min() - pad)
    max_y_b = float(y.max() + pad)
    world_w_b = max_x_b - min_x_b
    world_h_b = max_y_b - min_y_b

    genes = [
        "EPCAM",
        "CD3E",
        "CD68",
        "COL1A1",
        "MKI67",
        "PDCD1",
        "ACTA2",
        "CD8A",
        "CD4",
        "FOXP3",
        "LAG3",
        "MS4A1",
        "CD19",
        "PECAM1",
        "VIM",
        "KRT8",
        "KRT18",
        "SPP1",
        "CXCL12",
        "TGFB1",
        "ARG1",
        "NOS2",
        "DCN",
        "POSTN",
        "ACTB",
        "GAPDH",
    ]
    expr = {"cell_id": cell_ids}
    for g in genes:
        base = rng.lognormal(0, 0.45, n)
        if g == "EPCAM":
            base *= np.where(types == "tumor_core", 6.0, 0.35)
        elif g == "CD3E":
            base *= np.where(types == "immune_rich", 5.0, 0.4)
        elif g == "CD68":
            base *= np.where(np.isin(types, ["immune_rich", "stromal"]), 2.5, 0.3)
        elif g == "COL1A1":
            base *= np.where(types == "stromal", 4.0, 0.5)
        elif g in ("ACTA2", "POSTN", "DCN"):
            base *= np.where(types == "stromal", 3.2, 0.45)
        elif g in ("CD8A", "CD4", "FOXP3", "LAG3"):
            base *= np.where(types == "immune_rich", 4.2, 0.35)
        elif g in ("MS4A1", "CD19"):
            base *= np.where(types == "immune_rich", 3.0, 0.4)
        elif g in ("PECAM1",):
            base *= np.where(types == "stromal", 2.2, 0.5)
        elif g in ("KRT8", "KRT18"):
            base *= np.where(types == "tumor_core", 5.0, 0.4)
        elif g in ("SPP1", "CXCL12", "TGFB1"):
            base *= np.where(np.isin(types, ["necrosis_like", "tumor_core"]), 2.8, 0.45)
        elif g in ("ARG1", "NOS2"):
            base *= np.where(types == "immune_rich", 3.5, 0.38)
        elif g in ("ACTB", "GAPDH"):
            base *= rng.uniform(0.85, 1.15, n)
        elif g == "MKI67":
            base *= np.where(types == "tumor_core", 4.5, 0.45)
        elif g == "VIM":
            base *= np.where(types != "necrosis_like", 2.0, 0.25)
        expr[g] = np.log1p(base)

    pd.DataFrame(expr).to_parquet(out / "expression-wide.parquet", index=False)

    # Polygons: one segmentation-like footprint per cell (4–7 vertices); scales with viewport "Cell size"
    # in the UI. World radius from sample span and n (same order as typical inter-cell spacing).
    feats_lod: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    xspan = float(x.max() - x.min()) + 1e-6
    yspan = float(y.max() - y.min()) + 1e-6
    area_approx = (xspan + 40) * (yspan + 40)
    density_r = math.sqrt(area_approx / (math.pi * max(n, 1)))
    base_r_default = float(np.clip(density_r * 0.52, 5.2, 14.5))

    def irregular_cell_polygon(cx: float, cy: float, cell_rng: np.random.Generator, base_r: float) -> Polygon:
        nv = int(cell_rng.integers(4, 8))
        rot = float(cell_rng.uniform(0, 2 * math.pi))
        jitter = cell_rng.normal(0, 0.11, nv)
        thetas = rot + np.linspace(0, 2 * math.pi, nv, endpoint=False) + jitter
        rs = base_r * (0.82 + 0.36 * cell_rng.random(nv))
        xs = cx + rs * np.cos(thetas)
        ys = cy + rs * np.sin(thetas)
        coords = list(zip(xs.tolist(), ys.tolist()))
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            poly = Polygon(
                [
                    (cx - base_r, cy - base_r),
                    (cx + base_r, cy - base_r),
                    (cx + base_r, cy + base_r),
                    (cx - base_r, cy + base_r),
                ]
            )
        return poly

    lod_tolerances = (0.0, 1.15, 3.2)

    # LOD0 footprints also drive H&E rasterization (same alignment as polygons / centroids).
    cell_footprints: list[tuple[Polygon, str, float, float]] = []

    for i in range(n):
        cx, cy = float(x[i]), float(y[i])
        cid = str(cell_ids[i])
        cell_rng = np.random.default_rng(42 + i)
        br = float(np.clip(base_r_default * (0.88 + 0.24 * cell_rng.random()), 4.8, 15.0))
        poly0 = irregular_cell_polygon(cx, cy, cell_rng, br)
        cell_footprints.append((poly0, str(types[i]), cx, cy))
        for lod, tol in enumerate(lod_tolerances):
            poly = poly0
            if tol > 0:
                try:
                    sp = poly.simplify(tol, preserve_topology=True)
                    if not sp.is_empty and sp.area > 1e-9:
                        poly = sp
                except Exception:
                    pass
            gj = mapping(poly)
            b = poly.bounds
            feats_lod[lod].append(
                {
                    "id": f"cell_poly_{cid}",
                    "geometry_json": json.dumps(gj),
                    "cell_id": cid,
                    "min_x": b[0],
                    "min_y": b[1],
                    "max_x": b[2],
                    "max_y": b[3],
                }
            )

    for lod in (0, 1, 2):
        pd.DataFrame(feats_lod[lod]).to_parquet(out / f"polygons_lod{lod}.parquet", index=False)

    # Transcripts: coarse tiles z=0 only
    tx_n = min(35_000, max(400, int(n * 25)))
    tx_x = rng.uniform(x.min(), x.max(), tx_n)
    tx_y = rng.uniform(y.min(), y.max(), tx_n)
    tx_g = rng.choice(["EPCAM", "CD3E", "LYZ"], tx_n)
    df_tx = pd.DataFrame({"x": tx_x, "y": tx_y, "gene_id": tx_g})
    df_tx.to_parquet(out / "transcripts" / "0_0_0.parquet", index=False)

    # Precomputed UMAP (synthetic projection — not statistical analysis at runtime)
    theta = rng.uniform(0, 2 * math.pi, n)
    r = rng.gamma(2.0, 1.5, n)
    umap = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "u": r * np.cos(theta),
            "v": r * np.sin(theta),
            "cell_type": types,
        }
    )
    comp = {"labels": ["stromal", "immune_rich", "necrosis_like", "tumor_core"], "counts": []}
    for lab in comp["labels"]:
        comp["counts"].append(int(np.sum(types == lab)))

    (out / "plots").mkdir(exist_ok=True)
    (out / "plots" / "umap.json").write_text(
        json.dumps({"points": umap.to_dict(orient="records"), "composition": comp}, indent=2)
    )
    (out / "plots" / "composition.json").write_text(json.dumps(comp, indent=2))

    # H&E (synthetic): eosin cytoplasm + hematoxylin nuclei, world-aligned to cell footprints — medium resolution.
    he_max_edge = 1420  # px on long edge (~1–2 MB PNG typical for toy samples)
    he_scale = he_max_edge / max(world_w_b, world_h_b)
    img_w = max(1, int(round(world_w_b * he_scale)))
    img_h = max(1, int(round(world_h_b * he_scale)))

    def world_to_px(wx: float, wy: float) -> tuple[int, int]:
        ix = int(round((wx - min_x_b) / world_w_b * (img_w - 1)))
        iy = int(round((max_y_b - wy) / world_h_b * (img_h - 1)))
        return int(np.clip(ix, 0, img_w - 1)), int(np.clip(iy, 0, img_h - 1))

    cyt_colors = {
        "stromal": ((236, 204, 196), (118, 82, 118)),
        "immune_rich": ((243, 210, 205), (88, 64, 112)),
        "necrosis_like": ((205, 178, 162), (110, 84, 78)),
        "tumor_core": ((231, 188, 182), (72, 48, 92)),
    }

    rng_he = np.random.default_rng(137)
    grain = rng_he.normal(0, 5.5, (img_h, img_w, 3))
    bg = np.clip(np.array([241.0, 216.0, 208.0], dtype=np.float32) + grain, 0, 255).astype(np.uint8)
    img = Image.fromarray(bg, mode="RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.9))

    dr = ImageDraw.Draw(img)

    # Coarse collagen / matrix threads (sparse curves in world space)
    for _ in range(max(180, min(650, int(n * 0.35)))):
        ax = rng_he.uniform(min_x_b, max_x_b)
        ay = rng_he.uniform(min_y_b, max_y_b)
        bx = ax + rng_he.normal(0, 22)
        by = ay + rng_he.normal(0, 16)
        col = (
            int(rng_he.integers(215, 238)),
            int(rng_he.integers(188, 215)),
            int(rng_he.integers(178, 205)),
        )
        dr.line([world_to_px(ax, ay), world_to_px(bx, by)], fill=col, width=max(1, int(round(0.35 * he_scale))))

    # Draw cells: large polygons first so smaller cells stay visible on top
    order = sorted(range(len(cell_footprints)), key=lambda k: cell_footprints[k][0].area, reverse=True)
    for idx in order:
        poly, ctype, cx, cy = cell_footprints[idx]
        cell_rng = np.random.default_rng(9001 + idx)
        cyt, nuc = cyt_colors.get(ctype, cyt_colors["stromal"])
        # Slight per-cell tint
        cyt = tuple(int(np.clip(c + cell_rng.normal(0, 5), 0, 255)) for c in cyt)
        outline = tuple(int(np.clip(c * 0.82 + 35, 0, 255)) for c in cyt)

        ring = list(poly.exterior.coords)
        if len(ring) < 4:
            continue
        pix = [world_to_px(float(px), float(py)) for px, py in ring[:-1]]
        dr.polygon(pix, fill=cyt, outline=outline)

        # Hematoxylin nucleus (dataset x/y); draw in pixel space so ovals stay round.
        r_w = float(np.clip(1.9 + cell_rng.normal(0, 0.35), 1.2, 3.4))
        ox = cell_rng.normal(0, r_w * 0.25)
        oy = cell_rng.normal(0, r_w * 0.25)
        nxc, nyc = cx + ox, cy + oy
        pcx, pcy = world_to_px(nxc, nyc)
        rx_px = max(2, int(round(r_w * he_scale)))
        ry_px = max(2, int(round(r_w * he_scale * rng_he.uniform(0.86, 1.12))))
        nuc_fill = tuple(int(np.clip(c + cell_rng.normal(0, 6), 0, 255)) for c in nuc)
        dr.ellipse(
            [pcx - rx_px, pcy - ry_px, pcx + rx_px, pcy + ry_px],
            fill=nuc_fill,
            outline=tuple(max(0, c - 25) for c in nuc_fill),
        )

        # Chromatin speckle
        if cell_rng.random() > 0.18:
            spx, spy = world_to_px(nxc + cell_rng.normal(0, r_w * 0.3), nyc + cell_rng.normal(0, r_w * 0.28))
            dr.ellipse(
                [spx - 1, spy - 1, spx + 1, spy + 1],
                fill=(max(0, nuc_fill[0] - 20), max(0, nuc_fill[1] - 15), max(0, nuc_fill[2] - 10)),
            )

    # Global light blur to merge stained look, then mild sharpen for nuclear edge
    img = img.filter(ImageFilter.GaussianBlur(radius=0.35))
    img.save(out / "he.png", format="PNG", optimize=True, compress_level=9)

    bounds = {"min_x": min_x_b, "min_y": min_y_b, "max_x": max_x_b, "max_y": max_y_b}

    manifest = {
        "version": "1.0",
        "samples": [
            {
                "sample_id": sample_id,
                "title": ("Toy tissue (small)" if n <= 2000 else "Demonstration tissue (synthetic)"),
                "description": (
                    "Small synthetic spatial layout for quick testing."
                    if n <= 2000
                    else "Synthetic spatial layout with expression — for UI/performance testing only."
                ),
                "coordinate_unit": "microns",
                "bounds": bounds,
                "image": {
                    "type": "osd_simple",
                    "url": f"/api/assets/he/{sample_id}",
                    "alignment": {"mode": "identity", "bounds": bounds},
                },
                "layers": ["centroids", "segmentation", "transcripts", "he"],
                "genes_available": len(genes),
            }
        ],
    }
    (data_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote sample '{sample_id}' ({n} cells) under {out}")
    print(f"Manifest: {data_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
