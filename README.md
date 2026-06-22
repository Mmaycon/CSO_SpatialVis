# CSO SpatialVis

High-performance **viewer + annotation** tool for precomputed spatial omics. The server does **not** run clustering, DE, normalization, or other analysis — it serves pre-aligned assets, viewport-filtered tiles, metadata, and persisted region annotations.

## Stack

- **Frontend:** React (Vite) + deck.gl (WebGL) + optional H&E underlay (`BitmapLayer`; pyramids/OpenSeadragon described below).
- **Backend:** FastAPI + PostgreSQL (annotations + derived cell tags).
- **Data:** Parquet on disk (cells, metadata, polygons per LOD, wide expression matrix with column subsets).

## Export pipelines: required inputs

Both `scripts/xenium_to_spatialvis.py` and `scripts/cosmx_to_spatialvis.py` write the **same CSO_SpatialVis sample folder** under `--out-dir`:

```
data/my_sample_id/
  cells.parquet              # cell_id, x, y (micrometers)
  cell_metadata.parquet      # per-cell metadata columns
  expression-wide.parquet    # log1p wide matrix (cell_id + gene columns)
  polygons_lod0.parquet      # segmentation polygons (LOD 0–2)
  polygons_lod1.parquet
  polygons_lod2.parquet
  transcripts/0_0_0.parquet  # optional transcript tile (empty by default)
  he.png                     # morphology underlay (registration channel)
  Images/*.png               # per-channel morphology PNGs
  images_manifest.json       # channel list + micron mapping
  sample_manifest_entry.json # bounds, layers, image alignment
  plots/umap.json            # precomputed UMAP (real or synthetic)
  plots/composition.json
  pathology_cell_tags.parquet
```

Point the app at the folder with `CSO_DATA_ROOT=./data` (or place samples under `data/`).

### `xenium_to_spatialvis.py` — 10x Xenium region output

**CLI:** `--xenium-dir` (Xenium **region** folder), `--out-dir` (sample id folder name)

**Required inputs** under `--xenium-dir`:

| Object | Typical path | Used for |
|--------|--------------|----------|
| Cell table | `cells.parquet` or `cells.csv[.gz]` | Centroids (`cell_id`, `x`, `y` in **µm**) + metadata columns |
| Expression matrix | `cell_feature_matrix.h5` or `cell_feature_matrix/` MEX (`matrix.mtx` + barcodes + features) | `expression-wide.parquet` |
| Cell boundaries | `cell_boundaries.parquet` or `cell_boundaries.csv[.gz]` | `polygons_lod*.parquet` (square footprints if missing) |

**Optional inputs** (recommended for full viewer experience):

| Object | Typical path | Used for |
|--------|--------------|----------|
| Morphology OME-TIFF | `morphology_focus/ch0001_*.ome.tif`, … (multi-channel) **or** `morphology_focus/morphology_focus_0000.ome.tif`, `morphology_mip.ome.tif`, `morphology.ome.tif` | `he.png`, `Images/*.png`, `images_manifest.json` |
| Experiment metadata | `experiment.xenium` | Pixel size (`pixel_um`) for morphology alignment |
| Analysis CSVs | `analysis/**/umap*.csv`, `analysis/**/clusters.csv` | Real UMAP/composition plots (synthetic if absent) |
| Transcripts | `transcripts.parquet` or `transcripts.csv[.gz]` | `transcripts/0_0_0.parquet` (only with `--export-transcripts`) |

**Dependencies:** `pip install -r scripts/requirements-xenium-pipeline.txt`

```bash
python scripts/xenium_to_spatialvis.py \
  --xenium-dir /path/to/output-...__Region__... \
  --out-dir ./data/xenium_myregion_full
```

Use `scripts/xenium_subset_to_spatialvis.py` with `--n-cells` for a random subset instead of all cells.

### `cosmx_to_spatialvis.py` — CosMx SpatialData Zarr

**CLI:** `--sdata-zarr`, `--out-dir`; optional `--morphology-zarr`, `--transcripts-parquet`

**Required inputs:**

| Object | Typical path | Used for |
|--------|--------------|----------|
| SpatialData store | `cosmx.sdata.zarr/` | Root container |
| AnnData table | `cosmx.sdata.zarr/tables/table/` | Expression (`X`), gene names (`var`), cell metadata (`obs`) |
| Cell coordinates | `cosmx.sdata.zarr/tables/table/obsm/global` | Centroids in **global pixel** coords (converted to µm via `pixel_size_um`) |
| Store metadata | `cosmx.sdata.zarr/.zattrs` | `pixel_size_um`, `mosaic_shape_yx` (or `image_shape_yx`) |

**Recommended for morphology** (auto-discovered as sibling `morphology.ome.zarr`, or pass `--morphology-zarr`):

| Object | Typical path | Used for |
|--------|--------------|----------|
| OME-NGFF pyramid | `morphology.ome.zarr/` | Full-slide `he.png` + `Images/*.png` (5 channels: PanCK, G, CD298_B2M, CD45, DNA) |

**Optional inputs:**

| Object | Typical path | Used for |
|--------|--------------|----------|
| Cell polygons | `cosmx.sdata.zarr/shapes/cell_boundaries/shapes.parquet` | `polygons_lod*.parquet` |
| Transcript cache | `transcripts.parquet` (beside the zarr) | `transcripts/0_0_0.parquet` (only with `--export-transcripts`) |

**Dependencies:** `pip install -r scripts/requirements-cosmx-pipeline.txt`

```bash
python scripts/cosmx_to_spatialvis.py \
  --sdata-zarr /path/to/cosmx.sdata.zarr \
  --morphology-zarr /path/to/morphology.ome.zarr \
  --out-dir ./data/GLP1_BrCa_cosmx \
  --export-all-cells
```

Re-export morphology only on an existing sample: `--morphology-only`. For CosMx, use `--morphology-flip-y` (default in `scripts/get_cosmx_export.sh`) so the PNG aligns with `obsm['global']` centroids.

### Xenium morphology (UI notes)

See **Export pipelines** above for required Xenium input files. In the UI, **Morphology** shows **only the registration/reference plane** chosen in the toolbar. Enable **Multi-channel** (when the export has 2+ planes) to overlay **every** exported segmentation channel with separate toggles and opacity — without blending those planes into the Morphology checkbox.

### CosMx quick export

If you already built a CosMx SpatialData store (e.g. with `cosmx_to_spatialdata.py`), see the input table above. Convenience wrapper with GLP1 breast cancer defaults:

```bash
# Quick subset (default 5000 cells)
bash scripts/get_cosmx_export.sh

# Full sample (~631k cells)
N_CELLS=0 bash scripts/get_cosmx_export.sh
```

Defaults point at `/mnt/scratch2/Maycon/Visualization_tools/CosMx_images/GLP1_BrCa/cosmx.sdata.zarr`.
Conda env (optional): `conda env create -f scripts/conda-cosmx-pipeline.env.yml`.

## Quick start (Docker)

From this directory:

```bash
docker compose up --build
```

Then open **http://localhost:8080** (UI) and **http://localhost:8000/docs** (API). On first backend start, the container generates a synthetic `sample_demo` dataset under the `spatial_data` Docker volume.

Postgres is exposed on **localhost:5433** for inspection.

## Local development

### Local run

From the repository root, create an isolated Python environment, install backend dependencies, and generate a **small toy dataset** (~800 cells, sample id `toy_small`):

```bash
cd /path/to/CSO_SpatialVis
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
python scripts/generate_example_data.py --out ./data --toy
```

Use the **same activated `.venv`** when you run the backend commands below so FastAPI uses those packages.

### Optional: larger or custom datasets

Full-size demo (~12k cells, sample id `sample_demo`), using the same venv:

```bash
python scripts/generate_example_data.py --out ./data
```

Custom size:

```bash
python scripts/generate_example_data.py --out ./data --cells 1500 --sample-id my_sample
```

### Backend

From the **repository root** (where `./data` lives):

```bash
source .venv/bin/activate   # if not already active
export CSO_DATA_ROOT="$(pwd)/data"
export CSO_DATABASE_URL="postgresql+psycopg2://spatial:spatial@localhost:5433/spatialvis"
cd backend
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8000`.

## Design highlights

- **Viewport loads:** `/api/cells` and `/api/polygons` filter by bounding box; caps prevent huge payloads.
- **LOD:** Polygon simplification per `lod` level; zoom drives which LOD is requested.
- **Gene overlay:** `/api/gene_expression` reads **only** the requested gene column from Parquet (plus `cell_id`).
- **ROI annotations:** POST `/api/annotations` stores GeoJSON geometry; cells receive **derived** tags (`pathology_region*`) without rewriting upstream pipeline outputs on disk.
- **Linked panels:** UMAP JSON is precomputed; the UI highlights by selection IDs (no recomputation).

## OpenSeadragon / pyramids

The prototype uses deck.gl `BitmapLayer` for a single full-resolution PNG aligned to manifest bounds — sufficient for many demos. For production pyramids, add a DZI (or IIIF) asset and mount the included pattern: browser OpenSeadragon over the image, synchronized to the same world bounds (manifest `image.alignment`).

## API outline

| Endpoint | Role |
|----------|------|
| `GET /api/sample_manifest` | Samples, bounds, layers |
| `GET /api/cells` | Viewport-filtered centroids + metadata merge |
| `GET /api/polygons` | Viewport + LOD polygons |
| `GET /api/transcripts?tile=z/x/y` | Transcript tile (prototype) |
| `GET /api/gene_expression` | Column subset expression |
| `GET /api/plots/{type}` | Precomputed plot JSON |
| `GET/POST/PUT /api/annotations` | ROI persistence |

## Project layout

```
CSO_SpatialVis/
  backend/           # FastAPI application
  frontend/          # React + deck.gl UI
  scripts/           # Example data generator
  data/              # Generated assets (created locally or in volume)
  docker/            # Container recipes
```

## Notes for scale (5k–1M cells)

- Increase `CSO_MAX_CELLS_PER_VIEWPORT` as needed; rely on spatial binning / density layers for extreme zoom-out.
- Store expression in chunked Zarr or per-gene Parquet shards; keep the **column-subset read** pattern.
- Add an **aggregated density** layer at low zoom (pre-binned tiles) if point counts remain high.
