# CSO SpatialVis

High-performance **viewer + annotation** tool for precomputed spatial omics. The server does **not** run clustering, DE, normalization, or other analysis — it serves pre-aligned assets, viewport-filtered tiles, metadata, and persisted region annotations.

## Stack

- **Frontend:** React (Vite) + deck.gl (WebGL) + optional H&E underlay (`BitmapLayer`; pyramids/OpenSeadragon described below).
- **Backend:** FastAPI + PostgreSQL (annotations + derived cell tags).
- **Data:** Parquet on disk (cells, metadata, polygons per LOD, wide expression matrix with column subsets).

### Xenium morphology (many segmentation channels)

Under a Xenium **region** output folder you typically find an OME-TIFF such as `morphology_focus/morphology_focus_0000.ome.tif`, `morphology_mip.ome.tif`, or `morphology.ome.tif`. To turn those planes into PNGs the viewer can load, run the exporter from this repo (writes `Images/*.png`, `he.png`, `images_manifest.json`, etc.):

```bash
python scripts/xenium_to_spatialvis.py \
  --xenium-dir /path/to/output-...__Region__... \
  --out-dir ./data/my_sample_id
```

In the UI, **Morphology** shows **only the registration/reference plane** chosen in the toolbar. Enable **Multi-channel** (when the export has 2+ planes) to overlay **every** exported segmentation channel with separate toggles and opacity — without blending those planes into the Morphology checkbox.

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
