"""FastAPI routes — precomputed data + annotation persistence."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.cell_assignment import cells_inside_polygon
from app.config import settings
from app.data_store import (
    bbox_polygon_for_cell_ids,
    invalidate_sample_parquet_caches,
    list_color_columns,
    list_expression_genes,
    list_metadata_columns_ordered,
    load_manifest,
    overlay_pathology_metadata,
    persist_sample_image_translate_um,
    read_cells_viewport,
    read_gene_expression,
    read_plot,
    read_polygons_viewport,
    read_transcript_tile,
    sample_dir,
    sync_sample_roi_parquet_from_tags,
)
from app.database import CellRegionTagRow, RegionAnnotationRow, get_db
from app.schemas import (
    AnnotationByCellIds,
    AnnotationCreate,
    AnnotationRecord,
    AnnotationUpdate,
    CellsResponse,
    CellRecord,
    ColorColumnsResponse,
    MetadataColumnsResponse,
    GeneExpressionResponse,
    GenesResponse,
    ImageTranslatePatch,
    PlotEnvelope,
    PolygonsResponse,
    PolygonFeature,
    SampleManifest,
    TranscriptsResponse,
)

router = APIRouter()

# SQLite (and some drivers) cap SQL bind parameters per query (often 999). Batch large IN lists.
_IN_CHUNK = 500


def _db_roi_tags_for_sample_cells(db: Session, sample_id: str, cids: list[str]) -> dict[str, dict[str, object]]:
    """All ROI tag rows in DB for the given (sample, cell_id) pairs."""
    if not cids:
        return {}
    out: dict[str, dict] = {}
    for i in range(0, len(cids), _IN_CHUNK):
        part = cids[i : i + _IN_CHUNK]
        for t in (
            db.query(CellRegionTagRow)
            .filter(CellRegionTagRow.sample_id == sample_id)
            .filter(CellRegionTagRow.cell_id.in_(part))
            .all()
        ):
            out[t.cell_id] = {
                "pathology_region": t.pathology_region,
                "pathology_region_source": t.pathology_region_source,
                "pathology_region_confidence": t.pathology_region_confidence,
            }
    return out


def _merge_roi_tags_for_cells(
    db: Session,
    *,
    sample_id: str,
    annotation_id: str,
    label: str,
    confidence: float,
    cell_ids: list[str],
) -> None:
    """SQLite + Postgres: upsert one tag row per (sample_id, cell_id)."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for x in cell_ids:
        s = str(x).strip()
        if s and s not in seen_set:
            seen_set.add(s)
            seen.append(s)
    cell_ids = seen

    conf_f = float(confidence or 1.0)
    src = "roi_assignment"

    existing: dict[str, CellRegionTagRow] = {}
    for i in range(0, len(cell_ids), _IN_CHUNK):
        part = cell_ids[i : i + _IN_CHUNK]
        for row in (
            db.query(CellRegionTagRow)
            .filter(CellRegionTagRow.sample_id == sample_id)
            .filter(CellRegionTagRow.cell_id.in_(part))
            .all()
        ):
            existing[row.cell_id] = row

    for cid in cell_ids:
        row = existing.get(cid)
        if row:
            row.annotation_id = annotation_id
            row.pathology_region = label
            row.pathology_region_source = src
            row.pathology_region_confidence = conf_f
        else:
            db.add(
                CellRegionTagRow(
                    sample_id=sample_id,
                    cell_id=cid,
                    annotation_id=annotation_id,
                    pathology_region=label,
                    pathology_region_source=src,
                    pathology_region_confidence=conf_f,
                )
            )


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/sample_manifest", response_model=SampleManifest)
def sample_manifest():
    return load_manifest()


@router.patch("/sample_manifest/samples/{sample_id}/image_translate", response_model=SampleManifest)
def patch_sample_image_translate(sample_id: str, body: ImageTranslatePatch):
    try:
        persist_sample_image_translate_um(sample_id, body.translate_um[0], body.translate_um[1])
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return load_manifest()


@router.get("/cells", response_model=CellsResponse)
def cells(
    sample_id: str = Query(...),
    min_x: float = Query(...),
    min_y: float = Query(...),
    max_x: float = Query(...),
    max_y: float = Query(...),
    truncate: bool = Query(
        True,
        description="When false, return all cells in the bbox (up to CSO_MAX_CELLS_FULL_LOAD) for centroid overview.",
    ),
    db: Session = Depends(get_db),
):
    max_cells = settings.max_cells_per_viewport if truncate else settings.max_cells_full_load
    rows, total, truncated = read_cells_viewport(sample_id, min_x, min_y, max_x, max_y, max_cells)

    cids = [r["cell_id"] for r in rows]
    pathology_overlay = overlay_pathology_metadata(sample_id, cids)
    db_tags = _db_roi_tags_for_sample_cells(db, sample_id, cids)

    cells_out: list[CellRecord] = []
    for r in rows:
        cid = r["cell_id"]
        md = dict(r["metadata"])
        md.update(pathology_overlay.get(cid, {}))
        md.update(db_tags.get(cid, {}))
        cells_out.append(CellRecord(cell_id=cid, x=r["x"], y=r["y"], metadata=md))

    return CellsResponse(
        sample_id=sample_id,
        viewport={"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
        total_in_viewport=total,
        returned=len(cells_out),
        truncated=truncated,
        cells=cells_out,
    )


@router.get("/polygons", response_model=PolygonsResponse)
def polygons(
    sample_id: str = Query(...),
    lod: int = Query(0, ge=0, le=2),
    min_x: float = Query(...),
    min_y: float = Query(...),
    max_x: float = Query(...),
    max_y: float = Query(...),
):
    feats, poly_total, poly_truncated = read_polygons_viewport(
        sample_id, lod, min_x, min_y, max_x, max_y, settings.max_polygons_per_viewport
    )
    return PolygonsResponse(
        sample_id=sample_id,
        lod=lod,
        viewport={"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
        total_in_viewport=poly_total,
        returned=len(feats),
        truncated=poly_truncated,
        features=[PolygonFeature(**f) for f in feats],
    )


@router.get("/transcripts", response_model=TranscriptsResponse)
def transcripts(
    sample_id: str = Query(...),
    tile: str = Query(..., description="Format z/x/y"),
):
    parts = tile.split("/")
    if len(parts) != 3:
        raise HTTPException(400, "tile must be z/x/y")
    z, x, y = int(parts[0]), int(parts[1]), int(parts[2])
    pts = read_transcript_tile(sample_id, z, x, y)
    return TranscriptsResponse(sample_id=sample_id, tile_key=tile, points=pts)


@router.get("/genes", response_model=GenesResponse)
def list_genes(sample_id: str = Query(...)):
    genes = list_expression_genes(sample_id)
    return GenesResponse(sample_id=sample_id, genes=genes)


@router.get("/color_columns", response_model=ColorColumnsResponse)
def color_columns(sample_id: str = Query(...)):
    data = list_color_columns(sample_id)
    return ColorColumnsResponse(
        sample_id=sample_id,
        metadata_columns=data["metadata_columns"],
        genes=data["genes"],
        metadata_column_kinds=data.get("metadata_column_kinds") or {},
    )


@router.get("/metadata_columns", response_model=MetadataColumnsResponse)
def metadata_columns_only(sample_id: str = Query(...)):
    return MetadataColumnsResponse(
        sample_id=sample_id,
        metadata_columns=list_metadata_columns_ordered(sample_id),
    )


@router.get("/gene_expression", response_model=GeneExpressionResponse)
def gene_expression(
    sample_id: str = Query(...),
    gene: str = Query(...),
    cell_ids: str | None = Query(None, description="Comma-separated subset"),
):
    subset = cell_ids.split(",") if cell_ids else None
    if subset:
        subset = [s.strip() for s in subset if s.strip()]
    vals = read_gene_expression(sample_id, gene, subset)
    return GeneExpressionResponse(sample_id=sample_id, gene=gene, values=vals)


@router.get("/plots/{plot_type}", response_model=PlotEnvelope)
def plots(plot_type: str, sample_id: str = Query(...)):
    payload = read_plot(sample_id, plot_type)
    if not payload:
        raise HTTPException(404, "Plot not found")
    return PlotEnvelope(plot_type=plot_type, sample_id=sample_id, payload=payload)


@router.get("/annotations", response_model=list[AnnotationRecord])
def list_annotations(sample_id: str | None = None, db: Session = Depends(get_db)):
    q = db.query(RegionAnnotationRow)
    if sample_id:
        q = q.filter(RegionAnnotationRow.sample_id == sample_id)
    rows = q.order_by(RegionAnnotationRow.created_at.desc()).all()
    out: list[AnnotationRecord] = []
    for r in rows:
        out.append(
            AnnotationRecord(
                id=r.id,
                sample_id=r.sample_id,
                label=r.label,
                author=r.author,
                notes=r.notes or "",
                geometry=json.loads(r.geometry_json),
                confidence=float(r.confidence or 1.0),
                created_at=r.created_at,
                updated_at=r.updated_at,
                cells_assigned=int(r.cells_assigned or 0),
            )
        )
    return out


@router.post("/annotations", response_model=AnnotationRecord)
def create_annotation(body: AnnotationCreate, db: Session = Depends(get_db)):
    rid = uuid.uuid4().hex
    now = datetime.now(timezone.utc)

    cells_path = sample_dir(body.sample_id) / "cells.parquet"
    if not cells_path.exists():
        raise HTTPException(404, "Sample cells not found")

    import pyarrow.parquet as pq

    tbl = pq.read_table(cells_path, columns=["cell_id", "x", "y"])
    df = tbl.to_pandas()
    assigned = cells_inside_polygon(
        df["cell_id"].astype(str).tolist(),
        df["x"].to_numpy(dtype=np.float64),
        df["y"].to_numpy(dtype=np.float64),
        body.geometry,
    )

    row = RegionAnnotationRow(
        id=rid,
        sample_id=body.sample_id,
        label=body.label,
        author=body.author,
        notes=body.notes,
        geometry_json=json.dumps(body.geometry),
        confidence=body.confidence,
        cells_assigned=len(assigned),
        created_at=now,
        updated_at=now,
    )
    db.add(row)

    _merge_roi_tags_for_cells(
        db,
        sample_id=body.sample_id,
        annotation_id=rid,
        label=body.label,
        confidence=body.confidence,
        cell_ids=assigned,
    )

    db.commit()
    db.refresh(row)

    _push_roi_tags_to_parquet(db, body.sample_id)

    return AnnotationRecord(
        id=row.id,
        sample_id=row.sample_id,
        label=row.label,
        author=row.author,
        notes=row.notes or "",
        geometry=json.loads(row.geometry_json),
        confidence=float(row.confidence or 1.0),
        created_at=row.created_at,
        updated_at=row.updated_at,
        cells_assigned=len(assigned),
    )


@router.post("/annotations/by_cells", response_model=AnnotationRecord)
def create_annotation_by_cells(body: AnnotationByCellIds, db: Session = Depends(get_db)):
    """Persist ROI labels for an explicit cell ID list (e.g. rectangle selection). Syncs cell_metadata.parquet."""
    ids = sorted({str(x).strip() for x in body.cell_ids if str(x).strip()})
    if not ids:
        raise HTTPException(400, "cell_ids required")

    cells_path = sample_dir(body.sample_id) / "cells.parquet"
    if not cells_path.exists():
        raise HTTPException(404, "Sample cells not found")

    geometry = bbox_polygon_for_cell_ids(body.sample_id, ids)
    if not geometry.get("coordinates"):
        raise HTTPException(400, "No matching cells for given IDs in cells.parquet")

    rid = uuid.uuid4().hex
    now = datetime.now(timezone.utc)

    row = RegionAnnotationRow(
        id=rid,
        sample_id=body.sample_id,
        label=body.label,
        author=body.author,
        notes=body.notes,
        geometry_json=json.dumps(geometry),
        confidence=body.confidence,
        cells_assigned=len(ids),
        created_at=now,
        updated_at=now,
    )
    db.add(row)

    _merge_roi_tags_for_cells(
        db,
        sample_id=body.sample_id,
        annotation_id=rid,
        label=body.label,
        confidence=body.confidence,
        cell_ids=ids,
    )

    db.commit()
    db.refresh(row)

    _push_roi_tags_to_parquet(db, body.sample_id)

    return AnnotationRecord(
        id=row.id,
        sample_id=row.sample_id,
        label=row.label,
        author=row.author,
        notes=row.notes or "",
        geometry=json.loads(row.geometry_json),
        confidence=float(row.confidence or 1.0),
        created_at=row.created_at,
        updated_at=row.updated_at,
        cells_assigned=len(ids),
    )


def _push_roi_tags_to_parquet(db: Session, sample_id: str) -> None:
    rows = (
        db.query(CellRegionTagRow)
        .filter(CellRegionTagRow.sample_id == sample_id)
        .all()
    )
    tags = [
        {
            "cell_id": t.cell_id,
            "pathology_region": t.pathology_region,
            "pathology_region_source": t.pathology_region_source or "roi_assignment",
            "pathology_region_confidence": float(t.pathology_region_confidence or 1.0),
        }
        for t in rows
    ]
    sync_sample_roi_parquet_from_tags(sample_id, tags)
    invalidate_sample_parquet_caches(sample_id)


@router.put("/annotations/{annotation_id}", response_model=AnnotationRecord)
def update_annotation(annotation_id: str, body: AnnotationUpdate, db: Session = Depends(get_db)):
    r = db.query(RegionAnnotationRow).filter(RegionAnnotationRow.id == annotation_id).first()
    if not r:
        raise HTTPException(404, "Not found")

    now = datetime.now(timezone.utc)
    if body.label is not None:
        r.label = body.label
        for tag in db.query(CellRegionTagRow).filter(CellRegionTagRow.annotation_id == annotation_id).all():
            tag.pathology_region = r.label
    if body.notes is not None:
        r.notes = body.notes
    if body.confidence is not None:
        r.confidence = body.confidence
    if body.geometry is not None:
        r.geometry_json = json.dumps(body.geometry)

        cells_path = sample_dir(r.sample_id) / "cells.parquet"
        import pyarrow.parquet as pq

        tbl = pq.read_table(cells_path, columns=["cell_id", "x", "y"])
        df = tbl.to_pandas()
        assigned = cells_inside_polygon(
            df["cell_id"].astype(str).tolist(),
            df["x"].to_numpy(dtype=np.float64),
            df["y"].to_numpy(dtype=np.float64),
            body.geometry,
        )
        r.cells_assigned = len(assigned)

        db.query(CellRegionTagRow).filter(CellRegionTagRow.annotation_id == annotation_id).delete(
            synchronize_session=False
        )
        _merge_roi_tags_for_cells(
            db,
            sample_id=r.sample_id,
            annotation_id=annotation_id,
            label=r.label,
            confidence=float(r.confidence or 1.0),
            cell_ids=assigned,
        )

    r.updated_at = now
    db.commit()
    db.refresh(r)

    _push_roi_tags_to_parquet(db, r.sample_id)

    return AnnotationRecord(
        id=r.id,
        sample_id=r.sample_id,
        label=r.label,
        author=r.author,
        notes=r.notes or "",
        geometry=json.loads(r.geometry_json),
        confidence=float(r.confidence or 1.0),
        created_at=r.created_at,
        updated_at=r.updated_at,
        cells_assigned=int(r.cells_assigned or 0),
    )


@router.get("/assets/he/{sample_id}")
def he_image(sample_id: str):
    p = sample_dir(sample_id) / "he.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@router.get("/assets/sample_files/{sample_id}/{file_path:path}")
def sample_asset_file(sample_id: str, file_path: str):
    """Serve files under a sample folder (e.g. ``Images/*.png`` from morphology export)."""
    root = sample_dir(sample_id).resolve()
    rel = Path(file_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(400, "Invalid path")
    target = (root / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(403, "Path escapes sample directory") from exc
    if not target.is_file():
        raise HTTPException(404)
    suf = target.suffix.lower()
    media = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".json": "application/json",
    }.get(suf, "application/octet-stream")
    return FileResponse(target, media_type=media)


@router.get("/assets/manifest_bounds/{sample_id}")
def bounds(sample_id: str):
    m = load_manifest()
    for s in m.samples:
        if s.sample_id == sample_id:
            return s.bounds
    raise HTTPException(404)
