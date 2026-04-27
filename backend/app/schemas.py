"""API and domain schemas — precomputed data only; no analysis fields."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class SampleRef(BaseModel):
    sample_id: str
    title: str
    description: str = ""
    coordinate_unit: str = "microns"
    bounds: dict[str, float]  # min_x, min_y, max_x, max_y
    image: dict[str, Any] | None = None  # osd / alignment hints
    layers: list[str] = Field(default_factory=list)
    genes_available: int = 0


class SampleManifest(BaseModel):
    version: str = "1.0"
    samples: list[SampleRef]


class CellRecord(BaseModel):
    cell_id: str
    x: float
    y: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class CellsResponse(BaseModel):
    sample_id: str
    viewport: dict[str, float]
    total_in_viewport: int
    returned: int
    truncated: bool
    cells: list[CellRecord]


class PolygonFeature(BaseModel):
    id: str
    geometry: dict[str, Any]  # GeoJSON geometry
    properties: dict[str, Any] = Field(default_factory=dict)


class PolygonsResponse(BaseModel):
    sample_id: str
    lod: int
    viewport: dict[str, float]
    total_in_viewport: int = 0
    returned: int = 0
    truncated: bool = False
    features: list[PolygonFeature] = Field(default_factory=list)


class TranscriptsResponse(BaseModel):
    sample_id: str
    tile_key: str
    points: list[tuple[float, float, str]]  # x, y, gene_id — prototype


class GeneExpressionResponse(BaseModel):
    sample_id: str
    gene: str
    values: dict[str, float]  # cell_id -> value (subset for viewport cache on client)


class GenesResponse(BaseModel):
    sample_id: str
    genes: list[str]


class ColorColumnsResponse(BaseModel):
    sample_id: str
    metadata_columns: list[str]
    genes: list[str]


class MetadataColumnsResponse(BaseModel):
    sample_id: str
    metadata_columns: list[str]


class PlotEnvelope(BaseModel):
    plot_type: Literal["umap", "heatmap", "dotplot", "violin", "composition"]
    sample_id: str
    payload: dict[str, Any]


class AnnotationCreate(BaseModel):
    sample_id: str
    label: str
    author: str
    notes: str = ""
    geometry: dict[str, Any]  # GeoJSON Polygon or MultiPolygon
    confidence: float = 1.0


class AnnotationByCellIds(BaseModel):
    """Assign ROI label to explicit cell IDs (e.g. rectangle selection) without drawing a polygon."""

    sample_id: str
    label: str
    author: str
    notes: str = ""
    cell_ids: list[str]
    confidence: float = 1.0


class AnnotationRecord(BaseModel):
    id: str
    sample_id: str
    label: str
    author: str
    notes: str
    geometry: dict[str, Any]
    confidence: float
    created_at: datetime
    updated_at: datetime
    cells_assigned: int = 0


class AnnotationUpdate(BaseModel):
    label: str | None = None
    notes: str | None = None
    geometry: dict[str, Any] | None = None
    confidence: float | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
