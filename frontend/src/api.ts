import type { CellRecord, Manifest } from "./types";

export async function fetchManifest(): Promise<Manifest> {
  const r = await fetch("/api/sample_manifest");
  if (!r.ok) throw new Error(`manifest ${r.status}`);
  return r.json();
}

export async function fetchCells(
  sampleId: string,
  bbox: { min_x: number; min_y: number; max_x: number; max_y: number },
): Promise<{
  cells: CellRecord[];
  truncated: boolean;
  total_in_viewport: number;
  returned: number;
}> {
  const u = new URL("/api/cells", window.location.href);
  u.searchParams.set("sample_id", sampleId);
  u.searchParams.set("min_x", String(bbox.min_x));
  u.searchParams.set("min_y", String(bbox.min_y));
  u.searchParams.set("max_x", String(bbox.max_x));
  u.searchParams.set("max_y", String(bbox.max_y));
  const r = await fetch(u.toString());
  if (!r.ok) throw new Error(`cells ${r.status}`);
  return r.json();
}

export async function fetchPolygons(
  sampleId: string,
  lod: number,
  bbox: { min_x: number; min_y: number; max_x: number; max_y: number },
) {
  const u = new URL("/api/polygons", window.location.href);
  u.searchParams.set("sample_id", sampleId);
  u.searchParams.set("lod", String(lod));
  u.searchParams.set("min_x", String(bbox.min_x));
  u.searchParams.set("min_y", String(bbox.min_y));
  u.searchParams.set("max_x", String(bbox.max_x));
  u.searchParams.set("max_y", String(bbox.max_y));
  const r = await fetch(u.toString());
  if (!r.ok) throw new Error(`polygons ${r.status}`);
  return r.json();
}

export async function fetchColorColumns(sampleId: string): Promise<{
  sample_id: string;
  metadata_columns: string[];
  genes: string[];
}> {
  const u = new URL("/api/color_columns", window.location.href);
  u.searchParams.set("sample_id", sampleId);
  const r = await fetch(u.toString());
  if (r.ok) return r.json();

  // Older uvicorn processes may not have /api/color_columns; compose from genes + metadata schema.
  if (r.status === 404) {
    const base = window.location.href;
    const [gr, mr] = await Promise.all([
      fetch(new URL(`/api/genes?sample_id=${encodeURIComponent(sampleId)}`, base).toString()),
      fetch(new URL(`/api/metadata_columns?sample_id=${encodeURIComponent(sampleId)}`, base).toString()),
    ]);
    const genes = gr.ok ? (((await gr.json()) as { genes?: string[] }).genes ?? []) : [];
    const metaRaw = mr.ok ? (((await mr.json()) as { metadata_columns?: string[] }).metadata_columns ?? []) : [];
    return {
      sample_id: sampleId,
      metadata_columns: metaRaw.length ? metaRaw : ["cell_type"],
      genes,
    };
  }

  throw new Error(`color_columns ${r.status}`);
}

export async function fetchGenes(sampleId: string): Promise<{ sample_id: string; genes: string[] }> {
  const u = new URL("/api/genes", window.location.href);
  u.searchParams.set("sample_id", sampleId);
  const r = await fetch(u.toString());
  if (!r.ok) throw new Error(`genes ${r.status}`);
  return r.json();
}

/** Full sample column if `cellIds` omitted; subset when a non-empty list is passed. */
export async function fetchGeneExpression(
  sampleId: string,
  gene: string,
  cellIds?: string[],
): Promise<{ values: Record<string, number> }> {
  const u = new URL("/api/gene_expression", window.location.href);
  u.searchParams.set("sample_id", sampleId);
  u.searchParams.set("gene", gene);
  if (cellIds !== undefined && cellIds.length > 0) {
    u.searchParams.set("cell_ids", cellIds.join(","));
  }
  const r = await fetch(u.toString());
  if (!r.ok) throw new Error(`gene ${r.status}`);
  return r.json() as Promise<{ values: Record<string, number> }>;
}

export async function fetchPlot(sampleId: string, plotType: string) {
  const u = new URL(`/api/plots/${plotType}`, window.location.href);
  u.searchParams.set("sample_id", sampleId);
  const r = await fetch(u.toString());
  if (!r.ok) throw new Error(`plot ${r.status}`);
  return r.json();
}

export async function postAnnotation(body: {
  sample_id: string;
  label: string;
  author: string;
  notes?: string;
  geometry: GeoJSON.Polygon;
  confidence?: number;
}) {
  const r = await fetch("/api/annotations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`annotate ${r.status}`);
  return r.json() as Promise<{
    id: string;
    cells_assigned: number;
    label: string;
    sample_id: string;
  }>;
}

export async function postAnnotationByCells(body: {
  sample_id: string;
  label: string;
  author: string;
  notes?: string;
  cell_ids: string[];
  confidence?: number;
}) {
  const r = await fetch("/api/annotations/by_cells", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`annotate_cells ${r.status}`);
  return r.json() as Promise<{
    id: string;
    cells_assigned: number;
    label: string;
    sample_id: string;
  }>;
}
