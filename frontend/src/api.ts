import type { CellRecord, Manifest } from "./types";

/** Readable FastAPI error (`detail`) or raw body for debugging failed saves. */
async function apiFailureMessage(r: Response, label: string): Promise<string> {
  let raw = "";
  try {
    raw = await r.clone().text();
    const j = JSON.parse(raw) as { detail?: unknown };
    const d = j.detail;
    if (typeof d === "string") return `${label}: ${d}`;
    if (Array.isArray(d)) {
      const parts = d.map((x) =>
        typeof x === "object" && x !== null && "msg" in x ? String((x as { msg: string }).msg) : JSON.stringify(x),
      );
      return `${label}: ${parts.join("; ")}`;
    }
    if (d != null) return `${label}: ${JSON.stringify(d)}`;
  } catch {
    /* use raw */
  }
  if (raw) return `${label} (${r.status}): ${raw.slice(0, 800)}`;
  return `${label}: HTTP ${r.status}`;
}

export async function fetchManifest(): Promise<Manifest> {
  const r = await fetch("/api/sample_manifest");
  if (!r.ok) throw new Error(`manifest ${r.status}`);
  return r.json();
}

/** Persist morphology micron shift to ``data/manifest.json`` (merged into sample override). */
export async function patchSampleImageTranslate(
  sampleId: string,
  body: { translate_um: [number, number] },
): Promise<Manifest> {
  const r = await fetch(
    `/api/sample_manifest/samples/${encodeURIComponent(sampleId)}/image_translate`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ translate_um: body.translate_um }),
    },
  );
  if (!r.ok) throw new Error(`image_translate ${r.status}`);
  return r.json();
}

export async function fetchCells(
  sampleId: string,
  bbox: { min_x: number; min_y: number; max_x: number; max_y: number },
  opts?: { truncate?: boolean },
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
  if (opts?.truncate === false) {
    u.searchParams.set("truncate", "false");
  }
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
  metadata_column_kinds?: Record<string, string>;
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
      metadata_column_kinds: {},
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
/** Precomputed transcript tile (``data/<sample>/transcripts/z_x_y.parquet``). Usually ``0/0/0`` for exports. */
export async function fetchTranscripts(
  sampleId: string,
  tile: string,
): Promise<{ sample_id: string; tile_key: string; points: [number, number, string][] }> {
  const u = new URL("/api/transcripts", window.location.href);
  u.searchParams.set("sample_id", sampleId);
  u.searchParams.set("tile", tile);
  const r = await fetch(u.toString());
  if (!r.ok) throw new Error(`transcripts ${r.status}`);
  return r.json();
}

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
  const ctrl = new AbortController();
  const tid = window.setTimeout(() => ctrl.abort(), 180_000);
  try {
    const r = await fetch("/api/annotations/by_cells", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!r.ok) throw new Error(await apiFailureMessage(r, "annotate_cells"));
    return r.json() as Promise<{
      id: string;
      cells_assigned: number;
      label: string;
      sample_id: string;
    }>;
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new Error(
        "annotate_cells: request timed out after 3m — large exports can be slow; check that the API is running (port 8000) and try again.",
      );
    }
    throw e;
  } finally {
    window.clearTimeout(tid);
  }
}
