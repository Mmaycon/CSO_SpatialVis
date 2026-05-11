import DeckGL from "@deck.gl/react";
import { COORDINATE_SYSTEM, OrthographicView, OrthographicViewport } from "@deck.gl/core";
import { BitmapLayer, GeoJsonLayer, PathLayer, ScatterplotLayer } from "@deck.gl/layers";
import type { BitmapBoundingBox } from "@deck.gl/layers";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import * as api from "@/api";
import {
  VIRIDIS_GRADIENT_CSS,
  colorForMetadataKey,
  colorForPathology,
  colorFromExpression,
  continuousScaleModeLabel,
  overlayRgbForCell,
  parseMetadataNumeric,
  valueRangeForContinuousScale,
  type ContinuousScaleMode,
} from "@/colors";
import { useDebouncedCallback } from "@/hooks/useDebouncedCallback";
import type { CellRecord, SampleBounds } from "@/types";

const MORPH_TEXTURE_WARN_EDGE = 8192;

function morphologyBitmapBoundsForRect(
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  flipY: boolean,
  translateUm: readonly [number, number],
): BitmapBoundingBox {
  const [tx, ty] = translateUm;
  const bx = (x: number) => x + tx;
  const by = (y: number) => y + ty;
  if (!flipY) return [bx(minX), by(minY), bx(maxX), by(maxY)];
  return [
    [bx(minX), by(maxY)],
    [bx(minX), by(minY)],
    [bx(maxX), by(minY)],
    [bx(maxX), by(maxY)],
  ];
}

function morphologyBitmapBounds(bounds: SampleBounds, flipY: boolean, translateUm: readonly [number, number]): BitmapBoundingBox {
  return morphologyBitmapBoundsForRect(bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y, flipY, translateUm);
}

/** Micron-space ring (implicit edge from last vertex back to first). */
function pointInPolygonMicrons(x: number, y: number, ring: [number, number][]): boolean {
  const n = ring.length;
  if (n < 3) return false;
  let inside = false;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = ring[i][0];
    const yi = ring[i][1];
    const xj = ring[j][0];
    const yj = ring[j][1];
    const denom = yj - yi;
    if (Math.abs(denom) < 1e-18) continue;
    if ((yi > y) === (yj > y)) continue;
    const xinters = ((xj - xi) * (y - yi)) / denom + xi;
    if (x < xinters) inside = !inside;
  }
  return inside;
}

const LASSO_MIN_SAMPLE_PX = 2.5;
const LASSO_MIN_VERTICES = 3;

export type InteractionMode = "navigate" | "select_rect" | "select_lasso";

export type OrthoViewState = {
  ortho: {
    target: [number, number, number];
    zoom: number;
    minZoom: number;
    maxZoom: number;
    width?: number;
    height?: number;
  };
};

type Props = {
  sampleId: string;
  bounds: SampleBounds;
  mode: InteractionMode;
  showCentroids: boolean;
  showPolygons: boolean;
  showHe: boolean;
  heOpacity: number;
  centroidOpacity: number;
  polygonOpacity: number;
  /** Metadata column key (from cell_metadata / `$ManualAnno:*`) when paintMode is metadata */
  metadataColumn: string;
  /** From API / parquet inference: numeric columns use viridis (viewport min–max); others use categorical colors. */
  metadataColumnKind?: "numeric" | "categorical";
  /** Viridis endpoints: full min/max in view vs percentile clipping for gene & numeric metadata & overlays. */
  continuousScaleMode?: ContinuousScaleMode;
  paintMode: "metadata" | "gene";
  gene: string;
  /** Up to 3 genes: viridis (1) or RGB channels (2–3). Drawn on top of base coloring. */
  overlayGenes?: string[];
  overlayOpacity?: number;
  /** Bump after ROI save to refresh metadata from disk. */
  dataRevision?: number;
  selectedIds: ReadonlySet<string>;
  onToggleSelected: (ids: string[], replace: boolean) => void;
  /** Reference morphology plane only (registration channel). When absent, loads ``he.png`` only. */
  morphologyChannels?: Array<{ id: string; label: string; url: string; default_visible?: boolean }>;
  /** All segmentation PNGs from the OME export — separate layer from Morphology. */
  showMultiChannel?: boolean;
  multiChannelChannels?: Array<{ id: string; label: string; url: string; default_visible?: boolean }>;
  multiChannelEnabled?: Record<string, boolean>;
  multiChannelOpacity?: number;
  /** Multiplier for centroid pixel radius (1 = built-in zoom-aware default). */
  cellRadiusScale?: number;
  /** When true, flip morphology texture V vs micron Y (``image.alignment.flip_y`` / ``images_manifest.viewer_flip_y``). */
  morphologyFlipY?: boolean;
  /** Micron shift applied only to morphology BitmapLayer corners (``image.alignment.translate_um``). */
  morphologyTranslateUm?: readonly [number, number];
  /** ``morphology_frame.crop_microns``: morphology quads align here so overview matches cells on segmentation crop. */
  morphologyMicronExtent?: SampleBounds | null;
};

function initialOrthoForBounds(bounds: SampleBounds, width: number, height: number): OrthoViewState {
  const cx = (bounds.min_x + bounds.max_x) / 2;
  const cy = (bounds.min_y + bounds.max_y) / 2;
  const bw = bounds.max_x - bounds.min_x;
  const bh = bounds.max_y - bounds.min_y;
  const w = Math.max(320, width);
  const h = Math.max(240, height);
  const scale0 = Math.min(w / bw, h / bh);
  const zoom0 = Math.log2(scale0) - 0.2;
  return {
    ortho: {
      target: [cx, cy, 0],
      zoom: zoom0,
      minZoom: -12,
      maxZoom: 24,
    },
  };
}

function normalizeOrthoIncoming(incoming: unknown): Partial<{ target: [number, number, number]; zoom: number }> {
  if (incoming == null || typeof incoming !== "object") return {};
  const o = incoming as Record<string, unknown>;
  const ortho = o.ortho;
  const src =
    ortho != null && typeof ortho === "object"
      ? (ortho as Record<string, unknown>)
      : "target" in o || "zoom" in o
        ? o
        : {};
  const out: Partial<{ target: [number, number, number]; zoom: number }> = {};
  if (Array.isArray(src.target) && src.target.length >= 2) {
    const x = Number(src.target[0]);
    const y = Number(src.target[1]);
    const z = Number(src.target[2] ?? 0);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      out.target = [x, y, Number.isFinite(z) ? z : 0];
    }
  }
  if (typeof src.zoom === "number" && Number.isFinite(src.zoom)) {
    out.zoom = src.zoom;
  }
  return out;
}

function normalizeOrthoViewState(incoming: unknown, prev: OrthoViewState, width: number, height: number): OrthoViewState {
  const p = prev.ortho;
  const patch = normalizeOrthoIncoming(incoming);
  const target = patch.target ?? p.target;
  let zoom = patch.zoom ?? p.zoom;
  zoom = Math.min(p.maxZoom, Math.max(p.minZoom, zoom));
  const tw = Math.max(1, width);
  const th = Math.max(1, height);
  if (!Number.isFinite(target[0]) || !Number.isFinite(target[1]) || !Number.isFinite(zoom)) {
    return prev;
  }
  return {
    ortho: {
      target,
      zoom,
      minZoom: p.minZoom,
      maxZoom: p.maxZoom,
      width: tw,
      height: th,
    },
  };
}

function safeWorldBBoxFromViewState(vs: OrthoViewState, width: number, height: number) {
  const w = Math.max(1, width);
  const h = Math.max(1, height);
  try {
    const bbox = worldBBoxFromViewState(vs, w, h);
    if (
      Number.isFinite(bbox.min_x) &&
      Number.isFinite(bbox.max_x) &&
      Number.isFinite(bbox.min_y) &&
      Number.isFinite(bbox.max_y) &&
      bbox.min_x < bbox.max_x &&
      bbox.min_y < bbox.max_y
    ) {
      return bbox;
    }
  } catch {
    /* ignore */
  }
  return null;
}

/** Exterior rings for GeoJSON Polygon / MultiPolygon (for stroke PathLayer). */
function outlinePathsFromGeoJSON(fc: { features: any[] }): { path: [number, number][] }[] {
  const out: { path: [number, number][] }[] = [];
  for (const f of fc.features) {
    const g = f.geometry;
    if (!g) continue;
    if (g.type === "Polygon") {
      const ring = g.coordinates[0];
      if (ring?.length) out.push({ path: ring.map((p: number[]) => [p[0], p[1]] as [number, number]) });
    } else if (g.type === "MultiPolygon") {
      for (const poly of g.coordinates) {
        const ring = poly[0];
        if (ring?.length) out.push({ path: ring.map((p: number[]) => [p[0], p[1]] as [number, number]) });
      }
    }
  }
  return out;
}

function bboxCenterFromGeometry(geom: any): [number, number] | null {
  if (!geom) return null;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  const consumeRing = (ring: number[][]) => {
    for (const p of ring) {
      const x = p[0];
      const y = p[1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  };
  if (geom.type === "Polygon") {
    for (const ring of geom.coordinates ?? []) consumeRing(ring);
  } else if (geom.type === "MultiPolygon") {
    for (const poly of geom.coordinates ?? []) {
      for (const ring of poly) consumeRing(ring);
    }
  } else {
    return null;
  }
  if (!Number.isFinite(minX) || minX > maxX) return null;
  return [(minX + maxX) / 2, (minY + maxY) / 2];
}

function scaleRingAbout(cx: number, cy: number, ring: number[][], scale: number): number[][] {
  if (scale === 1) return ring;
  return ring.map((p) => {
    const x = p[0]!;
    const y = p[1]!;
    return [cx + (x - cx) * scale, cy + (y - cy) * scale];
  });
}

function scaleGeometryAbout(geom: any, cx: number, cy: number, scale: number): any {
  if (scale === 1 || !geom) return geom;
  const t = geom.type;
  if (t === "Polygon") {
    return {
      ...geom,
      coordinates: (geom.coordinates as number[][][]).map((ring: number[][]) => scaleRingAbout(cx, cy, ring, scale)),
    };
  }
  if (t === "MultiPolygon") {
    return {
      ...geom,
      coordinates: (geom.coordinates as number[][][][]).map((poly: number[][][]) =>
        poly.map((ring: number[][]) => scaleRingAbout(cx, cy, ring, scale)),
      ),
    };
  }
  return geom;
}

function scaleFeatureCollectionAboutCenters(
  fc: { type: "FeatureCollection"; features: any[] },
  cellById: Map<string, CellRecord>,
  scale: number,
): { type: "FeatureCollection"; features: any[] } {
  if (scale === 1) return fc;
  return {
    type: "FeatureCollection",
    features: fc.features.map((f) => {
      if (!f.geometry) return f;
      const cid = f.properties?.cell_id as string | undefined;
      const cell = typeof cid === "string" ? cellById.get(cid) : undefined;
      let cx: number | undefined;
      let cy: number | undefined;
      if (cell) {
        cx = cell.x;
        cy = cell.y;
      } else {
        const bc = bboxCenterFromGeometry(f.geometry);
        if (bc) [cx, cy] = bc;
      }
      if (cx === undefined || cy === undefined) return f;
      return {
        ...f,
        geometry: scaleGeometryAbout(f.geometry, cx, cy, scale),
      };
    }),
  };
}

function worldBBoxFromViewState(vs: OrthoViewState, width: number, height: number) {
  const o = vs.ortho;
  const vp = new OrthographicViewport({
    id: "ortho",
    width,
    height,
    target: o.target,
    zoom: o.zoom,
    flipY: false,
  });
  const corners = [
    [0, 0],
    [width, 0],
    [0, height],
    [width, height],
  ] as const;
  const xy = corners.map(([px, py]) => vp.unproject([px, py]));
  const xs = xy.map((p) => p[0]);
  const ys = xy.map((p) => p[1]);
  return {
    min_x: Math.min(...xs),
    max_x: Math.max(...xs),
    min_y: Math.min(...ys),
    max_y: Math.max(...ys),
  };
}

/**
 * Outlines are heavy for whole-slide view. We only call /api/polygons when the visible
 * world rectangle is at most this fraction of the full sample (area ratio). Zoom/pan
 * so your region of interest covers less of the tissue to load polygons for that area only.
 */
const POLYGON_MAX_VIEWPORT_AREA_FRACTION = 0.22;

/** Cap transcript glyphs per frame after viewport filter (dense Xenium tiles). */
const MAX_TRANSCRIPTS_DRAW = 120_000;

function isViewportSmallEnoughForPolygons(
  bbox: { min_x: number; min_y: number; max_x: number; max_y: number },
  bounds: SampleBounds,
): boolean {
  const bw = Math.max(1e-9, bbox.max_x - bbox.min_x);
  const bh = Math.max(1e-9, bbox.max_y - bbox.min_y);
  const fullW = Math.max(1e-9, bounds.max_x - bounds.min_x);
  const fullH = Math.max(1e-9, bounds.max_y - bounds.min_y);
  const viewArea = bw * bh;
  const fullArea = fullW * fullH;
  return viewArea / fullArea <= POLYGON_MAX_VIEWPORT_AREA_FRACTION;
}

function filterCellsInViewport(
  cache: ReadonlyMap<string, CellRecord>,
  bbox: { min_x: number; min_y: number; max_x: number; max_y: number } | null,
): CellRecord[] {
  if (!bbox) return [];
  const { min_x, min_y, max_x, max_y } = bbox;
  const out: CellRecord[] = [];
  for (const c of cache.values()) {
    if (c.x >= min_x && c.x <= max_x && c.y >= min_y && c.y <= max_y) out.push(c);
  }
  return out;
}

export function SpatialViewer(props: Props) {
  const cellRadiusScale = props.cellRadiusScale ?? 1;
  const morphWorldBounds = props.morphologyMicronExtent ?? props.bounds;
  const overlayGenes = props.overlayGenes ?? [];
  const overlayOpacityFactor = props.overlayOpacity ?? 0.9;
  const overlayActive = overlayGenes.length > 0;
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [viewState, setViewState] = useState<OrthoViewState>(() =>
    initialOrthoForBounds(props.bounds, 800, 600),
  );

  /**
   * Merged from every /api/cells response for this sample. In centroid mode the cache is filled once
   * with the full sample (no viewport subsampling); pan/zoom only changes which dots are visible.
   */
  const [cellCache, setCellCache] = useState(() => new Map<string, CellRecord>());
  const cellCacheRef = useRef(cellCache);
  cellCacheRef.current = cellCache;
  /** Whether the last full-sample centroid load hit CSO_MAX_CELLS_FULL_LOAD (HUD only). */
  const centroidLoadTruncatedRef = useRef(false);

  const [geojson, setGeojson] = useState<{ type: "FeatureCollection"; features: any[] }>({
    type: "FeatureCollection",
    features: [],
  });
  const [geneMap, setGeneMap] = useState<Record<string, number>>({});
  const [overlayMaps, setOverlayMaps] = useState<Record<string, Record<string, number>>>({});
  const [hoverCell, setHoverCell] = useState<CellRecord | null>(null);
  const [morphTextures, setMorphTextures] = useState<Record<string, HTMLImageElement>>({});
  const [multiChannelTextures, setMultiChannelTextures] = useState<Record<string, HTMLImageElement>>({});
  const [morphTextureMaxEdge, setMorphTextureMaxEdge] = useState<number | null>(null);
  const [dragRect, setDragRect] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(
    null,
  );
  const dragActive = useRef(false);
  /** Freehand lasso in overlay pixel coords (preview mirrors ref so pointer-up sees the full stroke). */
  const [lassoScreenPath, setLassoScreenPath] = useState<Array<{ x: number; y: number }> | null>(null);
  const lassoScreenPathRef = useRef<Array<{ x: number; y: number }>>([]);
  const lassoDraggingRef = useRef(false);
  /** Last /api/cells stats so you can confirm viewport loading vs dataset size */
  const [viewportCellStats, setViewportCellStats] = useState<{
    totalInViewport: number;
    truncated: boolean;
  } | null>(null);
  /** Last /api/polygons stats (polygons mode only) */
  const [viewportPolyStats, setViewportPolyStats] = useState<{
    totalInViewport: number;
    truncated: boolean;
  } | null>(null);
  /** True while /api/cells (and /api/polygons in polygon mode) for the current view are in flight */
  const [viewportLoading, setViewportLoading] = useState(false);
  const [viewportLoadError, setViewportLoadError] = useState<string | null>(null);
  const viewportFetchGen = useRef(0);
  const lodRef = useRef(2);

  const [transcriptTilePoints, setTranscriptTilePoints] = useState<[number, number, string][]>([]);
  const [transcriptLoadError, setTranscriptLoadError] = useState<string | null>(null);
  const [transcriptLoading, setTranscriptLoading] = useState(false);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      setSize({ w: Math.max(320, r.width), h: Math.max(240, r.height) });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    setLassoScreenPath(null);
    lassoScreenPathRef.current = [];
    lassoDraggingRef.current = false;
  }, [props.mode]);

  const bumpZoom = useCallback((delta: number) => {
    setViewState((vs) => ({
      ...vs,
      ortho: {
        ...vs.ortho,
        zoom: Math.min(vs.ortho.maxZoom, Math.max(vs.ortho.minZoom, vs.ortho.zoom + delta)),
      },
    }));
  }, []);

  const resetView = useCallback(() => {
    setViewState(initialOrthoForBounds(props.bounds, size.w, size.h));
  }, [props.bounds, size.h, size.w]);

  useEffect(() => {
    if (!props.showHe) {
      setMorphTextures({});
      setMorphTextureMaxEdge(null);
      return;
    }

    const chans = props.morphologyChannels ?? [];
    if (!chans.length) {
      const im = new Image();
      im.crossOrigin = "anonymous";
      im.onload = () => {
        setMorphTextures({ __legacy__: im });
        setMorphTextureMaxEdge(Math.max(im.naturalWidth, im.naturalHeight));
      };
      im.src = `/api/assets/he/${props.sampleId}`;
      return;
    }

    let cancelled = false;
    const loaders = chans.map(
        (c) =>
          new Promise<[string, HTMLImageElement]>((resolve, reject) => {
            const im = new Image();
            im.crossOrigin = "anonymous";
            im.onload = () => resolve([String(c.id), im]);
            im.onerror = () => reject(new Error(String(c.id)));
            const u = c.url.startsWith("/") || c.url.startsWith("http") ? c.url : `/${c.url}`;
            im.src = u;
          }),
      );

    void Promise.allSettled(loaders).then((results) => {
      if (cancelled) return;
      const map: Record<string, HTMLImageElement> = {};
      let maxEdge = 0;
      for (const r of results) {
        if (r.status === "fulfilled") {
          const [id, im] = r.value;
          map[id] = im;
          maxEdge = Math.max(maxEdge, im.naturalWidth, im.naturalHeight);
        }
      }
      setMorphTextures(map);
      setMorphTextureMaxEdge(maxEdge > 0 ? maxEdge : null);
    });

    return () => {
      cancelled = true;
    };
  }, [props.sampleId, props.showHe, props.morphologyChannels]);

  useEffect(() => {
    if (!props.showMultiChannel) {
      setMultiChannelTextures({});
      return;
    }
    const chans = props.multiChannelChannels ?? [];
    if (!chans.length) {
      setMultiChannelTextures({});
      return;
    }

    let cancelled = false;
    const loaders = chans
      .filter((c) => (props.multiChannelEnabled ?? {})[c.id] !== false)
      .map(
        (c) =>
          new Promise<[string, HTMLImageElement]>((resolve, reject) => {
            const im = new Image();
            im.crossOrigin = "anonymous";
            im.onload = () => resolve([String(c.id), im]);
            im.onerror = () => reject(new Error(String(c.id)));
            const u = c.url.startsWith("/") || c.url.startsWith("http") ? c.url : `/${c.url}`;
            im.src = u;
          }),
      );

    void Promise.allSettled(loaders).then((results) => {
      if (cancelled) return;
      const map: Record<string, HTMLImageElement> = {};
      for (const r of results) {
        if (r.status === "fulfilled") {
          const [id, im] = r.value;
          map[id] = im;
        }
      }
      setMultiChannelTextures(map);
    });

    return () => {
      cancelled = true;
    };
  }, [props.sampleId, props.showMultiChannel, props.multiChannelChannels, props.multiChannelEnabled]);

  const debouncedReload = useDebouncedCallback(
    async (bbox: { min_x: number; min_y: number; max_x: number; max_y: number }) => {
      const gen = ++viewportFetchGen.current;
      setViewportLoadError(null);
      if (props.showCentroids) {
        setGeojson({ type: "FeatureCollection", features: [] });
        setViewportPolyStats(null);
        const inView = filterCellsInViewport(cellCacheRef.current, bbox);
        setViewportCellStats({
          totalInViewport: inView.length,
          truncated: centroidLoadTruncatedRef.current,
        });
        return;
      }

      setViewportLoading(true);
      try {
        const cRes = await api.fetchCells(props.sampleId, bbox);
        if (gen !== viewportFetchGen.current) return;
        setCellCache((prev) => {
          const n = new Map(prev);
          for (const c of cRes.cells) {
            n.set(c.cell_id, c);
          }
          return n;
        });
        setViewportCellStats({
          totalInViewport: cRes.total_in_viewport,
          truncated: Boolean(cRes.truncated),
        });

        const polyAreaOk = isViewportSmallEnoughForPolygons(bbox, props.bounds);
        if (props.showPolygons && polyAreaOk) {
          const pRes = (await api.fetchPolygons(props.sampleId, lodRef.current, bbox)) as {
            features?: unknown[];
            total_in_viewport?: number;
            truncated?: boolean;
          };
          if (gen !== viewportFetchGen.current) return;
          const rawFeats = pRes.features ?? [];
          const feats = rawFeats.map((f) => {
            const x = f as { id: string; geometry: any; properties: any };
            return {
              type: "Feature" as const,
              id: x.id,
              geometry: x.geometry,
              properties: x.properties ?? {},
            };
          });
          setGeojson({ type: "FeatureCollection", features: feats });
          setViewportPolyStats({
            totalInViewport: Number(pRes.total_in_viewport ?? feats.length),
            truncated: Boolean(pRes.truncated),
          });
        } else {
          setGeojson({ type: "FeatureCollection", features: [] });
          setViewportPolyStats(null);
        }
      } catch (e) {
        if (gen !== viewportFetchGen.current) return;
        setViewportLoadError(e instanceof Error ? e.message : String(e));
        setViewportCellStats(null);
        setViewportPolyStats(null);
      } finally {
        if (gen === viewportFetchGen.current) {
          setViewportLoading(false);
        }
      }
    },
    140,
  );

  const reloadViewport = useCallback(() => {
    const bbox = safeWorldBBoxFromViewState(viewState, size.w, size.h);
    if (!bbox) return;
    const z = viewState.ortho.zoom;
    lodRef.current = z > 6 ? 0 : z > 3 ? 1 : 2;
    debouncedReload(bbox);
  }, [debouncedReload, props.bounds, props.sampleId, props.showPolygons, size.h, size.w, viewState]);

  const viewportBBoxForPolyHint = useMemo(
    () => safeWorldBBoxFromViewState(viewState, size.w, size.h),
    [viewState, size.w, size.h],
  );
  const polygonLoadAllowed = useMemo(() => {
    if (!viewportBBoxForPolyHint) return false;
    return isViewportSmallEnoughForPolygons(viewportBBoxForPolyHint, props.bounds);
  }, [viewportBBoxForPolyHint, props.bounds]);

  const visibleCells = useMemo(
    () => filterCellsInViewport(cellCache, viewportBBoxForPolyHint),
    [cellCache, viewportBBoxForPolyHint],
  );

  useEffect(() => {
    setCellCache(new Map());
  }, [props.dataRevision, props.sampleId]);

  useEffect(() => {
    if (!props.showCentroids) return;
    let cancelled = false;
    setViewportLoading(true);
    setViewportLoadError(null);
    void (async () => {
      try {
        const cRes = await api.fetchCells(props.sampleId, props.bounds, { truncate: false });
        if (cancelled) return;
        const n = new Map<string, CellRecord>();
        for (const c of cRes.cells) {
          n.set(c.cell_id, c);
        }
        setCellCache(n);
        centroidLoadTruncatedRef.current = Boolean(cRes.truncated);
        setViewportCellStats({
          totalInViewport: cRes.total_in_viewport,
          truncated: Boolean(cRes.truncated),
        });
      } catch (e) {
        if (cancelled) return;
        setViewportLoadError(e instanceof Error ? e.message : String(e));
        setViewportCellStats(null);
      } finally {
        if (!cancelled) {
          setViewportLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
      setViewportLoading(false);
    };
  }, [
    props.showCentroids,
    props.sampleId,
    props.dataRevision,
    props.bounds.min_x,
    props.bounds.min_y,
    props.bounds.max_x,
    props.bounds.max_y,
  ]);

  useEffect(() => {
    reloadViewport();
  }, [reloadViewport, props.sampleId]);

  useEffect(() => {
    reloadViewport();
  }, [props.dataRevision, reloadViewport]);

  /* Load full gene columns once per gene (not per viewport). Expression maps are by cell_id;
   * the scatter layer uses cached cells in the current view. */
  useEffect(() => {
    if (!overlayGenes.length) {
      setOverlayMaps({});
      return;
    }
    let cancelled = false;
    void (async () => {
      const entries = await Promise.all(
        overlayGenes.map(async (g) => {
          const res = await api.fetchGeneExpression(props.sampleId, g);
          return [g, res.values ?? {}] as const;
        }),
      );
      if (cancelled) return;
      const next: Record<string, Record<string, number>> = {};
      for (const [g, vals] of entries) next[g] = vals;
      setOverlayMaps(next);
    })();
    return () => {
      cancelled = true;
    };
  }, [overlayGenes, props.sampleId]);

  useEffect(() => {
    if (props.paintMode !== "gene") {
      setGeneMap({});
      return;
    }
    if (!props.gene) {
      setGeneMap({});
      return;
    }
    let cancelled = false;
    void (async () => {
      const res = await api.fetchGeneExpression(props.sampleId, props.gene);
      if (cancelled) return;
      setGeneMap(res.values ?? {});
    })();
    return () => {
      cancelled = true;
    };
  }, [props.paintMode, props.gene, props.sampleId]);

  const scaleMode = props.continuousScaleMode ?? "full";

  /** Viridis range from the current viewport (respects continuous scale mode). */
  const exprRange = useMemo(() => {
    const vals: number[] = [];
    for (const c of visibleCells) {
      const v = geneMap[c.cell_id];
      if (typeof v === "number" && Number.isFinite(v)) vals.push(v);
    }
    return valueRangeForContinuousScale(vals, scaleMode);
  }, [geneMap, visibleCells, scaleMode]);

  /** Viridis range for continuous metadata (viewport cells + scale mode). */
  const metadataNumericRange = useMemo(() => {
    if (props.paintMode !== "metadata" || props.metadataColumnKind !== "numeric") {
      return { vmin: 0, vmax: 1 };
    }
    const key = props.metadataColumn || "cell_type";
    const vals: number[] = [];
    for (const c of visibleCells) {
      const v = parseMetadataNumeric(c.metadata[key]);
      if (v !== null && Number.isFinite(v)) vals.push(v);
    }
    return valueRangeForContinuousScale(vals, scaleMode);
  }, [props.paintMode, props.metadataColumnKind, props.metadataColumn, visibleCells, scaleMode]);

  const overlayRanges = useMemo(() => {
    const out: Record<string, { vmin: number; vmax: number }> = {};
    for (const g of overlayGenes) {
      const m = overlayMaps[g] ?? {};
      const vals: number[] = [];
      for (const c of visibleCells) {
        const v = m[c.cell_id];
        if (typeof v === "number" && Number.isFinite(v)) vals.push(v);
      }
      out[g] = valueRangeForContinuousScale(vals, scaleMode);
    }
    return out;
  }, [overlayGenes, overlayMaps, visibleCells, scaleMode]);

  const cellById = useMemo(() => new Map([...cellCache.values()].map((c) => [c.cell_id, c])), [cellCache]);

  const scaledGeojson = useMemo(
    () => scaleFeatureCollectionAboutCenters(geojson, cellById, cellRadiusScale),
    [geojson, cellById, cellRadiusScale],
  );

  const layers = useMemo(() => {
    const geneVmin = exprRange.vmin;
    const geneVmax = exprRange.vmax;
    const metaVmin = metadataNumericRange.vmin;
    const metaVmax = metadataNumericRange.vmax;
    const zoom = viewState.ortho.zoom;
    const ls: unknown[] = [];
    const flipM = props.morphologyFlipY ?? false;
    const transM = props.morphologyTranslateUm ?? [0, 0];
    const morphBounds = morphologyBitmapBounds(morphWorldBounds, flipM, transM);

    if (props.showHe && Object.keys(morphTextures).length > 0) {
      const mc = props.morphologyChannels ?? [];
      if (morphTextures.__legacy__) {
        ls.push(
          new BitmapLayer({
            id: "he",
            image: morphTextures.__legacy__,
            bounds: morphBounds,
            opacity: props.heOpacity,
            coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          }),
        );
      } else {
        for (const c of mc) {
          const tex = morphTextures[String(c.id)];
          if (!tex) continue;
          ls.push(
            new BitmapLayer({
              id: `morph-${c.id}`,
              image: tex,
              bounds: morphBounds,
              opacity: props.heOpacity,
              coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
            }),
          );
        }
      }
    }

    const mcLayerOpacity = props.multiChannelOpacity ?? 0.78;
    if (props.showMultiChannel && Object.keys(multiChannelTextures).length > 0) {
      const mcc = props.multiChannelChannels ?? [];
      for (const c of mcc) {
        if ((props.multiChannelEnabled ?? {})[c.id] === false) continue;
        const tex = multiChannelTextures[String(c.id)];
        if (!tex) continue;
        ls.push(
          new BitmapLayer({
            id: `seg-ch-${c.id}`,
            image: tex,
            bounds: morphBounds,
            opacity: mcLayerOpacity,
            coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          }),
        );
      }
    }

    if (props.showPolygons && scaledGeojson.features.length) {
      const outlinePx = Math.max(1, 2.2 * cellRadiusScale);
      const basePolyOpacity = overlayActive ? props.polygonOpacity * 0.25 : props.polygonOpacity;

      ls.push(
        new GeoJsonLayer({
          id: "polygons-fill",
          data: scaledGeojson,
          opacity: basePolyOpacity,
          stroked: false,
          filled: true,
          getFillColor: (f: { properties?: { cell_id?: string } }) => {
            const cid = f.properties?.cell_id;
            if (typeof cid === "string" && props.selectedIds.has(cid)) {
              return [255, 255, 120, 255];
            }
            const rec = typeof cid === "string" ? cellById.get(cid) : undefined;
            if (!rec) return [80, 80, 90, 210];
            if (props.paintMode === "gene") {
              const v = geneMap[rec.cell_id];
              return colorFromExpression(v ?? NaN, geneVmin, geneVmax);
            }
            const key = props.metadataColumn || "cell_type";
            const raw = rec.metadata[key];
            if (props.metadataColumnKind === "numeric") {
              const v = parseMetadataNumeric(raw);
              return colorFromExpression(v ?? NaN, metaVmin, metaVmax);
            }
            return colorForMetadataKey(key, raw);
          },
          pickable: props.mode === "navigate",
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          updateTriggers: {
            getFillColor: [
              props.paintMode,
              props.metadataColumn,
              props.metadataColumnKind,
              props.selectedIds,
              geneMap,
              geneVmin,
              geneVmax,
              metaVmin,
              metaVmax,
              overlayActive,
              cellById,
            ],
          },
        }),
      );

      if (overlayActive) {
        const og = overlayGenes;
        ls.push(
          new GeoJsonLayer({
            id: "polygons-overlay-genes",
            data: scaledGeojson,
            opacity: props.polygonOpacity * overlayOpacityFactor,
            stroked: false,
            filled: true,
            getFillColor: (f: { properties?: { cell_id?: string } }) => {
              const cid = f.properties?.cell_id;
              if (typeof cid !== "string") return [0, 0, 0, 0];
              const [r, g, b, a] = overlayRgbForCell(cid, og, overlayMaps, overlayRanges);
              return [r, g, b, a] as [number, number, number, number];
            },
            pickable: false,
            coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
            updateTriggers: {
              getFillColor: [overlayGenes, overlayMaps, overlayRanges, overlayOpacityFactor],
            },
          }),
        );
      }

      const outlines = outlinePathsFromGeoJSON(scaledGeojson);
      if (outlines.length) {
        ls.push(
          new PathLayer({
            id: `polygons-stroke-${outlinePx.toFixed(2)}`,
            data: outlines,
            coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
            getPath: (d: { path: [number, number][] }) => d.path,
            getColor: [26, 32, 42, Math.min(255, Math.round(140 + 80 * props.polygonOpacity))],
            getWidth: outlinePx,
            widthUnits: "pixels",
            opacity: props.polygonOpacity,
            pickable: false,
            capRounded: true,
            jointRounded: true,
            updateTriggers: { getWidth: [cellRadiusScale] },
          }),
        );
      }
    }

    if (props.showCentroids && visibleCells.length) {
      const baseR = Math.max(1.1, Math.min(5, 1.7 + zoom * 0.12));
      const rPix = Math.max(0.5, baseR * cellRadiusScale);
      const baseAlpha = overlayActive ? props.centroidOpacity * 0.25 : props.centroidOpacity;
      ls.push(
        new ScatterplotLayer({
          id: "cells",
          data: visibleCells,
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          getPosition: (d: CellRecord) => [d.x, d.y, 0],
          getRadius: rPix,
          radiusUnits: "pixels",
          radiusMinPixels: Math.min(32, Math.max(0.5, 1 * cellRadiusScale)),
          radiusMaxPixels: Math.min(48, Math.max(2, 6 * cellRadiusScale)),
          stroked: true,
          filled: true,
          getFillColor: (d: CellRecord) => {
            const selected = props.selectedIds.has(d.cell_id) ? 1 : 0;
            if (props.paintMode === "gene") {
              const v = geneMap[d.cell_id];
              const c = colorFromExpression(v ?? NaN, geneVmin, geneVmax);
              if (selected) return [255, 255, 120, 255] as [number, number, number, number];
              return c;
            }
            const key = props.metadataColumn || "cell_type";
            const raw = d.metadata[key];
            const c =
              props.metadataColumnKind === "numeric"
                ? colorFromExpression(parseMetadataNumeric(raw) ?? NaN, metaVmin, metaVmax)
                : colorForMetadataKey(key, raw);
            if (selected) return [255, 255, 120, 255] as [number, number, number, number];
            return c;
          },
          getLineColor: [15, 20, 30, 220],
          getLineWidth: 1,
          lineWidthUnits: "pixels",
          pickable: props.mode === "navigate",
          opacity: baseAlpha,
          updateTriggers: {
            getFillColor: [
              props.paintMode,
              props.metadataColumn,
              props.metadataColumnKind,
              props.selectedIds,
              geneMap,
              geneVmin,
              geneVmax,
              metaVmin,
              metaVmax,
              overlayActive,
              visibleCells,
            ],
            getRadius: [zoom, cellRadiusScale],
          },
        }),
      );
    }

    if (props.showCentroids && visibleCells.length && overlayActive) {
      const baseR = Math.max(1.1, Math.min(5, 1.7 + zoom * 0.12));
      const rPix = Math.max(0.5, baseR * cellRadiusScale);
      const og = overlayGenes;
      ls.push(
        new ScatterplotLayer({
          id: "cells-overlay-genes",
          data: visibleCells,
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          getPosition: (d: CellRecord) => [d.x, d.y, 0],
          getRadius: rPix,
          radiusUnits: "pixels",
          radiusMinPixels: Math.min(32, Math.max(0.5, 1 * cellRadiusScale)),
          radiusMaxPixels: Math.min(48, Math.max(2, 6 * cellRadiusScale)),
          stroked: false,
          filled: true,
          getFillColor: (d: CellRecord) => {
            const [r, g, b, a] = overlayRgbForCell(d.cell_id, og, overlayMaps, overlayRanges);
            return [r, g, b, a] as [number, number, number, number];
          },
          pickable: false,
          opacity: props.centroidOpacity * overlayOpacityFactor,
          updateTriggers: {
            getFillColor: [overlayGenes, overlayMaps, overlayRanges, overlayOpacityFactor],
            getRadius: [zoom, cellRadiusScale],
          },
        }),
      );
    }

    return ls as any;
  }, [
    visibleCells,
    exprRange,
    metadataNumericRange,
    geneMap,
    scaledGeojson,
    cellById,
    morphTextures,
    multiChannelTextures,
    props.morphologyChannels,
    props.showMultiChannel,
    props.multiChannelChannels,
    props.multiChannelEnabled,
    props.multiChannelOpacity,
    props.bounds,
    morphWorldBounds,
    props.centroidOpacity,
    props.metadataColumn,
    props.metadataColumnKind,
    props.paintMode,
    props.gene,
    props.heOpacity,
    props.mode,
    props.polygonOpacity,
    props.selectedIds,
    props.showCentroids,
    props.showHe,
    props.showPolygons,
    cellRadiusScale,
    viewState.ortho.zoom,
    props.morphologyFlipY,
    props.morphologyTranslateUm,
    overlayActive,
    overlayGenes,
    overlayMaps,
    overlayOpacityFactor,
    overlayRanges,
  ]);

  const unproject = (px: number, py: number) => {
    const vp = new OrthographicViewport({
      id: "ortho",
      width: size.w,
      height: size.h,
      target: viewState.ortho.target,
      zoom: viewState.ortho.zoom,
      flipY: false,
    });
    const w = vp.unproject([px, py]);
    return [w[0], w[1]] as [number, number];
  };

  const onOverlayPointerDown = (e: React.PointerEvent) => {
    if (props.mode === "navigate") return;
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (props.mode === "select_rect") {
      dragActive.current = true;
      setDragRect({ x0: x, y0: y, x1: x, y1: y });
    } else if (props.mode === "select_lasso") {
      lassoDraggingRef.current = true;
      const start = [{ x, y }];
      lassoScreenPathRef.current = start;
      setLassoScreenPath(start);
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    }
  };

  const onOverlayPointerMove = (e: React.PointerEvent) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (props.mode === "select_rect") {
      if (!dragActive.current) return;
      setDragRect((dr) => (dr ? { ...dr, x1: x, y1: y } : dr));
      return;
    }

    if (props.mode !== "select_lasso" || !lassoDraggingRef.current) return;
    setLassoScreenPath((prev) => {
      if (!prev?.length) return prev;
      const last = prev[prev.length - 1];
      if (Math.hypot(x - last.x, y - last.y) < LASSO_MIN_SAMPLE_PX) return prev;
      const next = [...prev, { x, y }];
      lassoScreenPathRef.current = next;
      return next;
    });
  };

  const onOverlayPointerUp = (e: React.PointerEvent) => {
    if (props.mode === "select_rect" && dragActive.current && dragRect) {
      const x0 = Math.min(dragRect.x0, dragRect.x1);
      const x1 = Math.max(dragRect.x0, dragRect.x1);
      const y0 = Math.min(dragRect.y0, dragRect.y1);
      const y1 = Math.max(dragRect.y0, dragRect.y1);
      const c0 = unproject(x0, y1);
      const c1 = unproject(x1, y0);
      const box = {
        min_x: Math.min(c0[0], c1[0]),
        max_x: Math.max(c0[0], c1[0]),
        min_y: Math.min(c0[1], c1[1]),
        max_y: Math.max(c0[1], c1[1]),
      };
      const picked = visibleCells
        .filter((c) => c.x >= box.min_x && c.x <= box.max_x && c.y >= box.min_y && c.y <= box.max_y)
        .map((c) => c.cell_id);
      props.onToggleSelected(picked, !e.shiftKey);
      setDragRect(null);
    } else if (props.mode === "select_lasso" && lassoDraggingRef.current) {
      const rectUp = containerRef.current?.getBoundingClientRect();
      let path = lassoScreenPathRef.current;
      if (rectUp?.width != null && path.length) {
        const x = e.clientX - rectUp.left;
        const y = e.clientY - rectUp.top;
        const last = path[path.length - 1];
        if (Math.hypot(x - last.x, y - last.y) >= 0.5) {
          path = [...path, { x, y }];
          lassoScreenPathRef.current = path;
        }
      }
      if (path.length >= LASSO_MIN_VERTICES) {
        const ringWorld: [number, number][] = path.map((p) => unproject(p.x, p.y));
        const picked = visibleCells
          .filter((c) => pointInPolygonMicrons(c.x, c.y, ringWorld))
          .map((c) => c.cell_id);
        props.onToggleSelected(picked, !e.shiftKey);
      }
    }

    if (props.mode === "select_lasso" && lassoDraggingRef.current) {
      try {
        (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
      } catch {
        /* already released */
      }
    }

    lassoDraggingRef.current = false;
    lassoScreenPathRef.current = [];
    setLassoScreenPath(null);
    dragActive.current = false;
    setDragRect(null);
  };

  const deckViewState = useMemo(
    () => ({
      ...viewState,
      ortho: {
        ...viewState.ortho,
        width: size.w,
        height: size.h,
      },
    }),
    [size.h, size.w, viewState],
  );

  return (
    <div
      ref={containerRef}
      style={{ position: "relative", width: "100%", height: "100%", overflow: "hidden" }}
    >
      <DeckGL
        width={size.w}
        height={size.h}
        views={new OrthographicView({ id: "ortho", flipY: false })}
        viewState={deckViewState}
        parameters={{
          clearColor: [15 / 255, 18 / 255, 22 / 255, 1],
        }}
        controller={{
          dragPan: props.mode === "navigate",
          scrollZoom: true,
          doubleClickZoom: false,
        }}
        layers={layers}
        onViewStateChange={({ viewState: vs }) => {
          setViewState((prev) => normalizeOrthoViewState(vs, prev, size.w, size.h));
        }}
        onHover={(info) => {
          const raw = info.object as unknown;
          if (raw && typeof raw === "object") {
            const o = raw as Record<string, unknown>;
            if (typeof o.cell_id === "string" && "x" in o && "metadata" in o) {
              setHoverCell(raw as CellRecord);
              return;
            }
            const props = o.properties as Record<string, unknown> | undefined;
            const cid = props?.cell_id;
            if (typeof cid === "string") {
              setHoverCell(cellById.get(cid) ?? null);
              return;
            }
          }
          setHoverCell(null);
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 0,
          pointerEvents: props.mode === "navigate" ? "none" : "auto",
          cursor:
            props.mode === "select_rect" || props.mode === "select_lasso" ? "crosshair" : "default",
        }}
        onPointerDown={onOverlayPointerDown}
        onPointerMove={onOverlayPointerMove}
        onPointerUp={onOverlayPointerUp}
      />
      {dragRect && (
        <div
          style={{
            position: "absolute",
            left: Math.min(dragRect.x0, dragRect.x1),
            top: Math.min(dragRect.y0, dragRect.y1),
            width: Math.abs(dragRect.x1 - dragRect.x0),
            height: Math.abs(dragRect.y1 - dragRect.y0),
            border: "1px solid rgba(255,255,120,0.9)",
            background: "rgba(255,255,120,0.12)",
            pointerEvents: "none",
          }}
        />
      )}
      {lassoScreenPath && lassoScreenPath.length >= 2 ? (
        <svg
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
            zIndex: 1,
          }}
        >
          <polyline
            fill="none"
            stroke="rgba(255,255,120,0.95)"
            strokeWidth={1.75}
            strokeLinejoin="round"
            strokeLinecap="round"
            points={lassoScreenPath.map((p) => `${p.x},${p.y}`).join(" ")}
          />
          <line
            x1={lassoScreenPath[lassoScreenPath.length - 1].x}
            y1={lassoScreenPath[lassoScreenPath.length - 1].y}
            x2={lassoScreenPath[0].x}
            y2={lassoScreenPath[0].y}
            stroke="rgba(255,255,120,0.45)"
            strokeWidth={1}
            strokeDasharray="5 4"
          />
        </svg>
      ) : null}
      <div
        style={{
          position: "absolute",
          left: 10,
          bottom: 10,
          background: "rgba(0,0,0,0.55)",
          padding: "8px 10px",
          borderRadius: 8,
          fontSize: 12,
          pointerEvents: "none",
          maxWidth: "70%",
        }}
      >
        <div style={{ opacity: 0.85 }}>
          {viewportLoading ? (
            <span style={{ color: "#a8d4ff" }}>
              {props.showPolygons && polygonLoadAllowed
                ? "Loading cells & cell outlines…"
                : props.showCentroids
                  ? "Loading all cell centroids for this sample…"
                  : "Loading cells…"}{" "}
              <span style={{ opacity: 0.75 }}>(the API caches data after the first read in each process)</span>
            </span>
          ) : null}
          {viewportLoadError ? (
            <span style={{ color: "#ff9090", display: "block", marginBottom: 6 }}>
              Load failed: {viewportLoadError}
            </span>
          ) : null}
          {morphTextureMaxEdge != null &&
          morphTextureMaxEdge > MORPH_TEXTURE_WARN_EDGE &&
          props.showHe ? (
            <div style={{ color: "#ffb070", marginBottom: 6, lineHeight: 1.45 }}>
              Large morphology texture ({morphTextureMaxEdge}px long edge). Some GPUs may be slow or hit WebGL limits;
              re-export with a smaller --morphology-max-long-edge if needed.
            </div>
          ) : null}
          {!viewportLoading && props.showPolygons && !polygonLoadAllowed ? (
            <div style={{ color: "#ffb070", marginBottom: 6, lineHeight: 1.45 }}>
              Cell outlines: zoom in so the visible area is a smaller part of the slide (≤
              {Math.round(POLYGON_MAX_VIEWPORT_AREA_FRACTION * 100)}% of sample area). Only that region
              then loads outlines — full overview stays fast.
            </div>
          ) : null}
          {!viewportLoading ? (
            <>
              Cells drawn: {visibleCells.length}
              {cellCache.size > 0 ? (
                <span style={{ opacity: 0.7 }}>
                  {" "}
                  · {cellCache.size.toLocaleString()} in memory
                  {props.showCentroids ? " (full sample for centroids)" : " (pan/zoom merges new areas)"}
                </span>
              ) : null}
              {viewportCellStats != null ? (
                <>
                  {" "}
                  · {viewportCellStats.totalInViewport.toLocaleString()} in view rectangle
                  {viewportCellStats.truncated ? (
                    <span style={{ color: "#ffb070" }}>
                      {" "}
                      · cells capped (
                      {props.showCentroids ? "CSO_MAX_CELLS_FULL_LOAD" : "CSO_MAX_CELLS_PER_VIEWPORT"})
                    </span>
                  ) : null}
                </>
              ) : null}
              {props.showPolygons && viewportPolyStats != null ? (
                <>
                  {" "}
                  · {viewportPolyStats.totalInViewport.toLocaleString()} outlines in view
                  {viewportPolyStats.truncated ? (
                    <span style={{ color: "#ffb070" }}> · outlines capped (CSO_MAX_POLYGONS_PER_VIEWPORT)</span>
                  ) : null}
                </>
              ) : null}{" "}
              · LOD: {lodRef.current}
            </>
          ) : null}
          {hoverCell ? (
            <>
              {" · "}
              <strong>{hoverCell.cell_id}</strong>
              {typeof hoverCell.metadata.cell_type === "string"
                ? ` · type: ${hoverCell.metadata.cell_type}`
                : ""}
              {typeof hoverCell.metadata.pathology_region === "string"
                ? ` · ROI: ${hoverCell.metadata.pathology_region}`
                : ""}
            </>
          ) : (
            " · Hover a cell (Navigate mode)"
          )}
        </div>
        <div style={{ opacity: 0.65, marginTop: 4 }}>
          Scroll/pinch to zoom · Drag to pan (Navigate) · Rectangle or lasso select for ROI labels · Empty view?{" "}
          <span style={{ opacity: 0.9 }}>Reset</span> (top-right)
        </div>
      </div>

      <div
        style={{
          position: "absolute",
          top: 10,
          right: 10,
          display: "flex",
          flexDirection: "column",
          gap: 6,
          alignItems: "stretch",
          zIndex: 2,
        }}
      >
        <div style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
          <button
            type="button"
            title="Zoom in"
            onClick={() => bumpZoom(0.6)}
            style={{ padding: "6px 10px", cursor: "pointer" }}
          >
            +
          </button>
          <button
            type="button"
            title="Zoom out"
            onClick={() => bumpZoom(-0.6)}
            style={{ padding: "6px 10px", cursor: "pointer" }}
          >
            −
          </button>
          <button type="button" title="Fit sample bounds" onClick={resetView} style={{ padding: "6px 10px", cursor: "pointer" }}>
            Reset view
          </button>
        </div>
      </div>

      {overlayActive ? (
        <div
          style={{
            position: "absolute",
            right: 10,
            bottom:
              overlayActive &&
              (props.paintMode === "gene" ||
                (props.paintMode === "metadata" && props.metadataColumnKind === "numeric"))
                ? 118
                : 10,
            width: 220,
            background: "rgba(0,0,0,0.55)",
            padding: "8px 10px",
            borderRadius: 8,
            fontSize: 11,
            pointerEvents: "none",
          }}
        >
          <div style={{ opacity: 0.9, marginBottom: 6 }}>Overlay expression</div>
          <div style={{ opacity: 0.8, lineHeight: 1.4 }}>
            {overlayGenes.length === 1
              ? `${overlayGenes[0]} (viridis · ${continuousScaleModeLabel(scaleMode)})`
              : overlayGenes.length === 2
                ? `R: ${overlayGenes[0]} · G: ${overlayGenes[1]}`
                : `R: ${overlayGenes[0]} · G: ${overlayGenes[1]} · B: ${overlayGenes[2]}`}
          </div>
          {overlayGenes.map((g) => (
            <div key={g} style={{ opacity: 0.7, fontSize: 10, marginTop: 4 }}>
              {g}: [{overlayRanges[g]?.vmin.toPrecision(3)}, {overlayRanges[g]?.vmax.toPrecision(3)}]
            </div>
          ))}
        </div>
      ) : null}

      {props.paintMode === "gene" ? (
        <div
          style={{
            position: "absolute",
            right: 10,
            bottom: 10,
            width: 200,
            background: "rgba(0,0,0,0.55)",
            padding: "8px 10px",
            borderRadius: 8,
            fontSize: 11,
            pointerEvents: "none",
          }}
        >
          <div style={{ opacity: 0.9, marginBottom: 6 }}>Gene: {props.gene} (viridis)</div>
          <div
            style={{
              height: 10,
              borderRadius: 4,
              background: VIRIDIS_GRADIENT_CSS,
            }}
          />
          <div style={{ display: "flex", justifyContent: "space-between", opacity: 0.8, marginTop: 4 }}>
            <span>{exprRange.vmin.toPrecision(4)}</span>
            <span>{exprRange.vmax.toPrecision(4)}</span>
          </div>
          <div style={{ opacity: 0.65, marginTop: 4 }}>
            Scale: {continuousScaleModeLabel(scaleMode)} · viewport cells
          </div>
        </div>
      ) : props.paintMode === "metadata" && props.metadataColumnKind === "numeric" ? (
        <div
          style={{
            position: "absolute",
            right: 10,
            bottom: 10,
            width: 220,
            background: "rgba(0,0,0,0.55)",
            padding: "8px 10px",
            borderRadius: 8,
            fontSize: 11,
            pointerEvents: "none",
          }}
        >
          <div style={{ opacity: 0.9, marginBottom: 6 }}>
            Metadata: {props.metadataColumn} <span style={{ opacity: 0.75 }}>(continuous · viridis)</span>
          </div>
          <div
            style={{
              height: 10,
              borderRadius: 4,
              background: VIRIDIS_GRADIENT_CSS,
            }}
          />
          <div style={{ display: "flex", justifyContent: "space-between", opacity: 0.8, marginTop: 4 }}>
            <span>{metadataNumericRange.vmin.toPrecision(4)}</span>
            <span>{metadataNumericRange.vmax.toPrecision(4)}</span>
          </div>
          <div style={{ opacity: 0.65, marginTop: 4 }}>
            Scale: {continuousScaleModeLabel(scaleMode)} · viewport cells
          </div>
        </div>
      ) : null}
    </div>
  );
}
