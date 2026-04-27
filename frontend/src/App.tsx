import { useCallback, useEffect, useMemo, useState } from "react";

function useMutuallyExclusivePair(
  initialA: boolean,
  initialB: boolean,
): [
  boolean,
  boolean,
  (v: boolean) => void,
  (v: boolean) => void,
] {
  const [a, setA] = useState(initialA);
  const [b, setB] = useState(initialB);
  const setFirst = useCallback((v: boolean) => {
    setA(v);
    if (v) setB(false);
  }, []);
  const setSecond = useCallback((v: boolean) => {
    setB(v);
    if (v) setA(false);
  }, []);
  return [a, b, setFirst, setSecond];
}

import * as api from "@/api";
import { OverlayGeneExpression } from "@/components/OverlayGeneExpression";
import { SpatialViewer, type InteractionMode } from "@/components/SpatialViewer";
import { UmapPanel, type UmapPoint } from "@/components/UmapPanel";
import type { Manifest, SampleBounds, SampleRef } from "@/types";

export function App() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [sampleId, setSampleId] = useState<string>("sample_demo");
  const [bounds, setBounds] = useState<SampleBounds | null>(null);

  const [mode, setMode] = useState<InteractionMode>("navigate");
  const [showCentroids, showPolygons, setShowCentroids, setShowPolygons] = useMutuallyExclusivePair(true, false);
  const [showHe, setShowHe] = useState(true);
  const [heOpacity, setHeOpacity] = useState(0.35);
  const [centroidOpacity, setCentroidOpacity] = useState(1);
  const [polygonOpacity, setPolygonOpacity] = useState(0.55);

  const [paintMode, setPaintMode] = useState<"metadata" | "gene">("metadata");
  const [metadataColumn, setMetadataColumn] = useState("cell_type");
  const [metadataColumns, setMetadataColumns] = useState<string[]>(["cell_type"]);
  const [gene, setGene] = useState("EPCAM");
  const [geneFilter, setGeneFilter] = useState("");
  const [availableGenes, setAvailableGenes] = useState<string[]>([]);
  const DEFAULT_CELL_RADIUS = 1;
  const [cellRadiusScale, setCellRadiusScale] = useState(DEFAULT_CELL_RADIUS);

  const [overlayGenes, setOverlayGenes] = useState<string[]>([]);
  const [overlayOpacity, setOverlayOpacity] = useState(0.9);
  const [dataRevision, setDataRevision] = useState(0);

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  const [umapPoints, setUmapPoints] = useState<UmapPoint[]>([]);

  const [draftPoly, setDraftPoly] = useState<[number, number][] | null>(null);
  const [pendingRing, setPendingRing] = useState<[number, number][] | null>(null);

  const [annoLabel, setAnnoLabel] = useState("necrosis");
  const [annoAuthor, setAnnoAuthor] = useState("pathologist.demo");
  const [annoNotes, setAnnoNotes] = useState("");
  const [annoStatus, setAnnoStatus] = useState<string | null>(null);
  /** Short-lived banner after POST /annotations succeeds */
  const [roiSaveBanner, setRoiSaveBanner] = useState<string | null>(null);

  useEffect(() => {
    if (!roiSaveBanner) return;
    const tid = window.setTimeout(() => setRoiSaveBanner(null), 14000);
    return () => window.clearTimeout(tid);
  }, [roiSaveBanner]);

  useEffect(() => {
    void (async () => {
      try {
        const m = await api.fetchManifest();
        setManifest(m);
        const first = m.samples[0];
        if (first) {
          setSampleId(first.sample_id);
          setBounds(first.bounds);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  useEffect(() => {
    void (async () => {
      try {
        const payload = await api.fetchPlot(sampleId, "umap");
        const pts = (payload.payload.points ?? []) as UmapPoint[];
        setUmapPoints(pts);
      } catch {
        setUmapPoints([]);
      }
    })();
  }, [sampleId]);

  const loadColorColumns = useCallback(async (sid: string) => {
    try {
      const c = await api.fetchColorColumns(sid);
      setMetadataColumns(c.metadata_columns.length ? c.metadata_columns : ["cell_type"]);
      setAvailableGenes(c.genes ?? []);
      setGene((g) => (c.genes?.length && !c.genes.includes(g) ? c.genes[0]! : g));
      setMetadataColumn((prev) =>
        c.metadata_columns?.includes(prev)
          ? prev
          : (c.metadata_columns[0] ?? "cell_type"),
      );
    } catch {
      setMetadataColumns(["cell_type"]);
      setAvailableGenes([]);
    }
  }, []);

  useEffect(() => {
    void loadColorColumns(sampleId);
  }, [sampleId, loadColorColumns]);

  const samples = manifest?.samples ?? [];

  const filteredGenes = useMemo(() => {
    const q = geneFilter.trim().toLowerCase();
    const base = availableGenes.length ? availableGenes : [gene];
    if (!q) return base;
    return base.filter((g) => g.toLowerCase().includes(q));
  }, [availableGenes, gene, geneFilter]);

  const onToggleSelected = useCallback((ids: string[], replace: boolean) => {
    setSelectedIds((prev) => {
      const next = replace ? new Set<string>() : new Set(prev);
      for (const id of ids) next.add(id);
      return next;
    });
  }, []);

  const onPickUmapCell = useCallback((cellId: string, additive: boolean) => {
    setSelectedIds((prev) => {
      const next = additive ? new Set(prev) : new Set<string>();
      next.add(cellId);
      return next;
    });
  }, []);

  const refreshColumnsAfterRoi = useCallback(
    async (labelForManualCol: string) => {
      try {
        const c = await api.fetchColorColumns(sampleId);
        setMetadataColumns(c.metadata_columns.length ? c.metadata_columns : ["cell_type"]);
        setAvailableGenes(c.genes ?? []);
        const manualCol = `$ManualAnno:${labelForManualCol.trim()}`;
        if (c.metadata_columns.includes(manualCol)) {
          setMetadataColumn(manualCol);
          setPaintMode("metadata");
        }
      } catch {
        /* ignore */
      }
    },
    [sampleId],
  );

  const submitAnnotation = async () => {
    if (!pendingRing || pendingRing.length < 4) {
      setAnnoStatus("Close a polygon first (polygon mode → vertices → Enter).");
      return;
    }
    setAnnoStatus("Saving…");
    setRoiSaveBanner(null);
    try {
      const geometry: GeoJSON.Polygon = {
        type: "Polygon",
        coordinates: [pendingRing.map(([x, y]) => [x, y])],
      };
      const res = await api.postAnnotation({
        sample_id: sampleId,
        label: annoLabel,
        author: annoAuthor,
        notes: annoNotes,
        geometry,
        confidence: 1.0,
      });
      const manualCol = `$ManualAnno:${annoLabel.trim()}`;
      const n = res.cells_assigned ?? 0;
      setAnnoStatus(`Done — ${n} cells tagged.`);
      setRoiSaveBanner(
        `ROI saved successfully. ${n} cells assigned to label "${res.label}". ` +
          `Column "${manualCol}" was written to cell_metadata.parquet on the server.`,
      );
      setPendingRing(null);
      setDataRevision((k) => k + 1);
      await refreshColumnsAfterRoi(annoLabel);
    } catch (e) {
      setAnnoStatus(e instanceof Error ? e.message : String(e));
      setRoiSaveBanner(null);
    }
  };

  const submitSelectionAnnotation = async () => {
    if (selectedIds.size === 0) {
      setAnnoStatus("Select cells first (rectangle mode, or pick on UMAP), then save.");
      return;
    }
    setAnnoStatus("Saving…");
    setRoiSaveBanner(null);
    try {
      const res = await api.postAnnotationByCells({
        sample_id: sampleId,
        label: annoLabel,
        author: annoAuthor,
        notes: annoNotes,
        cell_ids: [...selectedIds],
        confidence: 1.0,
      });
      const manualCol = `$ManualAnno:${annoLabel.trim()}`;
      const n = res.cells_assigned ?? 0;
      setAnnoStatus(`Done — ${n} cells tagged.`);
      setRoiSaveBanner(
        `Selection saved. ${n} cells assigned to "${res.label}". Column "${manualCol}" in cell_metadata.parquet.`,
      );
      setDataRevision((k) => k + 1);
      await refreshColumnsAfterRoi(annoLabel);
    } catch (e) {
      setAnnoStatus(e instanceof Error ? e.message : String(e));
      setRoiSaveBanner(null);
    }
  };

  const composition = useMemo(() => {
    const byType: Record<string, number> = {};
    for (const p of umapPoints) {
      const t = String(p.cell_type ?? "unknown");
      byType[t] = (byType[t] ?? 0) + 1;
    }
    return byType;
  }, [umapPoints]);

  if (error) {
    return (
      <div style={{ padding: 24 }}>
        <h2>Failed to load</h2>
        <pre style={{ whiteSpace: "pre-wrap" }}>{error}</pre>
        <p style={{ opacity: 0.75 }}>
          Start the API (`docker compose up`) or run `uvicorn` locally with generated data in `./data`.
        </p>
      </div>
    );
  }

  if (!bounds) {
    return (
      <div style={{ padding: 24 }}>
        <p>Loading manifest…</p>
      </div>
    );
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(0, 1fr) minmax(280px, 380px)",
        height: "100%",
        minHeight: 0,
        overflow: "hidden",
        width: "100%",
      }}
    >
      <div style={{ display: "grid", gridTemplateRows: "auto 1fr", minHeight: 0, overflow: "hidden" }}>
        <Toolbar
          samples={samples}
          sampleId={sampleId}
          onSampleChange={(id) => {
            setSampleId(id);
            const s = samples.find((x) => x.sample_id === id);
            if (s) setBounds(s.bounds);
            setSelectedIds(new Set());
            setGeneFilter("");
            setOverlayGenes([]);
          }}
          mode={mode}
          onMode={setMode}
          showCentroids={showCentroids}
          setShowCentroids={setShowCentroids}
          showPolygons={showPolygons}
          setShowPolygons={setShowPolygons}
          showHe={showHe}
          setShowHe={setShowHe}
          heOpacity={heOpacity}
          setHeOpacity={setHeOpacity}
          centroidOpacity={centroidOpacity}
          setCentroidOpacity={setCentroidOpacity}
          polygonOpacity={polygonOpacity}
          setPolygonOpacity={setPolygonOpacity}
          paintMode={paintMode}
          setPaintMode={setPaintMode}
          metadataColumn={metadataColumn}
          setMetadataColumn={setMetadataColumn}
          metadataColumns={metadataColumns}
          gene={gene}
          setGene={setGene}
          geneFilter={geneFilter}
          setGeneFilter={setGeneFilter}
          filteredGenes={filteredGenes}
          cellRadiusScale={cellRadiusScale}
          setCellRadiusScale={setCellRadiusScale}
          defaultCellRadiusScale={DEFAULT_CELL_RADIUS}
          selectedCount={selectedIds.size}
          onClearSelection={() => setSelectedIds(new Set())}
        />
        <div style={{ position: "relative", minHeight: 0 }}>
          <SpatialViewer
            key={sampleId}
            sampleId={sampleId}
            bounds={bounds}
            mode={mode}
            showCentroids={showCentroids}
            showPolygons={showPolygons}
            showHe={showHe}
            heOpacity={heOpacity}
            centroidOpacity={centroidOpacity}
            polygonOpacity={polygonOpacity}
            paintMode={paintMode}
            metadataColumn={metadataColumn}
            gene={gene}
            selectedIds={selectedIds}
            onToggleSelected={onToggleSelected}
            onDraftPolygon={(ring) => setDraftPoly(ring)}
            onPolygonClosed={(ring) => setPendingRing(ring)}
            cellRadiusScale={cellRadiusScale}
            overlayGenes={overlayGenes}
            overlayOpacity={overlayOpacity}
            dataRevision={dataRevision}
          />
        </div>
      </div>

      <aside
        style={{
          borderLeft: "1px solid rgba(255,255,255,0.08)",
          background: "#12161d",
          height: "100%",
          minHeight: 0,
          overflowY: "auto",
          overflowX: "hidden",
          position: "relative",
          zIndex: 10,
          isolation: "isolate",
        }}
      >
        <div style={{ padding: 12, borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
          <div style={{ fontWeight: 650, marginBottom: 8 }}>Linked UMAP</div>
          <div style={{ height: 180 }}>
            <UmapPanel points={umapPoints} selectedIds={selectedIds} onPickCell={onPickUmapCell} />
          </div>
        </div>

        <div style={{ padding: 12, borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
          <div style={{ fontWeight: 650, marginBottom: 8 }}>Selection-linked composition</div>
          <div style={{ fontSize: 13, opacity: 0.85, lineHeight: 1.5 }}>
            {Object.entries(composition).map(([k, v]) => (
              <div key={k}>
                {k}: <span style={{ opacity: 0.75 }}>{v}</span>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 8, fontSize: 12, opacity: 0.65 }}>
            Plot is precomputed; highlighting reflects current selection IDs only (subset shown when linked queries are
            added).
          </div>
        </div>

        <div style={{ padding: 12, borderBottom: "1px solid rgba(255,255,255,0.08)", minHeight: 0 }}>
          <div style={{ fontWeight: 650, marginBottom: 8 }}>Available genes ({availableGenes.length})</div>
          <div
            style={{
              maxHeight: 160,
              overflow: "auto",
              fontSize: 12,
              lineHeight: 1.55,
              opacity: 0.92,
              fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
            }}
          >
            {availableGenes.length ? (
              <ul style={{ margin: 0, paddingLeft: 18 }}>
                {availableGenes.map((g) => (
                  <li key={g}>{g}</li>
                ))}
              </ul>
            ) : (
              <span style={{ opacity: 0.65 }}>No genes (needs expression-wide.parquet for this sample).</span>
            )}
          </div>
        </div>

        <OverlayGeneExpression
          availableGenes={availableGenes}
          overlayGenes={overlayGenes}
          onChangeOverlayGenes={setOverlayGenes}
          overlayOpacity={overlayOpacity}
          onOverlayOpacity={setOverlayOpacity}
        />

        <div style={{ padding: 12, borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
          <div style={{ fontWeight: 650, marginBottom: 8 }}>ROI annotation</div>
          {roiSaveBanner ? (
            <div
              role="status"
              style={{
                marginBottom: 10,
                padding: "10px 12px",
                borderRadius: 8,
                border: "1px solid rgba(80, 200, 120, 0.55)",
                background: "rgba(40, 90, 55, 0.35)",
                color: "#d8f5e0",
                fontSize: 13,
                lineHeight: 1.45,
              }}
            >
              {roiSaveBanner}
            </div>
          ) : null}
          <div style={{ display: "grid", gap: 8, fontSize: 13 }}>
            <label style={{ display: "grid", gap: 4 }}>
              Label
              <input value={annoLabel} onChange={(e) => setAnnoLabel(e.target.value)} />
            </label>
            <label style={{ display: "grid", gap: 4 }}>
              Author
              <input value={annoAuthor} onChange={(e) => setAnnoAuthor(e.target.value)} />
            </label>
            <label style={{ display: "grid", gap: 4 }}>
              Notes
              <input value={annoNotes} onChange={(e) => setAnnoNotes(e.target.value)} />
            </label>
            <button type="button" onClick={() => void submitAnnotation()}>
              Persist polygon ROI + assign cells
            </button>
            <button type="button" onClick={() => void submitSelectionAnnotation()}>
              Persist current selection + assign cells
            </button>
            {pendingRing ? (
              <div style={{ opacity: 0.75 }}>
                Pending polygon: {pendingRing.length} vertices (closed ring includes duplicate first point).
              </div>
            ) : (
              <div style={{ opacity: 0.65 }}>Polygon mode: click vertices, Enter closes.</div>
            )}
            {draftPoly && draftPoly.length ? (
              <div style={{ opacity: 0.65 }}>Draft vertices: {draftPoly.length}</div>
            ) : null}
            {annoStatus ? <div style={{ opacity: 0.85 }}>{annoStatus}</div> : null}
          </div>
        </div>

        <div style={{ padding: 12, overflow: "auto", fontSize: 12, opacity: 0.75 }}>
          <div style={{ fontWeight: 650, marginBottom: 8, opacity: 0.9 }}>Architecture notes</div>
          <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.55 }}>
            <li>Viewport-filtered cell / polygon loads (server-side bbox).</li>
            <li>Orthographic deck.gl + Bitmap H&E underlay (pyramids can swap to OpenSeadragon).</li>
            <li>Gene expression fetched per gene with column subset (Parquet), client cache.</li>
            <li>ROI persistence in PostgreSQL with derived cell tags (non-destructive vs on-disk metadata).</li>
          </ul>
        </div>
      </aside>
    </div>
  );
}

function Toolbar(props: {
  samples: SampleRef[];
  sampleId: string;
  onSampleChange: (id: string) => void;
  mode: InteractionMode;
  onMode: (m: InteractionMode) => void;
  showCentroids: boolean;
  setShowCentroids: (v: boolean) => void;
  showPolygons: boolean;
  setShowPolygons: (v: boolean) => void;
  showHe: boolean;
  setShowHe: (v: boolean) => void;
  heOpacity: number;
  setHeOpacity: (v: number) => void;
  centroidOpacity: number;
  setCentroidOpacity: (v: number) => void;
  polygonOpacity: number;
  setPolygonOpacity: (v: number) => void;
  paintMode: "metadata" | "gene";
  setPaintMode: (v: "metadata" | "gene") => void;
  metadataColumn: string;
  setMetadataColumn: (v: string) => void;
  metadataColumns: string[];
  gene: string;
  setGene: (v: string) => void;
  geneFilter: string;
  setGeneFilter: (v: string) => void;
  filteredGenes: string[];
  cellRadiusScale: number;
  setCellRadiusScale: (v: number) => void;
  defaultCellRadiusScale: number;
  selectedCount: number;
  onClearSelection: () => void;
}) {
  const geneOptions = Array.from(new Set([props.gene, ...props.filteredGenes].filter(Boolean)));

  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: 10,
        alignItems: "center",
        padding: 10,
        borderBottom: "1px solid rgba(255,255,255,0.08)",
        background: "#141a22",
      }}
    >
      <strong style={{ marginRight: 6 }}>CSO SpatialVis</strong>

      <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
        Sample
        <select value={props.sampleId} onChange={(e) => props.onSampleChange(e.target.value)}>
          {(props.samples.length ? props.samples : [{ sample_id: props.sampleId, title: props.sampleId } as any]).map(
            (s) => (
              <option key={s.sample_id} value={s.sample_id}>
                {s.title ?? s.sample_id}
              </option>
            ),
          )}
        </select>
      </label>

      <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
        Mode
        <select value={props.mode} onChange={(e) => props.onMode(e.target.value as InteractionMode)}>
          <option value="navigate">Navigate</option>
          <option value="select_rect">Rectangle select</option>
          <option value="annotate_polygon">Polygon ROI</option>
        </select>
      </label>

      <label style={{ display: "flex", gap: 8, alignItems: "center", opacity: props.showCentroids ? 1 : 0.45 }}>
        <input type="checkbox" checked={props.showCentroids} onChange={(e) => props.setShowCentroids(e.target.checked)} />
        Centroids
      </label>
      <label style={{ display: "flex", gap: 8, alignItems: "center", opacity: props.showPolygons ? 1 : 0.45 }}>
        <input type="checkbox" checked={props.showPolygons} onChange={(e) => props.setShowPolygons(e.target.checked)} />
        Polygons
      </label>
      <span style={{ fontSize: 11, opacity: 0.55 }}>(centroids and polygons are exclusive)</span>
      <label style={{ display: "flex", gap: 8, alignItems: "center", opacity: props.showHe ? 1 : 0.45 }}>
        <input type="checkbox" checked={props.showHe} onChange={(e) => props.setShowHe(e.target.checked)} />
        H&E
      </label>

      <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
        H&E opacity
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={props.heOpacity}
          onChange={(e) => props.setHeOpacity(Number(e.target.value))}
        />
      </label>
      <label
        style={{
          display: "flex",
          gap: 8,
          alignItems: "center",
          opacity: props.showCentroids ? 1 : 0.4,
        }}
      >
        Cells α
        <input
          type="range"
          min={0.05}
          max={1}
          step={0.05}
          disabled={!props.showCentroids}
          value={props.centroidOpacity}
          onChange={(e) => props.setCentroidOpacity(Number(e.target.value))}
        />
      </label>
      <label
        style={{
          display: "flex",
          gap: 8,
          alignItems: "center",
          opacity: props.showPolygons ? 1 : 0.4,
        }}
      >
        Poly α
        <input
          type="range"
          min={0.05}
          max={1}
          step={0.05}
          disabled={!props.showPolygons}
          value={props.polygonOpacity}
          onChange={(e) => props.setPolygonOpacity(Number(e.target.value))}
        />
      </label>

      <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
        Color
        <select
          value={props.paintMode}
          onChange={(e) => props.setPaintMode(e.target.value as "metadata" | "gene")}
        >
          <option value="metadata">Metadata column</option>
          <option value="gene">Gene expression (viridis)</option>
        </select>
      </label>

      <label style={{ display: "flex", gap: 8, alignItems: "center", opacity: props.paintMode === "metadata" ? 1 : 0.45 }}>
        Column
        <select
          value={
            props.metadataColumns.includes(props.metadataColumn)
              ? props.metadataColumn
              : props.metadataColumns[0] ?? ""
          }
          disabled={props.paintMode !== "metadata"}
          onChange={(e) => props.setMetadataColumn(e.target.value)}
          style={{ minWidth: 180, maxWidth: 260 }}
          title="cell_metadata.parquet columns including $ManualAnno:label after ROI save"
        >
          {props.metadataColumns.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </label>

      <label
        style={{
          display: "flex",
          gap: 8,
          alignItems: "center",
          opacity: props.paintMode === "gene" ? 1 : 0.45,
        }}
      >
        Gene
        <input
          placeholder="Filter…"
          value={props.geneFilter}
          disabled={props.paintMode !== "gene"}
          onChange={(e) => props.setGeneFilter(e.target.value)}
          style={{ width: 68 }}
          title="Filter gene list"
        />
        <select
          value={geneOptions.includes(props.gene) ? props.gene : geneOptions[0] ?? ""}
          disabled={props.paintMode !== "gene"}
          onChange={(e) => props.setGene(e.target.value)}
          style={{ minWidth: 120 }}
          title="Expression column from expression-wide.parquet"
        >
          {geneOptions.map((g) => (
            <option key={g} value={g}>
              {g}
            </option>
          ))}
        </select>
      </label>

      <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
        Cell size
        <input
          type="range"
          min={0.25}
          max={4}
          step={0.05}
          value={props.cellRadiusScale}
          onChange={(e) => props.setCellRadiusScale(Number(e.target.value))}
          title="Scales centroid marker size (centroids on) or polygon outline weight (polygons on)"
        />
        <button
          type="button"
          title="Restore default size scale"
          onClick={() => props.setCellRadiusScale(props.defaultCellRadiusScale)}
        >
          Default
        </button>
      </label>

      <button type="button" onClick={props.onClearSelection}>
        Clear selection ({props.selectedCount})
      </button>
    </div>
  );
}
