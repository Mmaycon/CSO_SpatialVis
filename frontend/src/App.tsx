import { useCallback, useEffect, useMemo, useRef, useState } from "react";

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
import type { ContinuousScaleMode } from "@/colors";
import type { Manifest, SampleBounds, SampleRef } from "@/types";
import { parseMorphologyFrame } from "@/types";
import { useDebouncedCallback } from "@/hooks/useDebouncedCallback";

function parseImageTranslateUm(img: unknown): [number, number] {
  if (!img || typeof img !== "object") return [0, 0];
  const al = (img as Record<string, unknown>).alignment;
  if (!al || typeof al !== "object") return [0, 0];
  const tu = (al as Record<string, unknown>).translate_um;
  if (!Array.isArray(tu) || tu.length !== 2) return [0, 0];
  const dx = Number(tu[0]);
  const dy = Number(tu[1]);
  if (!Number.isFinite(dx) || !Number.isFinite(dy)) return [0, 0];
  return [dx, dy];
}

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
  const [metadataColumnKinds, setMetadataColumnKinds] = useState<Record<string, "numeric" | "categorical">>({});
  const [continuousScaleMode, setContinuousScaleMode] = useState<ContinuousScaleMode>("full");
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
      const rawKinds = c.metadata_column_kinds ?? {};
      const kinds: Record<string, "numeric" | "categorical"> = {};
      for (const [k, v] of Object.entries(rawKinds)) {
        kinds[k] = v === "numeric" ? "numeric" : "categorical";
      }
      setMetadataColumnKinds(kinds);
    } catch {
      setMetadataColumns(["cell_type"]);
      setAvailableGenes([]);
      setMetadataColumnKinds({});
    }
  }, []);

  useEffect(() => {
    void loadColorColumns(sampleId);
  }, [sampleId, loadColorColumns]);

  const samples = manifest?.samples ?? [];

  const morphChannels = useMemo(() => {
    const s = samples.find((x) => x.sample_id === sampleId);
    const img = s?.image;
    if (!img || typeof img !== "object") return [];
    const raw = (img as Record<string, unknown>).channels;
    if (!Array.isArray(raw)) return [];
    const out: { id: string; label: string; url: string; default_visible?: boolean }[] = [];
    for (const x of raw) {
      if (!x || typeof x !== "object") continue;
      const o = x as Record<string, unknown>;
      const id = o.id;
      const url = o.url;
      if (typeof id !== "string" || typeof url !== "string") continue;
      const label = typeof o.label === "string" ? o.label : id;
      out.push({
        id,
        label,
        url,
        default_visible: typeof o.default_visible === "boolean" ? o.default_visible : undefined,
      });
    }
    return out;
  }, [samples, sampleId]);

  const morphologyFrame = useMemo(() => {
    const s = samples.find((x) => x.sample_id === sampleId);
    const img = s?.image;
    if (!img || typeof img !== "object") return null;
    return parseMorphologyFrame((img as Record<string, unknown>).morphology_frame);
  }, [samples, sampleId]);

  const morphologyMicronExtent = useMemo(() => morphologyFrame?.crop_microns ?? null, [morphologyFrame]);

  const registrationMeta = useMemo(() => {
    const s = samples.find((x) => x.sample_id === sampleId);
    const img = s?.image;
    if (!img || typeof img !== "object") return { channelId: null as string | null, referenceUrl: null as string | null };
    const o = img as Record<string, unknown>;
    return {
      channelId: typeof o.registration_channel_id === "string" ? o.registration_channel_id : null,
      referenceUrl: typeof o.registration_reference_url === "string" ? o.registration_reference_url : null,
    };
  }, [samples, sampleId]);

  const [registrationChannelId, setRegistrationChannelId] = useState<string>("");

  useEffect(() => {
    if (!morphChannels.length) {
      setRegistrationChannelId("");
      return;
    }
    const exported = registrationMeta.channelId;
    const matchExported = exported && morphChannels.some((c) => c.id === exported);
    const want =
      (matchExported ? exported : null) ??
      morphChannels.find((c) => c.label.toLowerCase().includes("dapi"))?.id ??
      morphChannels[0]!.id;
    setRegistrationChannelId(want);
  }, [sampleId, morphChannels, registrationMeta.channelId]);

  /** Single reference plane for the Morphology underlay (registration channel picker). */
  const morphologyChannelsSingle = useMemo(() => {
    if (!morphChannels.length) return [];
    const rid = registrationChannelId || morphChannels[0]!.id;
    const c = morphChannels.find((x) => x.id === rid);
    return c ? [c] : [morphChannels[0]!];
  }, [morphChannels, registrationChannelId]);

  /** All OME-derived segmentation PNGs (2+ channels) — separate Multi-channel overlay, not blended into Morphology. */
  const multiChannelAvailable = morphChannels.length >= 2;

  const [showMultiChannel, setShowMultiChannel] = useState(false);
  const [multiChannelOpacity, setMultiChannelOpacity] = useState(0.78);
  const [multiChannelEnabled, setMultiChannelEnabled] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const init: Record<string, boolean> = {};
    for (const c of morphChannels) {
      init[String(c.id)] = c.default_visible !== false;
    }
    setMultiChannelEnabled(init);
  }, [sampleId, morphChannels]);

  const toggleMultiChannel = useCallback((id: string) => {
    setMultiChannelEnabled((prev) => {
      const c = morphChannels.find((x) => x.id === id);
      const base = c?.default_visible !== false;
      const cur = prev[id] ?? base;
      return { ...prev, [id]: !cur };
    });
  }, [morphChannels]);

  const morphologyFlipY = useMemo(() => {
    const s = samples.find((x) => x.sample_id === sampleId);
    const img = s?.image;
    if (!img || typeof img !== "object") return false;
    const al = (img as Record<string, unknown>).alignment;
    if (!al || typeof al !== "object") return false;
    return (al as Record<string, unknown>).flip_y === true;
  }, [samples, sampleId]);

  const sampleIdRef = useRef(sampleId);
  sampleIdRef.current = sampleId;

  const [morphTranslateUm, setMorphTranslateUm] = useState<[number, number]>([0, 0]);
  const [morphShiftError, setMorphShiftError] = useState<string | null>(null);

  useEffect(() => {
    const s = samples.find((x) => x.sample_id === sampleId);
    setMorphTranslateUm(parseImageTranslateUm(s?.image ?? null));
    setMorphShiftError(null);
  }, [samples, sampleId]);

  const persistMorphShift = useCallback(async (sid: string, dx: number, dy: number) => {
    try {
      const m = await api.patchSampleImageTranslate(sid, { translate_um: [dx, dy] });
      if (sid !== sampleIdRef.current) return;
      setManifest(m);
      setMorphShiftError(null);
    } catch (e) {
      if (sid !== sampleIdRef.current) return;
      setMorphShiftError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const debouncedPersistMorphShift = useDebouncedCallback(persistMorphShift, 450);

  const updateMorphShift = useCallback(
    (dx: number, dy: number) => {
      setMorphTranslateUm([dx, dy]);
      debouncedPersistMorphShift(sampleId, dx, dy);
    },
    [sampleId, debouncedPersistMorphShift],
  );

  const filteredGenes = useMemo(() => {
    const q = geneFilter.trim().toLowerCase();
    const base = availableGenes.length ? availableGenes : [gene];
    if (!q) return base;
    return base.filter((g) => g.toLowerCase().includes(q));
  }, [availableGenes, gene, geneFilter]);

  const metadataColumnKind = useMemo(
    (): "numeric" | "categorical" =>
      metadataColumnKinds[metadataColumn] === "numeric" ? "numeric" : "categorical",
    [metadataColumnKinds, metadataColumn],
  );

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
    async (trimmedLabel: string) => {
      try {
        const c = await api.fetchColorColumns(sampleId);
        setMetadataColumns(c.metadata_columns.length ? c.metadata_columns : ["cell_type"]);
        setAvailableGenes(c.genes ?? []);
        const rawKinds = c.metadata_column_kinds ?? {};
        const kinds: Record<string, "numeric" | "categorical"> = {};
        for (const [k, v] of Object.entries(rawKinds)) {
          kinds[k] = v === "numeric" ? "numeric" : "categorical";
        }
        setMetadataColumnKinds(kinds);
        const manualCol = `$ManualAnno:${trimmedLabel}`;
        if (c.metadata_columns.includes(manualCol)) {
          setMetadataColumn(manualCol);
          setPaintMode("metadata");
        }
      } catch {
        /* ignore — column list / genes refresh failed (network or API); coloring dropdown may be stale */
      }
    },
    [sampleId],
  );

  const submitSelectionAnnotation = async () => {
    const labelTrimmed = annoLabel.trim();
    if (!labelTrimmed) {
      setAnnoStatus("Enter a non-empty label before saving.");
      return;
    }
    if (selectedIds.size === 0) {
      setAnnoStatus("Select cells first (rectangle or lasso mode, or pick on UMAP), then save.");
      return;
    }
    setAnnoStatus("Saving…");
    setRoiSaveBanner(null);
    try {
      const res = await api.postAnnotationByCells({
        sample_id: sampleId,
        label: labelTrimmed,
        author: annoAuthor,
        notes: annoNotes,
        cell_ids: [...selectedIds],
        confidence: 1.0,
      });
      const manualCol = `$ManualAnno:${labelTrimmed}`;
      const n = res.cells_assigned ?? 0;
      setAnnoStatus(`Done — ${n} cells tagged.`);
      setRoiSaveBanner(
        `Annotation saved. ${n} cells assigned to "${res.label}". Column "${manualCol}" in cell_metadata.parquet.`,
      );
      setDataRevision((k) => k + 1);
      void refreshColumnsAfterRoi(labelTrimmed);
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
          morphChannels={morphChannels}
          multiChannelAvailable={multiChannelAvailable}
          showMultiChannel={showMultiChannel}
          setShowMultiChannel={setShowMultiChannel}
          multiChannelOpacity={multiChannelOpacity}
          setMultiChannelOpacity={setMultiChannelOpacity}
          multiChannelEnabled={multiChannelEnabled}
          toggleMultiChannel={toggleMultiChannel}
          registrationChannelId={registrationChannelId}
          setRegistrationChannelId={setRegistrationChannelId}
          registrationExportChannelId={registrationMeta.channelId}
          registrationReferenceUrl={registrationMeta.referenceUrl}
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
          metadataColumnKinds={metadataColumnKinds}
          continuousScaleMode={continuousScaleMode}
          setContinuousScaleMode={setContinuousScaleMode}
          overlayGeneCount={overlayGenes.length}
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
          morphTranslateUm={morphTranslateUm}
          onMorphShiftChange={updateMorphShift}
          morphShiftError={morphShiftError}
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
            metadataColumnKind={metadataColumnKind}
            continuousScaleMode={continuousScaleMode}
            gene={gene}
            selectedIds={selectedIds}
            onToggleSelected={onToggleSelected}
            cellRadiusScale={cellRadiusScale}
            overlayGenes={overlayGenes}
            overlayOpacity={overlayOpacity}
            dataRevision={dataRevision}
            morphologyChannels={morphChannels.length ? morphologyChannelsSingle : undefined}
            showMultiChannel={showMultiChannel}
            multiChannelChannels={multiChannelAvailable ? morphChannels : undefined}
            multiChannelEnabled={multiChannelEnabled}
            multiChannelOpacity={multiChannelOpacity}
            morphologyFlipY={morphologyFlipY}
            morphologyTranslateUm={morphTranslateUm}
            morphologyMicronExtent={morphologyMicronExtent}
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
            <button type="button" onClick={() => void submitSelectionAnnotation()}>
              Save annotation (assign label to selection)
            </button>
            <div style={{ opacity: 0.65 }}>
              Select cells on the map (<strong>Rectangle</strong> or <strong>Lasso</strong> mode) or click the UMAP; then
              save. One selection set — whatever is selected gets the label.
            </div>
            {annoStatus ? <div style={{ opacity: 0.85 }}>{annoStatus}</div> : null}
          </div>
        </div>

        <div style={{ padding: 12, overflow: "auto", fontSize: 12, opacity: 0.75 }}>
          <div style={{ fontWeight: 650, marginBottom: 8, opacity: 0.9 }}>Architecture notes</div>
          <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.55 }}>
            <li>Viewport-filtered cell / polygon loads (server-side bbox).</li>
            <li>Orthographic deck.gl + morphology image underlay (multi-channel PNGs from ``Images/`` when present).</li>
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
  morphChannels: { id: string; label: string; url: string; default_visible?: boolean }[];
  multiChannelAvailable: boolean;
  showMultiChannel: boolean;
  setShowMultiChannel: (v: boolean) => void;
  multiChannelOpacity: number;
  setMultiChannelOpacity: (v: number) => void;
  multiChannelEnabled: Record<string, boolean>;
  toggleMultiChannel: (id: string) => void;
  registrationChannelId: string;
  setRegistrationChannelId: (id: string) => void;
  registrationExportChannelId: string | null;
  registrationReferenceUrl: string | null;
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
  metadataColumnKinds: Record<string, "numeric" | "categorical">;
  continuousScaleMode: ContinuousScaleMode;
  setContinuousScaleMode: (v: ContinuousScaleMode) => void;
  /** Used to show scale controls when overlay genes are on (viridis scales still apply to overlays). */
  overlayGeneCount: number;
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
  morphTranslateUm: [number, number];
  onMorphShiftChange: (dx: number, dy: number) => void;
  morphShiftError: string | null;
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
          <option value="select_lasso">Lasso select</option>
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
        Morphology
      </label>

      {props.morphChannels.length > 0 ? (
        <label
          style={{
            display: "flex",
            gap: 8,
            alignItems: "center",
            fontSize: 12,
            opacity: props.showHe ? 1 : 0.45,
          }}
          title="Offline registration: pick morphology channel (default DAPI from export). Export writes Images/registration_reference.png for the chosen --registration-channel."
        >
          Registration channel
          <select
            value={
              props.morphChannels.some((c) => c.id === props.registrationChannelId)
                ? props.registrationChannelId
                : props.morphChannels[0]!.id
            }
            disabled={!props.showHe}
            onChange={(e) => props.setRegistrationChannelId(e.target.value)}
          >
            {props.morphChannels.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
      ) : null}

      {props.morphChannels.length > 0 && props.registrationReferenceUrl ? (
        <span style={{ fontSize: 11, opacity: 0.62, maxWidth: 480, lineHeight: 1.35 }}>
          Registration PNG on disk:{" "}
          <code style={{ fontSize: 10 }}>Images/registration_reference.png</code>
          {props.registrationExportChannelId &&
          props.registrationChannelId === props.registrationExportChannelId
            ? " — matches export default channel."
            : " — export used a different channel; re-run export with --registration-channel to refresh the PNG."}
        </span>
      ) : null}

      <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
        Morphology opacity
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={props.heOpacity}
          onChange={(e) => props.setHeOpacity(Number(e.target.value))}
        />
      </label>

      {props.multiChannelAvailable ? (
        <>
          <label
            style={{
              display: "flex",
              gap: 8,
              alignItems: "center",
              fontSize: 12,
              borderLeft: "1px solid rgba(255,255,255,0.12)",
              paddingLeft: 10,
            }}
            title="All OME-TIFF segmentation planes exported as Images/*.png — independent from the Morphology reference layer."
          >
            <input
              type="checkbox"
              checked={props.showMultiChannel}
              onChange={(e) => props.setShowMultiChannel(e.target.checked)}
            />
            Multi-channel
          </label>
          <label style={{ display: "flex", gap: 8, alignItems: "center", opacity: props.showMultiChannel ? 1 : 0.45 }}>
            Multi α
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              disabled={!props.showMultiChannel}
              value={props.multiChannelOpacity}
              onChange={(e) => props.setMultiChannelOpacity(Number(e.target.value))}
            />
          </label>
          <span
            style={{
              display: "inline-flex",
              flexWrap: "wrap",
              gap: 8,
              alignItems: "center",
              fontSize: 12,
              opacity: props.showMultiChannel ? 1 : 0.45,
              maxWidth: 560,
            }}
            title="Toggle each exported segmentation channel (same µm bounds as morphology crop)."
          >
            {props.morphChannels.map((c) => (
              <label key={`mc-${c.id}`} style={{ display: "inline-flex", gap: 4, alignItems: "center", cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={props.multiChannelEnabled[c.id] ?? c.default_visible !== false}
                  disabled={!props.showMultiChannel}
                  onChange={() => props.toggleMultiChannel(c.id)}
                />
                <span>{c.label}</span>
              </label>
            ))}
          </span>
        </>
      ) : null}

      <span
        style={{
          display: "inline-flex",
          flexDirection: "column",
          gap: 4,
          fontSize: 12,
          opacity: props.showHe ? 1 : 0.45,
          minWidth: 210,
        }}
        title="Moves morphology underlay in µm; centroids/polygons stay put. Values persist to data/manifest.json (debounced)."
      >
        <span style={{ opacity: 0.88 }}>Morphology shift (µm)</span>
        <span style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <label style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
            Δx
            <input
              type="number"
              step={0.1}
              disabled={!props.showHe}
              value={props.morphTranslateUm[0]}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (!Number.isFinite(v)) return;
                props.onMorphShiftChange(v, props.morphTranslateUm[1]);
              }}
              style={{ width: 76 }}
            />
          </label>
          <label style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
            Δy
            <input
              type="number"
              step={0.1}
              disabled={!props.showHe}
              value={props.morphTranslateUm[1]}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (!Number.isFinite(v)) return;
                props.onMorphShiftChange(props.morphTranslateUm[0], v);
              }}
              style={{ width: 76 }}
            />
          </label>
          <button
            type="button"
            disabled={!props.showHe}
            onClick={() => props.onMorphShiftChange(0, 0)}
            style={{ fontSize: 11, padding: "2px 8px", cursor: "pointer" }}
          >
            Reset
          </button>
        </span>
        {props.morphShiftError ? (
          <span style={{ color: "#ff9090", fontSize: 11 }}>{props.morphShiftError}</span>
        ) : (
          <span style={{ opacity: 0.55, fontSize: 10 }}>Saved to manifest automatically.</span>
        )}
      </span>

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
              {props.metadataColumnKinds[c] === "numeric" ? " · continuous" : ""}
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

      {(props.paintMode === "gene" ||
        (props.paintMode === "metadata" && props.metadataColumnKinds[props.metadataColumn] === "numeric") ||
        props.overlayGeneCount > 0) && (
        <label
          style={{
            display: "flex",
            gap: 8,
            alignItems: "center",
            fontSize: 12,
          }}
          title="Viridis endpoints use cells in the current view. Percentiles trim outliers so the colormap uses the bulk of the distribution."
        >
          Continuous scale
          <select
            value={props.continuousScaleMode}
            onChange={(e) => props.setContinuousScaleMode(e.target.value as ContinuousScaleMode)}
            style={{ maxWidth: 220 }}
          >
            <option value="full">Min–max (viewport)</option>
            <option value="p01_p99">1st–99th percentile</option>
            <option value="p05_p95">5th–95th percentile</option>
          </select>
        </label>
      )}

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
