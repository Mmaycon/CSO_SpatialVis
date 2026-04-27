import { useEffect, useMemo, useState } from "react";

const MAX_OVERLAY = 3;

type Props = {
  availableGenes: string[];
  overlayGenes: string[];
  onChangeOverlayGenes: (genes: string[]) => void;
  overlayOpacity: number;
  onOverlayOpacity: (v: number) => void;
};

export function OverlayGeneExpression(props: Props) {
  const [filter, setFilter] = useState("");
  const [picked, setPicked] = useState("");

  const addable = useMemo(() => {
    const q = filter.trim().toLowerCase();
    return props.availableGenes.filter(
      (g) =>
        !props.overlayGenes.includes(g) &&
        props.overlayGenes.length < MAX_OVERLAY &&
        (!q || g.toLowerCase().includes(q)),
    );
  }, [filter, props.availableGenes, props.overlayGenes]);

  useEffect(() => {
    if (picked && !addable.includes(picked)) {
      setPicked("");
    }
  }, [addable, picked]);

  const canAddMore = props.overlayGenes.length < MAX_OVERLAY && addable.length > 0;

  const addGene = () => {
    const g = picked.trim();
    if (!g || props.overlayGenes.length >= MAX_OVERLAY || props.overlayGenes.includes(g)) return;
    if (!addable.includes(g)) return;
    props.onChangeOverlayGenes([...props.overlayGenes, g]);
    setPicked("");
    setFilter("");
  };

  const removeGene = (g: string) => {
    props.onChangeOverlayGenes(props.overlayGenes.filter((x) => x !== g));
  };

  const atCapacity = props.overlayGenes.length >= MAX_OVERLAY;

  return (
    <div
      style={{
        padding: 12,
        borderBottom: "1px solid rgba(255,255,255,0.08)",
        fontSize: 13,
        position: "relative",
        zIndex: 1,
      }}
    >
      <div style={{ fontWeight: 650, marginBottom: 6 }}>Overlay gene expression</div>
      <p
        style={{
          opacity: 0.72,
          margin: "0 0 10px",
          fontSize: 11,
          lineHeight: 1.45,
        }}
        title="One gene: viridis (same idea as Color → Gene). Two or three genes: mapped to R, G, B channels using viewport min/max per gene."
      >
        Stack up to {MAX_OVERLAY} genes on centroids · 1 = viridis · 2–3 = RGB (each scaled to viewport min/max).
      </p>

      <div style={{ display: "grid", gap: 8 }}>
        <label style={{ display: "grid", gap: 4 }}>
          <span style={{ opacity: 0.8, fontSize: 11 }}>Filter list</span>
          <input
            placeholder="Type to filter…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            autoComplete="off"
            spellCheck={false}
            disabled={atCapacity || !props.availableGenes.length}
            style={{
              width: "100%",
              padding: "6px 8px",
              fontSize: 13,
              borderRadius: 6,
              border: "1px solid rgba(255,255,255,0.14)",
              background: "rgba(0,0,0,0.25)",
              color: "inherit",
              boxSizing: "border-box",
            }}
          />
        </label>

        <div style={{ display: "flex", gap: 8, alignItems: "stretch", flexWrap: "wrap" }}>
          <select
            aria-label="Gene to add to overlay"
            value={picked}
            onChange={(e) => setPicked(e.target.value)}
            disabled={atCapacity || !canAddMore}
            style={{
              flex: "1 1 120px",
              minWidth: 0,
              padding: "6px 8px",
              fontSize: 13,
              borderRadius: 6,
              border: "1px solid rgba(255,255,255,0.14)",
              background: "#1a2230",
              color: "inherit",
              cursor: atCapacity ? "not-allowed" : "pointer",
            }}
          >
            <option value="">
              {atCapacity ? "Max genes" : !props.availableGenes.length ? "No genes loaded" : "Choose gene…"}
            </option>
            {addable.map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={addGene}
            disabled={atCapacity || !picked}
            style={{
              padding: "6px 14px",
              fontSize: 13,
              fontWeight: 600,
              borderRadius: 6,
              border: "1px solid rgba(120,180,255,0.45)",
              background: picked && !atCapacity ? "rgba(70,110,190,0.35)" : "rgba(255,255,255,0.06)",
              color: "inherit",
              cursor: atCapacity || !picked ? "not-allowed" : "pointer",
              flexShrink: 0,
            }}
          >
            Add
          </button>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 6,
          marginTop: 10,
          marginBottom: 10,
          minHeight: 26,
          alignItems: "center",
        }}
      >
        {props.overlayGenes.length === 0 ? (
          <span style={{ opacity: 0.55, fontSize: 12 }}>No overlay genes yet — pick one and click Add.</span>
        ) : (
          props.overlayGenes.map((g, i) => (
            <span
              key={g}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                padding: "4px 8px",
                borderRadius: 6,
                background: "rgba(255,255,255,0.09)",
                fontSize: 12,
                maxWidth: "100%",
              }}
            >
              <span style={{ fontWeight: 600, opacity: 0.92, overflow: "hidden", textOverflow: "ellipsis" }}>
                {props.overlayGenes.length === 1 ? `viridis · ${g}` : `[${["R", "G", "B"][i] ?? "?"}] ${g}`}
              </span>
              <button
                type="button"
                aria-label={`Remove ${g}`}
                style={{
                  fontSize: 14,
                  lineHeight: 1,
                  padding: "0 6px",
                  cursor: "pointer",
                  background: "transparent",
                  border: "none",
                  color: "inherit",
                  opacity: 0.85,
                }}
                onClick={() => removeGene(g)}
              >
                ×
              </button>
            </span>
          ))
        )}
      </div>

      <p style={{ margin: "10px 0 0", fontSize: 11, opacity: 0.55, lineHeight: 1.4 }}>
        Overlays paint on <strong>centroid</strong> dots. If you only see polygons, turn <strong>Centroids</strong> on
        in the toolbar.
      </p>

      <label style={{ display: "flex", gap: 10, alignItems: "center", fontSize: 12, flexWrap: "wrap", marginTop: 10 }}>
        <span style={{ opacity: 0.85, whiteSpace: "nowrap" }}>Overlay α</span>
        <input
          type="range"
          min={0.15}
          max={1}
          step={0.05}
          value={props.overlayOpacity}
          onChange={(e) => props.onOverlayOpacity(Number(e.target.value))}
          disabled={props.overlayGenes.length === 0}
          style={{ flex: "1 1 120px", minWidth: 80 }}
        />
      </label>
    </div>
  );
}
