export const TYPE_PALETTE: Record<string, [number, number, number, number]> = {
  stromal: [140, 180, 220, 200],
  immune_rich: [230, 120, 140, 200],
  necrosis_like: [190, 190, 190, 200],
  tumor_core: [245, 190, 120, 200],
};

/** Approximate matplotlib viridis (dark purple → teal → yellow). */
const VIRIDIS_STOPS: [number, number, number][] = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [109, 205, 89],
  [183, 222, 39],
  [253, 231, 36],
];

export function viridisRgb(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS_STOPS.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = VIRIDIS_STOPS[i]!;
  const b = VIRIDIS_STOPS[Math.min(i + 1, VIRIDIS_STOPS.length - 1)]!;
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

/** CSS linear-gradient matching `viridisRgb` for legends. */
export const VIRIDIS_GRADIENT_CSS =
  "linear-gradient(90deg, rgb(68,1,84) 0%, rgb(59,82,139) 25%, rgb(33,144,140) 50%, rgb(93,200,99) 75%, rgb(253,231,36) 100%)";

export function colorForType(cellType: unknown): [number, number, number, number] {
  if (typeof cellType !== "string") return [160, 200, 255, 200];
  return TYPE_PALETTE[cellType] ?? [160, 200, 255, 200];
}

function hslToRgb(h01: number, s: number, l: number): [number, number, number] {
  let r: number;
  let g: number;
  let b: number;
  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p: number, q: number, t: number) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h01 + 1 / 3);
    g = hue2rgb(p, q, h01);
    b = hue2rgb(p, q, h01 - 1 / 3);
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/** Stable categorical color for ROI / pathology_region labels. */
/** Categorical coloring for arbitrary metadata columns (cell_type uses palette; others hashed). */
export function colorForMetadataKey(
  fieldKey: string,
  value: unknown,
): [number, number, number, number] {
  if (value === null || value === undefined || value === "") return [88, 92, 102, 200];
  if (fieldKey === "cell_type") return colorForType(value);
  return colorForPathology(String(value));
}

export function colorForPathology(region: unknown): [number, number, number, number] {
  if (typeof region !== "string" || !region.trim()) return [120, 125, 135, 210];
  let h = 2166136261;
  for (let i = 0; i < region.length; i++) {
    h = Math.imul(h ^ region.charCodeAt(i), 16777619);
  }
  const hueDeg = (h >>> 0) % 360;
  const [r, g, b] = hslToRgb(hueDeg / 360, 0.62, 0.52);
  return [r, g, b, 220];
}

export function colorFromExpression(v: number, vmin: number, vmax: number): [number, number, number, number] {
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax) || vmax <= vmin) {
    return [80, 80, 90, 200];
  }
  if (!Number.isFinite(v)) return [80, 80, 90, 200];
  const t = Math.min(1, Math.max(0, (v - vmin) / (vmax - vmin)));
  const [r, g, b] = viridisRgb(t);
  return [r, g, b, 220];
}

/** Multi-gene overlay: 1 → viridis; 2–3 → R,G,B channel intensities (per-gene viewport min/max). */
export function overlayRgbForCell(
  cellId: string,
  overlayGenes: readonly string[],
  maps: Record<string, Record<string, number>>,
  ranges: Record<string, { vmin: number; vmax: number }>,
): [number, number, number, number] {
  const n = overlayGenes.length;
  if (n === 0) return [0, 0, 0, 0];
  const norm = (gene: string) => {
    const v = maps[gene]?.[cellId];
    const { vmin, vmax } = ranges[gene] ?? { vmin: 0, vmax: 1 };
    if (!Number.isFinite(v) || !Number.isFinite(vmin) || !Number.isFinite(vmax) || vmax <= vmin) return 0;
    return Math.min(1, Math.max(0, (v - vmin) / (vmax - vmin)));
  };
  if (n === 1) {
    const t = norm(overlayGenes[0]!);
    const [r, g, b] = viridisRgb(t);
    return [r, g, b, 255];
  }
  if (n === 2) {
    return [Math.round(255 * norm(overlayGenes[0]!)), Math.round(255 * norm(overlayGenes[1]!)), 30, 255];
  }
  return [
    Math.round(255 * norm(overlayGenes[0]!)),
    Math.round(255 * norm(overlayGenes[1]!)),
    Math.round(255 * norm(overlayGenes[2]!)),
    255,
  ];
}
