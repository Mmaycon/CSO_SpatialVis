import { useEffect, useMemo, useRef } from "react";

import { colorForType } from "@/colors";

export type UmapPoint = {
  cell_id: string;
  u: number;
  v: number;
  cell_type?: string;
};

export function UmapPanel(props: {
  points: UmapPoint[];
  selectedIds: ReadonlySet<string>;
  onPickCell: (cellId: string, additive: boolean) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const view = useRef({ sx: 1, tx: 0, ty: 0 });

  const bbox = useMemo(() => {
    if (!props.points.length) return { minU: -1, maxU: 1, minV: -1, maxV: 1 };
    let minU = Infinity,
      maxU = -Infinity,
      minV = Infinity,
      maxV = -Infinity;
    for (const p of props.points) {
      minU = Math.min(minU, p.u);
      maxU = Math.max(maxU, p.u);
      minV = Math.min(minV, p.v);
      maxV = Math.max(maxV, p.v);
    }
    const pad = 0.08 * Math.max(maxU - minU, maxV - minV);
    return { minU: minU - pad, maxU: maxU + pad, minV: minV - pad, maxV: maxV + pad };
  }, [props.points]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth;
    const h = c.clientHeight;
    c.width = Math.floor(w * dpr);
    c.height = Math.floor(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = "#12161d";
    ctx.fillRect(0, 0, w, h);

    const { minU, maxU, minV, maxV } = bbox;
    const pad = 14;
    const plotW = w - 2 * pad;
    const plotH = h - 2 * pad;
    const sx = plotW / Math.max(1e-9, maxU - minU);
    const sy = plotH / Math.max(1e-9, maxV - minV);
    const s = Math.min(sx, sy);
    const contentW = (maxU - minU) * s;
    const contentH = (maxV - minV) * s;
    const ox = pad + (plotW - contentW) / 2;
    const oy = pad + (plotH - contentH) / 2;
    const tx = ox - minU * s;
    const ty = oy - minV * s;
    view.current = { sx: s, tx, ty };

    const project = (u: number, v: number) => [u * s + tx, v * s + ty] as const;

    for (const p of props.points) {
      const [x, y] = project(p.u, p.v);
      const sel = props.selectedIds.has(p.cell_id);
      const [r, g, b] = colorForType(p.cell_type).slice(0, 3);
      ctx.beginPath();
      ctx.fillStyle = sel ? "rgba(255,255,120,0.95)" : `rgba(${r},${g},${b},0.55)`;
      ctx.strokeStyle = sel ? "rgba(40,40,10,0.9)" : "rgba(10,12,16,0.35)";
      ctx.lineWidth = sel ? 1.2 : 0.6;
      ctx.arc(x, y, sel ? 3.2 : 2.1, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    ctx.fillStyle = "rgba(230,235,245,0.75)";
    ctx.font = "12px ui-sans-serif, system-ui";
    ctx.fillText("Precomputed UMAP (linked selection)", 12, 18);
  }, [bbox, props.points, props.selectedIds]);

  const onClick = (e: React.MouseEvent) => {
    const c = canvasRef.current;
    if (!c) return;
    const rect = c.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const { sx, tx, ty } = view.current;
    const wx = (mx - tx) / sx;
    const wy = (my - ty) / sx;
    let best: { id: string; d2: number } | null = null;
    for (const p of props.points) {
      const dx = p.u - wx;
      const dy = p.v - wy;
      const d2 = dx * dx + dy * dy;
      if (!best || d2 < best.d2) best = { id: p.cell_id, d2 };
    }
    if (best && best.d2 < 0.06) {
      props.onPickCell(best.id, e.shiftKey);
    }
  };

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: "100%", display: "block", cursor: "crosshair" }}
      onClick={onClick}
    />
  );
}
