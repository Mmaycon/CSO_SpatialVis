export type SampleBounds = {
  min_x: number;
  min_y: number;
  max_x: number;
  max_y: number;
};

/** Pins morphology pixels ↔ µm (``images_manifest.morphology_frame``). */
export type MorphologyFrame = {
  version?: number;
  coordinate_unit?: string;
  convention_note?: string;
  pixel_um: number;
  crop_origin_ix: number;
  crop_origin_iy: number;
  crop_w: number;
  crop_h: number;
  crop_microns: SampleBounds;
  morphology_flip_y_export?: boolean;
  segmentation_reference_channel_id?: string;
  segmentation_reference_channel_label?: string;
  segmentation_reference_channel_index?: number;
  export_micron_bounds?: SampleBounds;
};

export function parseMorphologyFrame(mf: unknown): MorphologyFrame | null {
  if (!mf || typeof mf !== "object") return null;
  const o = mf as Record<string, unknown>;
  const cm = o.crop_microns;
  if (!cm || typeof cm !== "object") return null;
  const b = cm as Record<string, unknown>;
  const min_x = Number(b.min_x);
  const min_y = Number(b.min_y);
  const max_x = Number(b.max_x);
  const max_y = Number(b.max_y);
  const px = Number(o.pixel_um);
  const cix = Number(o.crop_origin_ix);
  const ciy = Number(o.crop_origin_iy);
  const cw = Number(o.crop_w);
  const ch = Number(o.crop_h);
  if (![min_x, min_y, max_x, max_y, px, cix, ciy, cw, ch].every(Number.isFinite)) return null;
  const emb = o.export_micron_bounds;
  let exportBounds: SampleBounds | undefined;
  if (emb && typeof emb === "object") {
    const e = emb as Record<string, unknown>;
    const a = Number(e.min_x);
    const c = Number(e.min_y);
    const d = Number(e.max_x);
    const f = Number(e.max_y);
    if ([a, c, d, f].every(Number.isFinite)) exportBounds = { min_x: a, min_y: c, max_x: d, max_y: f };
  }
  return {
    version: typeof o.version === "number" ? o.version : undefined,
    coordinate_unit: typeof o.coordinate_unit === "string" ? o.coordinate_unit : undefined,
    convention_note: typeof o.convention_note === "string" ? o.convention_note : undefined,
    pixel_um: px,
    crop_origin_ix: cix,
    crop_origin_iy: ciy,
    crop_w: cw,
    crop_h: ch,
    crop_microns: { min_x, min_y, max_x, max_y },
    morphology_flip_y_export: typeof o.morphology_flip_y_export === "boolean" ? o.morphology_flip_y_export : undefined,
    segmentation_reference_channel_id:
      typeof o.segmentation_reference_channel_id === "string" ? o.segmentation_reference_channel_id : undefined,
    segmentation_reference_channel_label:
      typeof o.segmentation_reference_channel_label === "string" ? o.segmentation_reference_channel_label : undefined,
    segmentation_reference_channel_index:
      typeof o.segmentation_reference_channel_index === "number" ? o.segmentation_reference_channel_index : undefined,
    export_micron_bounds: exportBounds,
  };
}

export type SampleRef = {
  sample_id: string;
  title: string;
  description: string;
  coordinate_unit: string;
  bounds: SampleBounds;
  image: Record<string, unknown> | null;
  layers: string[];
  genes_available: number;
};

export type Manifest = {
  version: string;
  samples: SampleRef[];
};

export type CellRecord = {
  cell_id: string;
  x: number;
  y: number;
  metadata: Record<string, unknown>;
};
