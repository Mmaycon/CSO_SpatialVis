export type SampleBounds = {
  min_x: number;
  min_y: number;
  max_x: number;
  max_y: number;
};

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
