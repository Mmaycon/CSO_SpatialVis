#!/usr/bin/env bash
# Export GLP1_BrCa CosMx SpatialData zarr into CSO_SpatialVis sample assets.
#
# Prerequisite: conda env with spatialdata (see scripts/conda-cosmx-pipeline.env.yml)
# or: pip install -r scripts/requirements-cosmx-pipeline.txt
#
# Full export (~631k cells) is large and may take hours. For a quick smoke test:
#   N_CELLS=5000 bash scripts/get_cosmx_export.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SDATA_ZARR="${SDATA_ZARR:-/mnt/scratch2/Maycon/Visualization_tools/CosMx_images/GLP1_BrCa/cosmx.sdata.zarr}"
MORPHOLOGY_ZARR="${MORPHOLOGY_ZARR:-/mnt/scratch2/Maycon/Visualization_tools/CosMx_images/GLP1_BrCa/morphology.ome.zarr}"
TRANSCRIPTS_PARQUET="${TRANSCRIPTS_PARQUET:-/mnt/scratch2/Maycon/Visualization_tools/CosMx_images/GLP1_BrCa/transcripts.parquet}"
OUT_DIR="${OUT_DIR:-$ROOT/data/GLP1_BrCa_cosmx}"
N_CELLS="${N_CELLS:-0}"

mkdir -p "${OUT_DIR}"

EXTRA=()
if [[ "${N_CELLS}" == "0" ]]; then
  EXTRA+=(--export-all-cells)
else
  EXTRA+=(--n-cells "${N_CELLS}")
fi

python3 scripts/cosmx_to_spatialvis.py \
  --sdata-zarr "${SDATA_ZARR}" \
  --morphology-zarr "${MORPHOLOGY_ZARR}" \
  --transcripts-parquet "${TRANSCRIPTS_PARQUET}" \
  --out-dir "${OUT_DIR}" \
  --registration-channel DNA \
  --he-long-edge 8192 \
  --morphology-flip-y \
  "${EXTRA[@]}" \
  "$@"

echo
echo "Sample written to ${OUT_DIR}"
echo "Set CSO_DATA_ROOT=${ROOT}/data and restart the backend to load it."
