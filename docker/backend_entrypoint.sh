#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${CSO_DATA_ROOT:-/app/data}"
mkdir -p "$DATA_ROOT"
# Regenerating on every start overwrote cell_metadata.parquet and wiped ROI annotations.
# Generate only when the default demo sample is missing.
if [[ ! -f "$DATA_ROOT/sample_demo/cells.parquet" ]]; then
  echo "CSO SpatialVis: generating initial demo data under $DATA_ROOT"
  python /app/scripts/generate_example_data.py --out "$DATA_ROOT"
fi
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
