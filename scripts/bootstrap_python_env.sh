#!/usr/bin/env bash
# Create CSO_SpatialVis/.venv and install backend + tooling deps (parquet, API, xenium helper scripts).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r backend/requirements.txt
echo
echo "Python env ready: $ROOT/.venv"
echo "Activate: source $ROOT/.venv/bin/activate"
echo "Example:   python scripts/add_polygon_bboxes_to_parquet.py data/your_sample"
