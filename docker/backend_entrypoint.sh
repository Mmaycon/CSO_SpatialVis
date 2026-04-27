#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${CSO_DATA_ROOT:-/app/data}"
python /app/scripts/generate_example_data.py --out "${CSO_DATA_ROOT:-/app/data}"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
