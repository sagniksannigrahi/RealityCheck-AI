#!/usr/bin/env bash
# Local (non-Docker) setup: Python 3.12 venv via uv, ML extras, frontend deps, DB migration, model weights.
# Usage: scripts/setup.sh [--no-ml]
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }
command -v npm >/dev/null || { echo "Install Node.js 20+ first"; exit 1; }
EXTRAS="--extra dev"
if [[ "${1:-}" != "--no-ml" ]]; then EXTRAS="$EXTRAS --extra ml"; fi
uv sync $EXTRAS
(cd frontend && npm ci && npm run build)
mkdir -p data
.venv/bin/alembic upgrade head
if [[ "${1:-}" != "--no-ml" ]]; then .venv/bin/python scripts/download_models.py; fi
echo "Done. Start with: .venv/bin/uvicorn app.main:app --app-dir backend --port 8000   (UI at http://localhost:8000)"
