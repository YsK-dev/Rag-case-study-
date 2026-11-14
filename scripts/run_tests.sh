#!/usr/bin/env bash
# Run the unit tests in the light test mode (no models are loaded).
set -euo pipefail
cd "$(dirname "$0")/.."
export RAG_TESTING=1
python -m pytest -v "$@"
