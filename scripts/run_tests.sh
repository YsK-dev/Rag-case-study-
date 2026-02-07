#!/usr/bin/env bash
set -euo pipefail

# Lightweight test mode to avoid heavy model initialization
export RAG_TESTING=1

pytest '/Users/ysk/Desktop/Rag(case-study)/tests/test_app.py' -v
