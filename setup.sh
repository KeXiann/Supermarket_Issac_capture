#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
case "${1:-}" in
  sources) python3 scripts/fetch_sources.py ;;
  runtime) bash scripts/setup_uv_runtime.sh ;;
  assets) env -u PYTHONPATH -u PYTHONHOME .venv/bin/python scripts/prepare_assets.py ;;
  teleop) bash scripts/setup_teleop.sh ;;
  video) bash scripts/start_ego_camera_vr.sh --setup ;;
  *) echo 'Usage: bash setup.sh {sources|runtime|assets|teleop|video}' >&2; exit 2 ;;
esac
