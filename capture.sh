#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-batch}" in
  batch) shift "$(( $# > 0 ? 1 : 0 ))"; exec bash "$ROOT/scripts/start_pico_test.sh" batch-supermarket "$@" ;;
  video) shift; exec bash "$ROOT/scripts/start_ego_camera_vr.sh" "$@" ;;
  stop|check|finalize|assess|bridge-supermarket|sonic|manager)
    exec bash "$ROOT/scripts/start_pico_test.sh" "$@" ;;
  *) echo 'Usage: bash capture.sh {batch|video|stop|check|finalize|assess} [args]' >&2; exit 2 ;;
esac
