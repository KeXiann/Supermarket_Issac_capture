#!/usr/bin/env bash
# Feed the existing PICO XRoboToolkit application's native Listen receiver.
set -euo pipefail
BRIDGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO_PY="$BRIDGE_ROOT/.venv_ego_video/bin/python"
ACTION="${1:-native}"
case "$ACTION" in
  --setup)
    if [[ ! -x "$VIDEO_PY" ]]; then
      "$HOME/.local/bin/uv" venv --python 3.12 "$BRIDGE_ROOT/.venv_ego_video"
    fi
    exec "$HOME/.local/bin/uv" pip install --python "$VIDEO_PY" -r "$BRIDGE_ROOT/configs/video-requirements.txt"
    ;;
  --check|--restart|native|xr|--xr) ;;
  *) echo "Usage: $0 [--check|--restart|--setup]" >&2; exit 2 ;;
esac
if [[ ! -x "$VIDEO_PY" ]]; then
  echo "Run: bash $BRIDGE_ROOT/start_ego_camera_vr.sh --setup" >&2
  exit 2
fi
"$VIDEO_PY" -c 'import av, zmq, msgpack, PIL, numpy; av.CodecContext.create("libx264", "w"); print("Native PICO video dependencies: OK")'
if [[ "$ACTION" == --check ]]; then exit 0; fi
mkdir -p "$BRIDGE_ROOT/outputs"
STATUS_FILE="$BRIDGE_ROOT/outputs/pico_video_status.json"
for VIDEO_PID in $(pgrep -f '[p]ico_video_server.py --status' || true); do
  VIDEO_CMD="$(tr '\0' ' ' < "/proc/$VIDEO_PID/cmdline" 2>/dev/null || true)"
  if [[ "$VIDEO_CMD" == *"$BRIDGE_ROOT/bridge/pico_video_server.py"* && "$VIDEO_CMD" == *"$STATUS_FILE"* ]]; then
    RELOAD_NEEDED="$("$VIDEO_PY" - "$STATUS_FILE" "$BRIDGE_ROOT/bridge/pico_video_server.py" "$VIDEO_PID" <<'PY'
import hashlib,json,sys
from pathlib import Path
try:
    state=json.loads(Path(sys.argv[1]).read_text())
except (OSError, ValueError):
    state={}
expected=hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()[:16]
current=state.get("server_code_hash")==expected and state.get("server_pid")==int(sys.argv[3])
print("no" if current else "yes")
PY
)"
    if [[ "$ACTION" == --restart || "$RELOAD_NEEDED" == yes ]]; then
      echo "Reloading only the PICO video service (PID $VIDEO_PID); robot control is unaffected."
      kill -TERM "$VIDEO_PID"
      for _ in {1..80}; do
        if ! kill -0 "$VIDEO_PID" 2>/dev/null; then break; fi
        sleep .1
      done
      if kill -0 "$VIDEO_PID" 2>/dev/null; then
        echo "Video service did not exit; refusing to start a conflicting copy." >&2
        exit 1
      fi
      continue
    fi
    echo "Native PICO video server is already running (PID $VIDEO_PID); reusing it."
    VIDEO_STATE="$("$VIDEO_PY" -c 'import json,os,sys,time; p=sys.argv[1]; d=json.load(open(p)) if os.path.exists(p) else {}; age=time.time()-d.get("updated_unix_s",0); print("last session: connected={} source_live={} status_age={:.0f}s".format(d.get("connected",False),d.get("source_live",False),age))' "$STATUS_FILE")"
    echo "$VIDEO_STATE"
    echo 'Do not start another copy. In PICO, select Remote Vision -> ZEDMINI -> Listen again.'
    exit 0
  fi
done
VIDEO_LOG="$BRIDGE_ROOT/outputs/pico_video_$(date +%Y%m%d_%H%M%S).log"
echo "Log: $VIDEO_LOG"
echo 'In PICO XRoboToolkit: Remote Vision -> ZEDMINI -> Listen -> this PC IP.'
echo 'Keep Data & Control / Send enabled. The browser page is unnecessary.'
exec > >(tee "$VIDEO_LOG") 2>&1
exec "$VIDEO_PY" -u "$BRIDGE_ROOT/bridge/pico_video_server.py" \
  --status "$BRIDGE_ROOT/outputs/pico_video_status.json"
