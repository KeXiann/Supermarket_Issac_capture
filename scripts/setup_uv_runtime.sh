#!/usr/bin/env bash
set -euo pipefail
BRIDGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BRIDGE_ROOT"
command -v uv >/dev/null
command -v patchelf >/dev/null
python3 scripts/fetch_sources.py isaaclab
export SONIC_ROOT="${SONIC_ROOT:-$(dirname "$BRIDGE_ROOT")/GR00T-WholeBodyControl}"
SDK_DDS="$SONIC_ROOT/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty"
[[ -d "$SDK_DDS/include/dds" ]] || { echo 'Run bash setup.sh sources first.' >&2; exit 1; }
export CYCLONEDDS_HOME="$BRIDGE_ROOT/.runtime/cyclonedds-sdk"
mkdir -p "$CYCLONEDDS_HOME/bin"
ln -sfn "$SDK_DDS/include" "$CYCLONEDDS_HOME/include"
ln -sfn "$SDK_DDS/lib/x86_64" "$CYCLONEDDS_HOME/lib"
uv sync --frozen --managed-python
env -u PYTHONPATH -u PYTHONHOME .venv/bin/python scripts/repair_dds_runtime.py
env -u PYTHONPATH -u PYTHONHOME .venv/bin/python bridge/verify_uv_runtime.py
