#!/usr/bin/env bash
# A separate Python 3.10 uv environment is required by the XRoboToolkit binding.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export SONIC_ROOT="${SONIC_ROOT:-$(dirname "$ROOT")/GR00T-WholeBodyControl}"
[[ -d "$SONIC_ROOT/gear_sonic" ]] || { echo 'Run bash setup.sh sources first.' >&2; exit 1; }
if [[ ! -x "$SONIC_ROOT/.venv_teleop/bin/python" ]]; then
  uv venv --python 3.10 --managed-python "$SONIC_ROOT/.venv_teleop"
fi
TELEOP_PY="$SONIC_ROOT/.venv_teleop/bin/python"
uv pip install --python "$TELEOP_PY" -r "$ROOT/configs/teleop-requirements.txt"
uv pip install --python "$TELEOP_PY" --no-deps -e "$SONIC_ROOT/gear_sonic"
uv pip install --python "$TELEOP_PY" --no-build-isolation "$SONIC_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
"$TELEOP_PY" -c 'import xrobotoolkit_sdk, pinocchio, torch; print("Teleop Python imports OK; headset/service connection still required")'
