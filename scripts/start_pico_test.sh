#!/usr/bin/env bash
# Current pinned task-1 collection. Historical scene/model launchers are removed.
set -euo pipefail
BRIDGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export SONIC_ROOT="${SONIC_ROOT:-$(dirname "$BRIDGE_ROOT")/GR00T-WholeBodyControl}"
export PICO_RUNTIME=uv
export SONIC_SIM_STARTUP_REFERENCE=1
CAPTURE_PYTHON="$BRIDGE_ROOT/.venv/bin/python"
BRIDGE_CWD="$BRIDGE_ROOT/.runtime/IsaacLab-2.3.2-sparse"
if [[ ! -x "$CAPTURE_PYTHON" || ! -d "$BRIDGE_CWD" ]]; then
  echo "Missing $PICO_RUNTIME runtime: $CAPTURE_PYTHON / $BRIDGE_CWD" >&2
  exit 1
fi
case "${PICO_MANAGER_PREVIEW:-0}" in
  0|1) ;;
  *) echo 'PICO_MANAGER_PREVIEW must be 0 or 1.' >&2; exit 2 ;;
esac
case "${1:-}" in
  batch-supermarket)
    shift
    exec "$CAPTURE_PYTHON" "$BRIDGE_ROOT/bridge/batch_capture.py" --scene supermarket "$@"
    ;;
  bridge-supermarket)
    for CAMERA_ENV in PICO_EGO_PITCH_DOWN_DEG PICO_EGO_FOCAL_LENGTH_MM PICO_EGO_MOUNT_X_M PICO_EGO_MOUNT_Y_M PICO_EGO_MOUNT_Z_M; do
      if [[ -v "$CAMERA_ENV" ]]; then
        echo "Formal collection uses pinned upstream cameras; unset $CAMERA_ENV first." >&2
        exit 2
      fi
    done
    if pgrep -f '[p]ython[^ ]* .*sonic_isaacsim_bridge.py' >/dev/null; then
      echo 'An existing Bridge is running. End it before starting another.' >&2
      exit 1
    fi
    TASK_DEFAULT="$(cd "$BRIDGE_ROOT" && "$CAPTURE_PYTHON" -c 'from upstream_contract import document; print(document("configs/task1/init/task.json")["instruction"])')"
    TASK_LABEL="${PICO_TASK:-$TASK_DEFAULT}"
    TASK_SLUG="$(printf '%s' "$TASK_LABEL" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')"
    if [[ -z "$TASK_SLUG" ]]; then
      echo 'PICO_TASK must contain at least one ASCII letter or digit.' >&2
      exit 2
    fi
    SCENE="$BRIDGE_ROOT/assets/scenes/pico_supermarket_task1_7e5fd9d.usda"
    [[ -s "$SCENE" ]] || { echo "Missing current scene: $SCENE" >&2; exit 1; }
    PUBLICATION_ROOT="$BRIDGE_ROOT/outputs/collected_data"
    WORK_ROOT="$PUBLICATION_ROOT/.capture_work"
    mkdir -p "$WORK_ROOT"
    RUN_DIR="$(mktemp -d "$WORK_ROOT/${TASK_SLUG}_XXXXXX")"
    MENTOR_OUTPUT="$("$CAPTURE_PYTHON" "$BRIDGE_ROOT/bridge/batch_runtime.py" --outputs "$PUBLICATION_ROOT" --task "$TASK_LABEL")"
    export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
    HEADLESS_ARGS=()
    case "${PICO_HEADLESS:-0}" in
      1) HEADLESS_ARGS=(--headless) ;;
      0) ;;
      *) echo 'PICO_HEADLESS must be 0 or 1.' >&2; exit 2 ;;
    esac
    "$CAPTURE_PYTHON" "$BRIDGE_ROOT/bridge/capture_session.py" init --run "$RUN_DIR" \
      --profile bridge-supermarket --manual-recording --task "$TASK_LABEL" \
      --mentor-only-output "$MENTOR_OUTPUT" --publication-root "$PUBLICATION_ROOT"
    printf '%s\n' "$RUN_DIR" > "$BRIDGE_ROOT/outputs/collected_data/latest_pico_run.txt"
    printf '%s\n' "$SCENE" > "$RUN_DIR/scene_path.txt"
    printf '%s\n' revo2 > "$RUN_DIR/hand_profile.txt"
    git -C "$BRIDGE_ROOT/assets/upstream/7e5fd9d0ae2f" rev-parse HEAD > "$RUN_DIR/scene_commit.txt"
    echo "Run directory: $RUN_DIR"
    echo "Output after successful save: $MENTOR_OUTPUT"
    echo 'Native upstream USD; physics target 200 Hz, control 50 Hz, native camera 25 Hz.'
    echo 'After entering POSE, wait for [PERF] READY before recording.'
    "$CAPTURE_PYTHON" "$BRIDGE_ROOT/bridge/capture_manifest.py" --run "$RUN_DIR" --component bridge --scene-file "$SCENE"
    cd "$BRIDGE_CWD"
    "$CAPTURE_PYTHON" "$BRIDGE_ROOT/bridge/capture_session.py" run-component --run "$RUN_DIR" --component bridge -- \
      "$CAPTURE_PYTHON" "$BRIDGE_ROOT/bridge/sonic_isaacsim_bridge.py" \
      --mode drive --target-policy sonic-pd --scene "$SCENE" --dexterous-hands \
      --wait-for-policy-log "$RUN_DIR/sonic.log" --startup-timeout "${PICO_STARTUP_TIMEOUT:-3600}" \
      --pd-target-slew none --ego-camera --ego-camera-height 360 \
      --ego-camera-record "$RUN_DIR/ego_view.mp4" --ego-camera-zmq-port 5555 \
      "${HEADLESS_ARGS[@]}" --device cpu --pd-backend external \
      --kp-scale 1 --kd-scale 1 --torque-scale 1 --render-every 8 --watchdog .3 --duration 0 \
      --stop-request-file "$RUN_DIR/simulation_stop.json" --record "$RUN_DIR/bridge.npz" --checkpoint-seconds 5
    ;;
  sonic|manager)
    if [[ ! -f "$BRIDGE_ROOT/outputs/collected_data/latest_pico_run.txt" ]]; then
      echo 'Start the bridge terminal first.' >&2
      exit 1
    fi
    RUN_DIR="${PICO_RUN_DIR:-$(cat "$BRIDGE_ROOT/outputs/collected_data/latest_pico_run.txt")}"
    COMPONENT=manager
    if [[ "$1" == sonic* ]]; then COMPONENT=sonic; fi
    "$CAPTURE_PYTHON" "$BRIDGE_ROOT/bridge/capture_session.py" preflight --run "$RUN_DIR" --component "$COMPONENT"
    if ! pgrep -f '[p]ython[^ ]* .*sonic_isaacsim_bridge.py' >/dev/null; then
      echo 'The Bridge is not running. Start the bridge terminal first.' >&2
      exit 1
    fi
    echo "Run directory: $RUN_DIR"
    if [[ "$1" == sonic ]]; then
      if pgrep -f '[/]g1_deploy_onnx_ref( |$)' >/dev/null; then
        echo 'An existing SONIC process is running. End it before starting another.' >&2
        exit 1
      fi
      cd "$SONIC_ROOT/gear_sonic_deploy"
      export TensorRT_ROOT="${TensorRT_ROOT:-$HOME/TensorRT}"
      set +u
      source scripts/setup_env.sh
      set -u
      SONIC_PROFILE=release
      MODEL_DIR="policy/$SONIC_PROFILE"
      for MODEL_FILE in model_decoder.onnx model_encoder.onnx observation_config.yaml; do
        if [[ ! -s "$MODEL_DIR/$MODEL_FILE" ]]; then
          echo "Incomplete matched SONIC profile: $MODEL_DIR/$MODEL_FILE" >&2
          exit 1
        fi
      done
      (set -C; printf '%s\n' "$SONIC_PROFILE" > "$RUN_DIR/sonic_profile.txt")
      echo "SONIC matched encoder/decoder/config: $SONIC_PROFILE"
      echo 'PICO Manager enables the pose stream when entering POSE. Do not press Enter in SONIC.'
      "$CAPTURE_PYTHON" "$BRIDGE_ROOT/bridge/capture_manifest.py" \
        --run "$RUN_DIR" --component sonic --profile "$SONIC_PROFILE"
      "$CAPTURE_PYTHON" "$BRIDGE_ROOT/bridge/capture_session.py" run-component --run "$RUN_DIR" --component sonic -- \
      "$CAPTURE_PYTHON" "$BRIDGE_ROOT/bridge/run_sonic_with_debug.py" --run "$RUN_DIR" --port 5557 -- \
      ./target/release/g1_deploy_onnx_ref lo \
        "$MODEL_DIR/model_decoder.onnx" reference/example/ \
        --obs-config "$MODEL_DIR/observation_config.yaml" \
        --encoder-file "$MODEL_DIR/model_encoder.onnx" \
        --planner-file planner/target_vel/V2/planner_sonic.onnx \
        --disable-crc-check --input-type zmq_manager --output-type zmq --zmq-out-port 5557 \
        --target-motion-logfile "$RUN_DIR/sonic_target.csv" \
        --planner-motion-logfile "$RUN_DIR/sonic_planner.csv" \
        --enable-csv-logs --logs-dir "$RUN_DIR/sonic_state"
    else
      if pgrep -f '[p]ython[^ ]* .*pico_manager_thread_server.py' >/dev/null; then
        echo 'An existing PICO manager is running. End it before starting another.' >&2
        exit 1
      fi
      cd "$SONIC_ROOT"
      set +u
      source .venv_teleop/bin/activate
      set -u
      export PYTHONUNBUFFERED=1
      MANAGER_ARGS=()
      if [[ "${PICO_MANAGER_PREVIEW:-0}" == 1 ]]; then
        MANAGER_ARGS+=(--vis_vr3pt --vis_smpl)
        echo 'Manager pose previews enabled for diagnostics; rendering shares the pose publishing loop.'
      else
        echo 'Manager pose previews disabled; FK calibration, pose tracking and headset video remain active.'
      fi
      if "$CAPTURE_PYTHON" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1])).get("manual_recording"))' "$RUN_DIR/session.json"; then
        MANAGER_ARGS+=(--recording-control-file "$RUN_DIR/recording_control.json")
        echo 'Recording: calibrate ABXY, enter POSE with A+X, release buttons; left grip + A STARTS; repeat STOPS/SAVES; left grip + B DISCARDS.'
        echo 'Wait for [CAPTURE] Mentor-only output retained. Batch coordinator, if enabled, prepares the next episode automatically.'
      fi
      if [[ -f "$RUN_DIR/hand_profile.txt" ]] && [[ "$(cat "$RUN_DIR/hand_profile.txt")" == revo2 ]]; then
        MANAGER_ARGS+=(--hand-profile revo2)
        echo 'Revo2 hands: main trigger = proportional full-hand grasp; release = open. Side grip only controls recording chords.'
      fi
      (set -C; printf '%s\n' "$1" > "$RUN_DIR/manager_profile.txt")
      python "$BRIDGE_ROOT/bridge/capture_manifest.py" --run "$RUN_DIR" --component manager
      "$CAPTURE_PYTHON" "$BRIDGE_ROOT/bridge/capture_session.py" run-component --run "$RUN_DIR" --component manager -- \
      python gear_sonic/scripts/pico_manager_thread_server.py \
        --manager --record_dir "$RUN_DIR/pico_input" \
        --sim-stop-file "$RUN_DIR/simulation_stop.json" \
        "${MANAGER_ARGS[@]}"
    fi
    ;;
  stop|check|finalize|assess)
    RUN_DIR="${2:-${PICO_RUN_DIR:-$(cat "$BRIDGE_ROOT/outputs/collected_data/latest_pico_run.txt")}}"
    if [[ "$1" == assess ]]; then
      if [[ -f "$RUN_DIR/meta/submission.json" || ( -d "$RUN_DIR/data" && -d "$RUN_DIR/meta" && -d "$RUN_DIR/videos" && ! -f "$RUN_DIR/session.json" ) ]]; then
        echo 'Formal Mentor-only output has no raw readiness artifacts; use prepare_mentor_sample.py --check for format validation.' >&2
        exit 2
      fi
      "$CAPTURE_PYTHON" "$BRIDGE_ROOT/bridge/assess_training_readiness.py" "$RUN_DIR"
    else
      "$CAPTURE_PYTHON" "$BRIDGE_ROOT/bridge/capture_session.py" "$1" "$RUN_DIR"
    fi
    ;;
  *)
    echo "Usage: bash $BRIDGE_ROOT/scripts/start_pico_test.sh {batch-supermarket|bridge-supermarket|sonic|manager|stop|check|finalize|assess}" >&2
    exit 2
    ;;
esac
