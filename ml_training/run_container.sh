#!/usr/bin/env bash
# Build the ml_training image (cached, so fast after the first time) and open
# an interactive shell in it with this folder mounted at /workspace.
#
#   ./run_container.sh                   # interactive bash
#   ./run_container.sh python training/train.py training/configs/robust_walking.yaml   # run a command instead
#
# Ports: 8888 (Jupyter: jupyter notebook --ip 0.0.0.0), 6006 (TensorBoard: tensorboard --logdir . --bind_all)
set -euo pipefail

IMAGE="spotmicro-ml-training"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker build \
    --build-arg USER_ID="$(id -u)" \
    --build-arg GROUP_ID="$(id -g)" \
    -t "$IMAGE" \
    "$SCRIPT_DIR"

run_args=(
    --rm -it
    --name "$IMAGE"
    -v "$SCRIPT_DIR":/workspace
    -p 8888:8888
    -p 6006:6006
    --shm-size=2g
    --hostname "$(hostname)"   # runs are tagged with the machine they trained on
)

# Expose NVIDIA GPUs when the host has them (needs the NVIDIA Container Toolkit).
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    run_args+=(--gpus all)
fi

# MLflow / Databricks tracking (see training/tracking.py): forward the settings that are
# set on the host, and the Databricks CLI profiles (~/.databrickscfg) if there are any.
for var in MLFLOW_TRACKING_URI MLFLOW_EXPERIMENT_NAME \
           DATABRICKS_HOST DATABRICKS_TOKEN DATABRICKS_CONFIG_PROFILE; do
    [[ -n "${!var:-}" ]] && run_args+=(-e "$var")
done
if [[ -f "$HOME/.databrickscfg" ]]; then
    run_args+=(-v "$HOME/.databrickscfg":/home/dev/.databrickscfg:ro)
fi

# Forward the display so the MuJoCo viewer opens on the host (X11, incl. WSLg).
if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
    run_args+=(-e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix)
fi
if [[ -d /mnt/wslg ]]; then
    run_args+=(-v /mnt/wslg:/mnt/wslg)
    [[ -n "${WAYLAND_DISPLAY:-}" ]] && run_args+=(-e WAYLAND_DISPLAY="$WAYLAND_DISPLAY")
    [[ -n "${XDG_RUNTIME_DIR:-}" ]] && run_args+=(-e XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR")
    [[ -n "${PULSE_SERVER:-}" ]] && run_args+=(-e PULSE_SERVER="$PULSE_SERVER")
fi

exec docker run "${run_args[@]}" "$IMAGE" "$@"
