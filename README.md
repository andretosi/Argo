# Spot Micro AEA PoliMI

An open-source quadruped robotics project by AEA PoliMI, built around the
SpotMicro platform. The repository combines robot models, simulator backends,
reinforcement-learning training tools, and early ROS/Gazebo assets for moving
policies from simulation toward a real robot.

The current focus is robust locomotion: training PPO policies that can walk,
turn, recover from disturbances, and gradually handle harder terrain, friction,
motor noise, sensor noise, and external pushes.

## What is in this repo

| Area | Purpose |
| --- | --- |
| `ml_training` | Self-contained Python project, with its own `requirements.txt`. |
| `ml_training/environments` | Gymnasium environments (`SpotmicroEnv`) and terrain generation. |
| `ml_training/robot` | Robot body model: joints, state, action handling, and the command `Controller`. |
| `ml_training/physics_backends` | PyBullet and MuJoCo backends behind a common `PhysicsBackend` interface. |
| `ml_training/command_devices` | Velocity command sources: keyboard, joystick, random and fixed controllers. |
| `ml_training/configuration` | Config system (`Config`, `@configurable`). |
| `ml_training/visualization` | Debug visuals: kinematic ghost and command tracker. |
| `ml_training/robot_models` | URDF and MuJoCo XML robot models. |
| `ml_training/training` | PPO training scripts, reward functions, curriculum callbacks, and notebooks. |
| `ml_training/tests` | Unit and integration tests for configuration, controllers, curricula, and environment wiring. |
| `ml_training/calibration` | MuJoCo PD gain tuning (`tune_pd.py`) and a load test for standing policies (`load_test.py`). |
| `spot_micro` | ROS/Gazebo-facing model and launch assets. |
| `docs` / `mkdocs.yml` | MkDocs documentation sources. |

## Core ideas

- **Backend-agnostic environment**: `SpotmicroEnv` routes simulator calls
  through a `PhysicsBackend`, so PyBullet and MuJoCo can share the same high
  level training loop.
- **RL-first locomotion stack**: Stable-Baselines3 PPO is used for walking and
  robust policy training.
- **Curriculum learning**: terrain, pushes, friction, motor noise, sensor noise,
  and competence tracking are split into focused callbacks.
- **Configurable components**: the local config system keeps tunable parameters
  serializable while excluding runtime-only objects such as simulator backends,
  devices, reward functions, and loggers.
- **Sim-to-real direction**: ROS/Gazebo assets, actuator realism, command
  smoothing, and robust evaluation are active areas of development.

## Current status

This is an active research and development repo. The tested Python environment
is Python 3.12 with the dependencies listed in `ml_training/requirements.txt`.

Known active work includes:

- improving robust-walking reward alignment;
- validating the MuJoCo path more deeply;
- cleaning up legacy environment and agent paths;
- adding better long-run evaluation and checkpoint wiring;
- preparing smoother sim-to-real behavior with action limits, delays, and
  actuator modeling.

## Quick start

```bash
git clone https://github.com/Andrea18500/aea_spot_micro.git
cd aea_spot_micro/ml_training

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The training scripts add `ml_training/` to `sys.path` themselves. For notebooks
and anything else that imports these modules, run from `ml_training/` with:

```bash
export PYTHONPATH="$PWD"
```

### Docker

Instead of a local venv, you can work inside a container:

```bash
cd ml_training
./run_container.sh                                   # interactive shell
./run_container.sh python training/train_robust.py   # or run one command
```

The script builds the image from `ml_training/Dockerfile` (cached after the
first build, and rebuilt when `requirements.txt` changes) and mounts
`ml_training/` at `/workspace`, so edits on the host appear in the container
immediately. GPUs are used when the host has the NVIDIA Container Toolkit, and
the GUI is forwarded over X11/WSLg. Ports 8888 (Jupyter, start it with
`--ip 0.0.0.0`) and 6006 (TensorBoard, `--bind_all`) are published.

## Verify the project

Run the test suite from `ml_training/`:

```bash
source .venv/bin/activate
python -m pytest tests
```

Build the documentation from the repository root with:

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs build --strict
```

## Training

The main robust-training entry point is:

```bash
python training/train_robust.py   # from ml_training/
```

Important knobs are intentionally near the top of that file:

- physics backend: `pybullet` or `mujoco`;
- GUI/headless mode;
- total PPO steps;
- checkpoint frequency;
- curriculum schedules for terrain, pushes, friction, motor noise, sensor noise,
  and competence-based progression.

Older and simpler training scripts are also present in `ml_training/training/`, including
`train_policy.py`, `train_further.py`, and policy test scripts. Some notebooks
under `ml_training/training/notebooks/` are experimental snapshots and may reflect older
training flows.

## Simulation backends

The active backend factory is:

```python
from physics_backends.factory import create_backend

backend = create_backend("pybullet", use_gui=False)
```

Supported backend names:

- `pybullet`
- `mujoco`

Both backends are intended to expose the same `PhysicsBackend` interface so the
agent, environment, reward functions, and curricula can stay simulator-agnostic.

## Related project

The robot also depends on embedded work for STM32G431CBU6 motor controllers over
FDCAN. That custom bootloader is developed here:

https://github.com/NicoDelle/can_bootloader

## Credits

Thanks to the authors of the original
[Spot Micro](https://spotmicroai.readthedocs.io/en/latest/) project for the CAD
models and inspiration.

Built with love by
[AEA PoliMI](https://www.aeapolimi.it/), the Automation Engineering Association
of Politecnico di Milano.
