# Argo

[Acknowledgements — the people who have contributed to Argo](ACKNOWLEDGEMENTS.md)

An open-source quadruped robotics project by AEA PoliMI, built around the
SpotMicro platform. The repository combines a MuJoCo robot model and simulation
with reinforcement-learning training tools for moving policies from simulation
toward a real robot.

Argo is a student robotics research and development project within AEA PoliMI
at Politecnico di Milano, continuing the earlier AEA Spot Micro project. It
brings together mechanical design, electronics, software, and robot control,
with the broader goal of developing robotic systems and exploring how robots
move, perceive their surroundings, and interact with people.

The immediate focus is establishing a team for **RoboCup Rescue**, while working
toward institutional project approval and initial funding. A quadruped is the
intended starting platform, preferably a commercial Unitree robot; acquisition
is still subject to approval and funding. Competition provides a concrete
near-term objective within Argo's wider research mission, which includes
human–robot interaction and interests in actuation, energy systems, and
bio-inspired mechanical design.

Work is coordinated across mechanical, electronics, and software activities,
with shared engineering documentation and tasks in
[Notion — Home & Wiki](https://www.notion.so/14df12925861808981f0c21cc49d1000)
and code in this repository. The existing codebase builds on Spot Micro robot
models, MuJoCo simulation and reinforcement-learning tools. Its current locomotion work trains PPO policies
to walk, turn, recover from disturbances, and handle progressively harder
terrain, friction, sensor noise, motor noise, and external pushes.

## What is in this repo

| Area | Purpose |
| --- | --- |
| `ml_training` | Self-contained Python project, with its own `requirements.txt`. |
| `ml_training/spotmicro` | The simulated robot: Gymnasium environment (`env.py`), MuJoCo simulation (`sim.py`), joints/state/actions (`agent.py`), terrain generation (`terrain.py`). |
| `ml_training/commands` | Velocity command sources: random, fixed, keyboard and joystick (`Device`, `Input`). |
| `ml_training/robot_models` | MuJoCo XML robot model and meshes. |
| `ml_training/training` | `train.py` / `evaluate.py`, run configs (`configs/`), rewards (`rewards/`), curricula, notebooks, and run outputs (`data/`). |
| `ml_training/tests` | Unit tests (run configs, controllers, curricula) and an end-to-end train/resume/evaluate test. |
| `ml_training/tools` | MuJoCo PD gain tuning (`tune_pd.py`), a load test for standing policies (`load_test.py`), and a model viewer (`view_model.py`). |
| `docs` / `mkdocs.yml` | MkDocs documentation sources. |

## Core ideas

- **MuJoCo simulation**: `SpotmicroEnv` creates its own `MujocoSim`; the agent,
  reward functions and curricula reach the simulator through `env.sim`.
- **RL-first locomotion stack**: Stable-Baselines3 PPO is used for walking and
  robust policy training.
- **One config per run**: a YAML file picks the reward, the command source and the
  curricula, each with its own parameters; the fully resolved config is saved with
  every run.
- **Curriculum learning**: terrain, pushes, friction, motor noise and sensor noise get
  harder with training time or with measured competence.
- **Sim-to-real direction**: actuator realism, command smoothing, and robust
  evaluation are active areas of development.

## Current status

This is an active research and development repo. The tested Python environment
is Python 3.12 with the dependencies listed in `ml_training/requirements.txt`.

Known active work includes:

- improving robust-walking reward alignment;
- validating the MuJoCo path more deeply;
- preparing smoother sim-to-real behavior with action limits, delays, and
  actuator modeling.

## Quick start

```bash
git clone https://github.com/andretosi/Argo.git
cd Argo/ml_training

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
./run_container.sh python training/train.py training/configs/robust_walking.yaml   # or one command
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

Every run is described by one YAML config in `ml_training/training/configs/`:

| Config | Task |
| --- | --- |
| `robust_walking.yaml` | Omnidirectional walking with curricula (terrain, pushes, friction, noise) |
| `standing.yaml` | Stand still at the homing pose |
| `trotting.yaml` | Trot forward, starting from a standing policy (`run.init_from`) |
| `walking.yaml` | Walking with the original hand-designed reward |
| `jumping.yaml` | Periodic crouch-and-jump |

From `ml_training/`:

```bash
python training/train.py training/configs/robust_walking.yaml
python training/train.py --resume training/data/robust_walking/checkpoints/step_000500000.zip
python training/evaluate.py training/data/robust_walking/final.zip
python training/evaluate.py <model.zip> --command joystick   # drive the policy yourself
```

A config has these sections (only `command` and `reward` are required):

```yaml
run:     {name: my_run, total_steps: 10_000_000, n_envs: 16, init_from: null, checkpoints: {every_steps: 500_000}}
env:     {max_episode_len: 3000}            # SpotmicroEnv settings
agent:   {}                                 # robot settings (homing pose, limits, ...)
command:                                    # where the velocity command comes from
  random: {p_base2walk: 0.7}                #   random | fixed | keyboard | joystick
reward:                                     # which reward, then its own parameters
  robust_walking: {tracking_sigma: 0.25}    #   robust_walking | walking | trotting | standing | jumping
curriculum:                                 # optional
  progression: competence                   #   or: time
  terrain: {z_max: [0.01, 0.10]}            #   terrain | push | friction | motor_noise | sensor_noise
ppo:     {learning_rate: 3.0e-4, learning_rate_final: 1.0e-5}
```

`run.n_envs` runs that many simulations in parallel, one process each (MuJoCo runs on the
CPU, so this is what speeds training up; `--gui` trains with one). On Linux/WSL with an
x86-64 CPU, the processes exchange steps through shared memory
(`training/vec_env.py`), which avoids the per-step messaging cost of pipes; elsewhere
SB3's `SubprocVecEnv` is used. Use about two thirds of your CPU threads at most (16 on a
24-thread CPU): the workers poll for work, and more of them than cores slows everything. PPO collects `ppo.n_steps`
steps from each, so keep `n_envs * n_steps` at the rollout size you want (2048 in
`robust_walking.yaml`: 16 x 128). `run.torch_threads` (default 1) is the number of PyTorch
threads; one is fastest for the small policy network.

Unknown names or parameters are errors; anything left out keeps its default. A run writes
`training/data/<name>/` with the resolved `config.yaml`, checkpoints (each with a
`.state.yaml` holding the curriculum state, used to resume or replay), `final.zip`, and
TensorBoard logs. To add a reward, write a `Reward` subclass in `training/rewards/` with a
`name` (its key under `reward:`), a nested `Params(RewardConfig)` dataclass (its
parameters) and a `compute(env, action)` method returning the step's reward components, and
add it to `REWARDS` in `training/rewards/__init__.py`. The environment only decides when an
episode ends (`env.termination`: `"fall"` or `"tipping"`, and `env.truncated` at the time
limit); every reward scores that with its `fall_penalty`, `tipping_penalty` and
`survival_bonus` parameters, inherited from `RewardConfig`. In the loaded config,
`config.reward` is that `Params` object: `config.reward.name`, `config.reward.make()`.
Command devices (`DeviceConfig`) and curricula (`CurriculumConfig`) work the same way.

The notebooks in `ml_training/training/notebooks/` (one per config) call the same
`train` / `evaluate` functions. `evaluate.py` replays models from run folders only: it
takes the config and curriculum state saved with the model.

## Simulation

The robot is simulated in MuJoCo, from `ml_training/robot_models/spotmicroai.mujoco.xml`.
The environment creates the simulation itself:

```python
env = SpotmicroEnv(command, reward, params=EnvParams(), use_gui=False)
env.sim  # the MujocoSim instance (model, data, viewer)
```

Commands, observations and the agent state use the **robot frame**: x towards the head,
y to the left, z up. A positive `vx` command means walking head first, positive `vy` moves
left, and a positive yaw rate turns left. The physics timestep is `1 / env.sim_frequency`.

## Related project

The robot also depends on embedded work for STM32G431CBU6 motor controllers over
FDCAN. That custom bootloader is developed here:

https://github.com/NicoDelle/can_bootloader

## Credits

Thanks to the authors of the original
[Spot Micro](https://spotmicroai.readthedocs.io/en/latest/) project for the CAD
models and inspiration. The robot meshes in `ml_training/robot_models/meshes` are
[Spotmicro - robot dog](http://www.thingiverse.com/thing:3445283) by KDY0523,
licensed under [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/).

Built with love by
[AEA PoliMI](https://www.aeapolimi.it/), the Automation Engineering Association
of Politecnico di Milano.
