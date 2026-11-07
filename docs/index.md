# Spot Micro AEA PoliMI

Reinforcement-learning locomotion for the SpotMicro quadruped, simulated in MuJoCo and
trained with Stable-Baselines3 PPO. Setup, training commands and the run-config format
are in the repository [README](https://github.com/Andrea18500/aea_spot_micro#readme).

## Code layout (`ml_training/`)

| Package | Contents |
| --- | --- |
| `spotmicro` | `SpotmicroEnv` (Gymnasium environment), `MujocoSim`, `Agent` (joints, state, actions), terrain heightmaps |
| `commands` | Velocity command sources (`Device`) and the command they produce (`Input`) |
| `training` | `train.py`, `evaluate.py`, run configs, rewards, curricula, notebooks |
| `robot_models` | MuJoCo XML model and meshes |
| `tools` | PD gain tuning, standing load test, model viewer |

How the pieces connect:

```
PPO (SB3) <-> SpotmicroEnv <-> Agent <-> MujocoSim
                  |  ^             ^
             reward  curricula     command device
```

The [API reference](api.md) is generated from the docstrings.
