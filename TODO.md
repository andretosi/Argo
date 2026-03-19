# RULES FOR AGENTS
To the agents: only actually do the todos after you have been explicitly told so. Do not do them on your own.

# TODO
- test legacy / fallback paths more systematically
  - legacy env classes in `src/spotmicro/env/spotmicro_env_pybullet.py` and `src/spotmicro/env/spotmicro_env_mujoco.py`
  - legacy agent classes in `src/spotmicro/agent/agent_pybullet.py` and `src/spotmicro/agent/agent_mujoco.py`
  - older training scripts that still rely on pre-backend-abstraction patterns
- decide whether to keep those legacy paths or fully standardize on `src/spotmicro/env/spotmicro_env.py`
- create a cleaner dedicated robust training entry point outside the notebook/test flow
  - e.g. `training/robust_training.py`
  - keep project structure changes minimal
- add cleaner long-run evaluation / checkpoint wiring for robust training
  - periodic checkpoints
  - separate eval env
  - clearer logs for PyBullet vs MuJoCo
- run longer production-style training validations
  - longer than the smoke/stability runs already tested in this session
  - compare PyBullet and MuJoCo learning curves over larger horizons
- review whether all runtime-only args using `@configurable` are excluded where appropriate
- consider adding a config-template export feature later
  - separate from the current resolved-config save behavior
