"""Build environments and curricula from a run config, and handle checkpoint state files.

Shared by ``train.py`` and ``evaluate.py``.
"""

import random
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

from commands.device import Device
from spotmicro.env import SpotmicroEnv
from training.config import RunConfig, load_config
from training.curriculum import (
    CompetenceProgress,
    Curriculum,
    CurriculumEnv,
    Progress,
    TimeProgress,
)

ML_ROOT = Path(__file__).resolve().parents[1]  # ml_training/


def make_env(
    config: RunConfig, gui: bool | None = None, command: Device | None = None
) -> SpotmicroEnv:
    """The environment described by ``config`` (``gui``/``command`` override the config)."""
    return SpotmicroEnv(
        command=command if command is not None else config.command.make(),
        reward=config.reward.make(),
        params=config.env,
        agent_params=config.agent,
        use_gui=config.run.gui if gui is None else gui,
    )


def env_factory(config: RunConfig, rank: int, subprocess: bool = False) -> Callable[[], gym.Env]:
    """A function creating training environment number ``rank`` (for SB3's VecEnvs)."""

    def make() -> gym.Env:
        if subprocess:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        if config.run.seed is not None:
            np.random.seed(config.run.seed + rank)
            random.seed(config.run.seed + rank)
        env = make_env(config)
        env.steps_per_step = config.run.n_envs
        if config.curriculum is not None:
            return CurriculumEnv(env, make_curricula(config))
        return env

    return make


def make_progress(config: RunConfig) -> Progress:
    curriculum = config.curriculum
    if curriculum is not None and curriculum.progression == "competence":
        return CompetenceProgress(curriculum.competence)
    return TimeProgress(config.run.total_steps)


def make_curricula(config: RunConfig) -> dict[str, Curriculum]:
    if config.curriculum is None:
        return {}
    return {name: params.make() for name, params in config.curriculum.curricula.items()}


def resolve_path(path: str | Path) -> Path:
    """Paths in configs are relative to ml_training/."""
    path = Path(path)
    return path if path.is_absolute() else ML_ROOT / path


def run_dir_of(checkpoint: str | Path) -> Path:
    """The run folder holding ``checkpoint`` (the nearest ancestor with a config.yaml)."""
    for parent in Path(checkpoint).resolve().parents:
        if (parent / "config.yaml").exists():
            return parent
    raise FileNotFoundError(f"{checkpoint} is not inside a run folder (no config.yaml above it)")


def load_run(checkpoint: str | Path) -> RunConfig:
    """The config of the run that wrote ``checkpoint``."""
    return load_config(run_dir_of(checkpoint) / "config.yaml")


def state_path(model_path: str | Path) -> Path:
    """``foo.zip`` (or ``foo``) -> ``foo.state.yaml``."""
    model_path = Path(model_path)
    stem = model_path.with_suffix("") if model_path.suffix == ".zip" else model_path
    return stem.with_name(stem.name + ".state.yaml")


def _plain(value: Any) -> Any:
    """numpy scalars -> Python numbers, recursively (YAML safe_dump rejects numpy types)."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_state(
    path: Path, num_timesteps: int, env_steps: int, curriculum: dict[str, Any] | None
) -> None:
    """Save what a checkpoint needs besides the model: step counts and curriculum state."""
    state = {
        "num_timesteps": int(num_timesteps),
        "env_steps": int(env_steps),
        "curriculum": curriculum,
    }
    with open(path, "w") as f:
        yaml.safe_dump(_plain(state), f, sort_keys=False)


def read_state(model_path: str | Path) -> dict[str, Any]:
    """The state file train.py saved next to a model (step counts, curriculum state)."""
    with open(state_path(model_path)) as f:
        return yaml.safe_load(f)
