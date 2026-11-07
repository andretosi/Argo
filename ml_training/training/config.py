"""Run configuration: one YAML file describes a training run.

Top-level sections (only ``command`` and ``reward`` are required)::

    run:        name, seed, total steps, parallel environments, init_from, checkpoints, gui
    env:        SpotmicroEnv parameters (EnvParams)
    agent:      robot parameters (AgentParams)
    command:    exactly one command source, e.g. ``random: {...}``
    reward:     exactly one reward, e.g. ``robust_walking: {...}``
    curriculum: optional; progression mode, competence settings, and any curricula
    ppo:        PPO hyperparameters

Every section is validated: unknown names or parameters are errors, and anything left out
keeps its default. ``save_config`` writes the fully resolved config (defaults included).
"""

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

from commands import COMMANDS, DeviceConfig
from spotmicro.agent import AgentParams
from spotmicro.env import EnvParams
from training.curriculum import CURRICULA, CompetenceProgress, CurriculumConfig
from training.rewards import REWARDS, RewardConfig

T = TypeVar("T")


class ConfigError(ValueError):
    """The run config is malformed."""


def build(cls: type[T], values: dict[str, Any] | None, where: str) -> T:
    """Create dataclass ``cls`` from ``values``; unknown keys are errors, missing keys keep defaults."""
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ConfigError(f"{where}: expected a mapping of parameters, got {values!r}")
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(values) - set(names))
    if unknown:
        raise ConfigError(f"{where}: unknown parameter(s) {unknown}; valid: {names}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, value in values.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = build(hint, value, f"{where}.{name}")
        elif isinstance(value, list) and typing.get_origin(hint) is tuple:
            kwargs[name] = tuple(value)
        elif isinstance(value, str) and float in (typing.get_args(hint) or (hint,)):
            # PyYAML reads e.g. `3e-4` (no dot) as a string
            try:
                kwargs[name] = float(value)
            except ValueError:
                raise ConfigError(f"{where}.{name}: expected a number, got {value!r}") from None
        else:
            kwargs[name] = value
    return cls(**kwargs)


def build_component(section: Any, registry: dict[str, Any], where: str) -> Any:
    """The parameters of the one component a section names, e.g. ``{"standing": {...}}``."""
    if not isinstance(section, dict) or len(section) != 1:
        raise ConfigError(
            f"{where}: give exactly one of {sorted(registry)}, with its parameters below it"
        )
    ((name, values),) = section.items()
    if name not in registry:
        raise ConfigError(f"{where}: unknown {name!r}; choose one of {sorted(registry)}")
    return build(registry[name].Params, values, f"{where}.{name}")


@dataclass
class Checkpoints:
    every_steps: int = 500_000  # 0 disables step checkpoints
    every_minutes: float = 5.0  # 0 disables timed checkpoints


@dataclass
class RunSection:
    name: str = "run"
    seed: int | None = 0
    total_steps: int = 1_000_000
    # Simulations run in parallel, one process each; PPO collects ppo.n_steps from each one,
    # so a rollout has n_envs * n_steps steps
    n_envs: int = 1
    # PyTorch CPU threads for the policy; 1 is fastest for small networks and batches
    torch_threads: int = 1
    init_from: str | None = None  # start from another policy (.zip), e.g. stand -> trot
    output_dir: str = (
        "training/data"  # relative to ml_training/; the run goes in <output_dir>/<name>
    )
    gui: bool = False
    checkpoints: Checkpoints = field(default_factory=Checkpoints)


@dataclass
class PPOParams:
    learning_rate: float = 3e-4
    learning_rate_final: float | None = None  # None: constant; else linear decay down to this
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Log of the policy's initial action std (SB3's default: 0, std 1). Lower starts the
    # robot near the homing pose (action 0) instead of flailing; ignored by init_from/resume.
    log_std_init: float = 0.0
    device: str = "cpu"
    verbose: int = 1


@dataclass
class CurriculumSection:
    progression: str = "time"  # "time" (fraction of total steps) or "competence"
    competence: CompetenceProgress.Params = field(default_factory=CompetenceProgress.Params)
    curricula: dict[str, CurriculumConfig] = field(default_factory=dict)  # by name


@dataclass
class RunConfig:
    command: DeviceConfig
    reward: RewardConfig
    run: RunSection = field(default_factory=RunSection)
    env: EnvParams = field(default_factory=EnvParams)
    agent: AgentParams = field(default_factory=AgentParams)
    curriculum: CurriculumSection | None = None
    ppo: PPOParams = field(default_factory=PPOParams)


SECTIONS = ("run", "env", "agent", "command", "reward", "curriculum", "ppo")


def _build_curriculum(section: Any) -> CurriculumSection | None:
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ConfigError("curriculum: expected a mapping")
    section = dict(section)
    progression = section.pop("progression", "time")
    if progression not in ("time", "competence"):
        raise ConfigError(f"curriculum.progression: 'time' or 'competence', got {progression!r}")
    competence = build(
        CompetenceProgress.Params, section.pop("competence", None), "curriculum.competence"
    )
    curricula = {}
    for name, values in section.items():
        if name not in CURRICULA:
            raise ConfigError(
                f"curriculum: unknown {name!r}; valid: "
                f"{['progression', 'competence', *sorted(CURRICULA)]}"
            )
        curricula[name] = build(CURRICULA[name].Params, values, f"curriculum.{name}")
    return CurriculumSection(progression, competence, curricula)


def config_from_dict(data: dict[str, Any]) -> RunConfig:
    if not isinstance(data, dict):
        raise ConfigError("the config file must be a mapping of sections")
    unknown = sorted(set(data) - set(SECTIONS))
    if unknown:
        raise ConfigError(f"unknown section(s) {unknown}; valid: {list(SECTIONS)}")
    for required in ("command", "reward"):
        if required not in data:
            raise ConfigError(f"missing required section {required!r}")
    config = RunConfig(
        command=build_component(data["command"], COMMANDS, "command"),
        reward=build_component(data["reward"], REWARDS, "reward"),
        run=build(RunSection, data.get("run"), "run"),
        env=build(EnvParams, data.get("env"), "env"),
        agent=build(AgentParams, data.get("agent"), "agent"),
        curriculum=_build_curriculum(data.get("curriculum")),
        ppo=build(PPOParams, data.get("ppo"), "ppo"),
    )
    if config.run.n_envs < 1:
        raise ConfigError(f"run.n_envs: must be at least 1, got {config.run.n_envs}")
    return config


def load_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    try:
        return config_from_dict(data)
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from None


def _plain(value: Any) -> Any:
    """Dataclasses -> dicts, tuples -> lists, recursively (for YAML)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def config_to_dict(config: RunConfig) -> dict[str, Any]:
    data: dict[str, Any] = {
        "run": _plain(config.run),
        "env": _plain(config.env),
        "agent": _plain(config.agent),
        "command": {config.command.name: _plain(config.command)},
        "reward": {config.reward.name: _plain(config.reward)},
    }
    if config.curriculum is not None:
        c = config.curriculum
        data["curriculum"] = {
            "progression": c.progression,
            "competence": _plain(c.competence),
            **{name: _plain(params) for name, params in c.curricula.items()},
        }
    data["ppo"] = _plain(config.ppo)
    return data


def save_config(config: RunConfig, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(config_to_dict(config), f, sort_keys=False, default_flow_style=None)
