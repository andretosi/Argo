"""Curriculum learning: make training conditions harder as the policy improves.

A progress tracker turns training into a number in [0, 1] (``TimeProgress``: fraction of the
total steps; ``CompetenceProgress``: advances when the policy performs well). Each curriculum
maps that progress to a difficulty factor (after an optional warmup, linearly or
quadratically) and interpolates its settings between their initial and final values.

During training, each environment is wrapped in a ``CurriculumEnv`` that runs its own copy
of the curricula (so they also work with environments in separate processes); the training
callback in ``train.py`` tracks progress from the episode scores and sends it to every
environment. Each curriculum also saves and restores its exact state, so a run can be
resumed or replayed under the same conditions.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import numpy as np

from spotmicro.terrain import Heightmap

if TYPE_CHECKING:
    from spotmicro.env import SpotmicroEnv

_logger = logging.getLogger(__name__)


def interpolate(span: tuple[float, float], factor: float) -> float:
    """Value at ``factor`` (0..1) between ``span = (initial, final)``."""
    return span[0] + factor * (span[1] - span[0])


# ── Progress ─────────────────────────────────────────────────────────────────


class TimeProgress:
    """Progress = fraction of the total training steps done."""

    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self.value = 0.0

    def on_step(self, num_timesteps: int) -> None:
        self.value = min(1.0, num_timesteps / max(1, self.total_steps))

    def on_episode_end(self, scores: dict[str, float]) -> None:
        pass

    def metrics(self) -> dict[str, float]:
        return {"curriculum/progress": self.value}

    def state_dict(self) -> dict[str, Any]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.value = float(state["value"])


class CompetenceProgress:
    """Progress that advances by ``advance_step`` whenever the policy is competent enough.

    Each finished episode gets a score in [0, 1] from velocity tracking (reward components
    ``tracking_lin_vel``/``tracking_ang_vel``), survival (episode length) and stability
    (``orientation``/``base_height`` penalties); its exponential moving average must reach
    ``threshold``, and advances are at least ``min_episodes_between_advances`` episodes apart.
    """

    @dataclass
    class Params:
        ema_alpha: float = 0.10
        threshold: float = 0.70
        advance_step: float = 0.05
        min_episodes_between_advances: int = 10

    def __init__(self, params: "CompetenceProgress.Params | None" = None) -> None:
        self.params = params if params is not None else self.Params()
        self.value = 0.0
        self.ema: float | None = None
        self.raw_score: float | None = None
        self.episodes = 0
        self.last_advance_episode = 0
        self._scores: dict[str, float] = {}

    def on_step(self, num_timesteps: int) -> None:
        pass

    @staticmethod
    def _mean(history: list[dict[str, float]], key: str) -> float:
        if not history:
            return 0.0
        return float(np.mean([float(step.get(key, 0.0)) for step in history]))

    def score_episode(self, env: "SpotmicroEnv") -> dict[str, float]:
        """Score the env's previous episode (the one before its latest reset)."""
        return self.score(
            env.last_episode_reward_info,
            env.last_episode_length,
            env.max_episode_len,
            env.reward.params,
        )

    @classmethod
    def score(
        cls, history: list[dict[str, float]], length: int, max_length: int, weights: Any
    ) -> dict[str, float]:
        """Score one episode from its per-step reward components and its length."""
        _mean = cls._mean
        lin_weight = float(getattr(weights, "tracking_lin_vel", 1.0) or 1.0)
        ang_weight = float(getattr(weights, "tracking_ang_vel", 1.0) or 1.0)
        tracking_lin = np.clip(_mean(history, "tracking_lin_vel") / lin_weight, 0.0, 1.0)
        tracking_ang = np.clip(_mean(history, "tracking_ang_vel") / ang_weight, 0.0, 1.0)
        tracking = float(0.5 * (tracking_lin + tracking_ang))
        survival = float(np.clip(length / max(1, max_length), 0, 1))
        orientation = float(np.exp(-max(0.0, -_mean(history, "orientation"))))
        base_height = float(np.exp(-max(0.0, -_mean(history, "base_height"))))
        stability = float(np.clip(0.5 * (orientation + base_height), 0.0, 1.0))
        raw = float(np.clip(0.5 * tracking + 0.3 * survival + 0.2 * stability, 0.0, 1.0))
        return {"tracking": tracking, "survival": survival, "stability": stability, "raw": raw}

    def update(self, raw_score: float) -> None:
        """Fold one episode's score into the EMA and advance progress if competent."""
        p = self.params
        self.episodes += 1
        self.raw_score = float(np.clip(raw_score, 0.0, 1.0))
        if self.ema is None:
            self.ema = self.raw_score
        else:
            self.ema = p.ema_alpha * self.raw_score + (1.0 - p.ema_alpha) * self.ema
        if (
            self.ema >= p.threshold
            and self.episodes - self.last_advance_episode >= p.min_episodes_between_advances
        ):
            self.value = min(1.0, self.value + p.advance_step)
            self.last_advance_episode = self.episodes

    def on_episode_end(self, scores: dict[str, float]) -> None:
        self._scores = dict(scores)
        self.update(self._scores["raw"])

    def metrics(self) -> dict[str, float]:
        out = {"curriculum/progress": self.value}
        if self.ema is not None:
            out["competence/ema"] = self.ema
        out.update({f"competence/{k}_score": v for k, v in self._scores.items()})
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "ema": self.ema,
            "raw_score": self.raw_score,
            "episodes": self.episodes,
            "last_advance_episode": self.last_advance_episode,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.value = float(state["value"])
        self.ema = state["ema"]
        self.raw_score = state["raw_score"]
        self.episodes = int(state["episodes"])
        self.last_advance_episode = int(state["last_advance_episode"])


Progress = TimeProgress | CompetenceProgress


# ── Curricula ────────────────────────────────────────────────────────────────


@dataclass
class CurriculumConfig:
    """The parameters of a curriculum: the base of every curriculum's nested ``Params``.

    In a run config they are the values under ``curriculum: <name>:``. ``name`` and the
    curriculum built by ``make()`` come from the curriculum class the ``Params`` belong to.
    """

    name: ClassVar[str] = ""
    curriculum_class: ClassVar[type["Curriculum"]]

    schedule: str = "linear"  # "linear" or "exponential" (quadratic: slow start)
    warmup: float = 0.05  # progress below this keeps the initial difficulty

    def make(self) -> "Curriculum":
        return self.curriculum_class(self)


class Curriculum(ABC):
    """One aspect of the training conditions that gets harder with progress.

    Subclasses set ``name`` (the key under ``curriculum:`` in a run config) and define a
    nested ``Params(CurriculumConfig)`` dataclass.

    Lifecycle: ``attach(env)`` once (applies the current level), then every control step
    ``set_factor(factor)`` + ``on_step(env)``, and ``on_episode_end(env)`` after each episode.
    For evaluation, load a saved state and call only ``attach`` and ``on_step``.
    """

    name: ClassVar[str] = ""
    Params: ClassVar[type[CurriculumConfig]] = CurriculumConfig
    params: Any

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "Params" in vars(cls):  # link the parameters to their curriculum
            cls.Params.name = cls.name
            cls.Params.curriculum_class = cls

    def __init__(self, params: Any = None) -> None:
        self.params = params if params is not None else self.Params()
        if self.params.schedule not in ("linear", "exponential"):
            raise ValueError(
                f"schedule must be 'linear' or 'exponential', got {self.params.schedule!r}"
            )
        self.factor = 0.0

    def factor_at(self, progress: float) -> float:
        """Difficulty factor (0..1) for a progress value (0..1)."""
        progress = float(np.clip(progress, 0.0, 1.0))
        if progress < self.params.warmup:
            return 0.0
        adjusted = (progress - self.params.warmup) / (1.0 - self.params.warmup)
        return adjusted**2 if self.params.schedule == "exponential" else adjusted

    def set_factor(self, factor: float) -> None:
        self.factor = float(factor)

    def jump_to_factor(self, factor: float) -> None:
        """Set the factor and also move any sampled condition there now (for evaluation)."""
        self.set_factor(factor)

    def attach(self, env: "SpotmicroEnv") -> None:
        """Apply the current level to a freshly created environment."""

    def on_step(self, env: "SpotmicroEnv") -> None:
        """Called once per control step."""

    def on_episode_end(self, env: "SpotmicroEnv") -> None:
        """Called after every episode (training only)."""

    @abstractmethod
    def metrics(self) -> dict[str, float]: ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]: ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class TerrainCurriculum(Curriculum):
    """Random noise terrain, regenerated every ``change_every_episodes`` episodes."""

    name = "terrain"

    @dataclass
    class Params(CurriculumConfig):
        z_max: tuple[float, float] = (0.02, 0.3)  # max terrain height [m]: (initial, final)
        change_every_episodes: int = 50
        terrain_size: int = 256  # heightmap resolution (cells per side)
        scale: tuple[float, float, float] = (0.02, 0.02, 1.0)  # cell size x, y [m]; height scale
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    params: Params

    def __init__(self, params: Params | None = None) -> None:
        super().__init__(params)
        self.z_max = self.params.z_max[0]  # height of the terrain currently spawned
        self.seed: int | None = None
        self.episodes_since_change = 0
        self.changes = 0
        self._handle: int | None = None

    def spawn(self, env: "SpotmicroEnv", z_max: float, seed: int) -> None:
        p = self.params
        heightmap = Heightmap.from_noise(x=p.terrain_size, y=p.terrain_size, z_max=z_max, seed=seed)
        try:
            if self._handle is not None:
                env.sim.remove_terrain(self._handle)
            self._handle = env.sim.spawn_terrain(heightmap.data, p.scale, p.origin)
        except Exception:
            _logger.warning("[TerrainCurriculum] Could not spawn terrain", exc_info=True)
            return
        self.z_max, self.seed = float(z_max), int(seed)
        self.changes += 1

    def jump_to_factor(self, factor: float) -> None:
        super().jump_to_factor(factor)
        self.z_max = interpolate(self.params.z_max, self.factor)
        self.seed = None  # new random terrain at that height

    def attach(self, env: "SpotmicroEnv") -> None:
        if self.seed is None:
            self.seed = int(np.random.randint(0, 1_000_000))
        self.spawn(env, self.z_max, self.seed)

    def on_episode_end(self, env: "SpotmicroEnv") -> None:
        self.episodes_since_change += 1
        if self.episodes_since_change >= self.params.change_every_episodes:
            self.episodes_since_change = 0
            seed = int(np.random.randint(0, 1_000_000))
            self.spawn(env, interpolate(self.params.z_max, self.factor), seed)

    def metrics(self) -> dict[str, float]:
        return {"curriculum/terrain_z_max": self.z_max, "curriculum/terrain_changes": self.changes}

    def state_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "z_max": self.z_max,
            "seed": self.seed,
            "episodes_since_change": self.episodes_since_change,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.factor = float(state["factor"])
        self.z_max = float(state["z_max"])
        self.seed = state["seed"]
        self.episodes_since_change = int(state["episodes_since_change"])


class PushCurriculum(Curriculum):
    """Random horizontal pushes on the body, every ~``interval_s`` seconds.

    The force is sized to change a free body's velocity by 50-100% of the current push
    velocity over ``duration_steps`` control steps.
    """

    name = "push"

    @dataclass
    class Params(CurriculumConfig):
        velocity: tuple[float, float] = (0.1, 1.5)  # push strength [m/s]: (initial, final)
        interval_s: float = 15.0
        interval_range: tuple[float, float] = (0.8, 1.2)  # random factor on the interval
        duration_steps: int = 2

    params: Params

    def __init__(self, params: Params | None = None) -> None:
        super().__init__(params)
        self.velocity = self.params.velocity[0]
        self.pushes = 0
        self._mass = 2.5
        self._control_frequency = 60
        self._steps_since_push = 0
        self._next_push_at = 0
        self._push_steps_remaining = 0
        self._force = np.zeros(3)

    def set_factor(self, factor: float) -> None:
        super().set_factor(factor)
        self.velocity = interpolate(self.params.velocity, factor)

    def _sample_interval(self) -> int:
        base = int(self.params.interval_s * self._control_frequency)
        return int(base * np.random.uniform(*self.params.interval_range))

    def attach(self, env: "SpotmicroEnv") -> None:
        self._mass = env.sim.get_base_mass()
        self._control_frequency = env.control_frequency
        self._next_push_at = self._sample_interval()

    def on_step(self, env: "SpotmicroEnv") -> None:
        self._steps_since_push += 1
        if self._push_steps_remaining > 0:
            env.sim.apply_external_force(self._force)
            self._push_steps_remaining -= 1
            return
        if self._steps_since_push >= self._next_push_at:
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.5, 1.0) * self.velocity
            duration = self.params.duration_steps / self._control_frequency
            magnitude = self._mass * speed / duration
            self._force = np.array([magnitude * np.cos(angle), magnitude * np.sin(angle), 0.0])
            self._push_steps_remaining = self.params.duration_steps - 1
            self._steps_since_push = 0
            self._next_push_at = self._sample_interval()
            env.sim.apply_external_force(self._force)
            self.pushes += 1

    def on_episode_end(self, env: "SpotmicroEnv") -> None:
        self._push_steps_remaining = 0
        self._force = np.zeros(3)

    def metrics(self) -> dict[str, float]:
        return {"curriculum/push_velocity": self.velocity, "curriculum/push_count": self.pushes}

    def state_dict(self) -> dict[str, Any]:
        return {"factor": self.factor, "velocity": self.velocity}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.factor = float(state["factor"])
        self.velocity = float(state["velocity"])


class FrictionCurriculum(Curriculum):
    """Ground friction resampled every episode from a range that widens with progress."""

    name = "friction"

    @dataclass
    class Params(CurriculumConfig):
        low: tuple[float, float] = (0.9, 0.4)  # lower bound of the range: (initial, final)
        high: tuple[float, float] = (1.1, 1.5)  # upper bound of the range: (initial, final)

    params: Params

    def __init__(self, params: Params | None = None) -> None:
        super().__init__(params)
        self.friction: float | None = None

    def _apply(self, env: "SpotmicroEnv", friction: float) -> None:
        try:
            env.sim.set_friction(friction)
            self.friction = float(friction)
        except Exception:
            _logger.warning("[FrictionCurriculum] Error setting friction", exc_info=True)

    def _sample(self) -> float:
        low = interpolate(self.params.low, self.factor)
        high = interpolate(self.params.high, self.factor)
        return float(np.random.uniform(low, high))

    def jump_to_factor(self, factor: float) -> None:
        super().jump_to_factor(factor)
        self.friction = None  # resampled from the range at this factor on attach

    def attach(self, env: "SpotmicroEnv") -> None:
        self._apply(env, self.friction if self.friction is not None else self._sample())

    def on_episode_end(self, env: "SpotmicroEnv") -> None:
        self._apply(env, self._sample())

    def metrics(self) -> dict[str, float]:
        return {"curriculum/friction": self.friction} if self.friction is not None else {}

    def state_dict(self) -> dict[str, Any]:
        return {"factor": self.factor, "friction": self.friction}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.factor = float(state["factor"])
        self.friction = state["friction"]


class MotorNoiseCurriculum(Curriculum):
    """Noise added to every action before it reaches the motors."""

    name = "motor_noise"

    @dataclass
    class Params(CurriculumConfig):
        std: tuple[float, float] = (0.0, 0.05)  # noise size (normalized action units)
        noise_type: str = "gaussian"  # or "uniform" (std is then the half-width)

    params: Params

    def __init__(self, params: Params | None = None) -> None:
        super().__init__(params)
        if self.params.noise_type not in ("gaussian", "uniform"):
            raise ValueError(
                f"noise_type must be 'gaussian' or 'uniform', got {self.params.noise_type!r}"
            )
        self.std = self.params.std[0]

    def set_factor(self, factor: float) -> None:
        super().set_factor(factor)
        self.std = interpolate(self.params.std, factor)

    def _noisy(self, action: np.ndarray) -> np.ndarray:
        if self.std <= 0:
            return action
        if self.params.noise_type == "gaussian":
            noise = np.random.normal(0, self.std, action.shape)
        else:
            noise = np.random.uniform(-self.std, self.std, action.shape)
        return np.clip(action + noise, -1.0, 1.0)

    def attach(self, env: "SpotmicroEnv") -> None:
        env.action_filters.append(self._noisy)

    def metrics(self) -> dict[str, float]:
        return {"curriculum/motor_noise_std": self.std}

    def state_dict(self) -> dict[str, Any]:
        return {"factor": self.factor, "std": self.std}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.factor = float(state["factor"])
        self.std = float(state["std"])


class SensorNoiseCurriculum(Curriculum):
    """Noise added to the observation; each block's noise is its base level times ``scale``."""

    name = "sensor_noise"

    @dataclass
    class Params(CurriculumConfig):
        scale: tuple[float, float] = (0.0, 1.0)  # global noise multiplier: (initial, final)
        dof_pos_noise: float = 0.01  # joint positions [rad]
        dof_vel_noise: float = 1.5  # joint velocities [rad/s]
        lin_vel_noise: float = 0.1  # base linear velocity [m/s]
        ang_vel_noise: float = 0.2  # base angular velocity [rad/s]
        gravity_noise: float = 0.05
        height_noise: float = 0.02

    params: Params

    def __init__(self, params: Params | None = None) -> None:
        super().__init__(params)
        self.scale = self.params.scale[0]

    def set_factor(self, factor: float) -> None:
        super().set_factor(factor)
        self.scale = interpolate(self.params.scale, factor)

    def _noisy(self, obs: np.ndarray) -> np.ndarray:
        if self.scale <= 0:
            return obs
        p, s = self.params, self.scale
        noisy = obs.copy()
        noisy[0:3] += np.random.normal(0, p.gravity_noise * s, 3)  # gravity vector
        norm = np.linalg.norm(noisy[0:3])
        if norm > 1e-8:
            noisy[0:3] /= norm
        noisy[3] += np.random.normal(0, p.height_noise * s)  # height
        noisy[4:7] += np.random.normal(0, p.lin_vel_noise * s, 3)  # base linear velocity
        noisy[7:10] += np.random.normal(0, p.ang_vel_noise * s, 3)  # base angular velocity
        noisy[10:22] += np.random.normal(0, p.dof_pos_noise * s, 12)  # joint positions
        noisy[22:34] += np.random.normal(0, p.dof_vel_noise * s, 12)  # joint velocities
        noisy[34:46] += np.random.normal(0, p.dof_pos_noise * s, 12)  # history positions
        noisy[58:70] += np.random.normal(0, p.dof_pos_noise * s, 12)
        noisy[46:58] += np.random.normal(0, p.dof_vel_noise * s, 12)  # history velocities
        noisy[70:82] += np.random.normal(0, p.dof_vel_noise * s, 12)
        return noisy

    def attach(self, env: "SpotmicroEnv") -> None:
        env.observation_filters.append(self._noisy)

    def metrics(self) -> dict[str, float]:
        return {"curriculum/sensor_noise_scale": self.scale}

    def state_dict(self) -> dict[str, Any]:
        return {"factor": self.factor, "scale": self.scale}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.factor = float(state["factor"])
        self.scale = float(state["scale"])


CURRICULA: dict[str, type[Curriculum]] = {
    curriculum.name: curriculum
    for curriculum in (
        TerrainCurriculum,
        PushCurriculum,
        FrictionCurriculum,
        MotorNoiseCurriculum,
        SensorNoiseCurriculum,
    )
}


# ── Environment wrapper ──────────────────────────────────────────────────────


class CurriculumEnv(gym.Wrapper):
    """Runs a set of curricula on the environment it wraps.

    Each environment (also one in a separate process) gets its own copy of the curricula.
    The training callback calls ``start_curriculum`` once and ``set_progress`` whenever the
    progress changes; at the end of every episode, ``info["curriculum"]`` carries the
    episode's competence scores and the curricula metrics.
    """

    def __init__(self, env: "SpotmicroEnv", curricula: dict[str, Curriculum]) -> None:
        super().__init__(env)
        self.curricula = curricula
        self._env: SpotmicroEnv = env.unwrapped  # type: ignore[assignment]

    def start_curriculum(self, progress: float, state: dict[str, Any] | None = None) -> None:
        """Restore saved curriculum state (when resuming), set the level and apply it."""
        if state is not None:
            for name, curriculum in self.curricula.items():
                curriculum.load_state_dict(state[name])
        self.set_progress(progress)
        for curriculum in self.curricula.values():
            curriculum.attach(self._env)

    def set_progress(self, progress: float) -> None:
        for curriculum in self.curricula.values():
            curriculum.set_factor(curriculum.factor_at(progress))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        env = self._env
        for curriculum in self.curricula.values():
            curriculum.on_step(env)
        if terminated or truncated:
            scores = CompetenceProgress.score(
                env.episode_reward_info, env.episode_length, env.max_episode_len, env.reward.params
            )
            for curriculum in self.curricula.values():
                curriculum.on_episode_end(env)
            info["curriculum"] = {"scores": scores, "metrics": self.curriculum_metrics()}
        return obs, reward, terminated, truncated, info

    def curriculum_metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for curriculum in self.curricula.values():
            out.update(curriculum.metrics())
        return out

    def curriculum_state(self) -> dict[str, Any]:
        return {name: c.state_dict() for name, c in self.curricula.items()}
