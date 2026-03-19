"""
Base Curriculum Callback
========================

Provides shared curriculum progression logic for all curriculum callbacks.

Usage:
    from training.callbacks.base_curriculum import BaseCurriculumCallback

    class MyCallback(BaseCurriculumCallback):
        def _apply_curriculum(self, env, factor: float):
            # factor is 0.0 at start, 1.0 at end
            ...
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from abc import abstractmethod

from spotmicro.tools.config import Config


class BaseCurriculumCallback(BaseCallback):
    """
    Base class for curriculum learning callbacks.

    Provides:
    - Progress tracking (timesteps/episodes)
    - Curriculum factor calculation with warmup
    - Linear/exponential scheduling
    - Environment unwrapping utilities

    Subclasses must implement:
    - _apply_curriculum(env, factor): Apply curriculum at given factor (0.0-1.0)
    - _on_episode_end(env): Called at end of each episode
    """
    __config_exclude__ = {"env"}

    def __init__(
        self,
        config: Config,
        env=None,
        total_timesteps: int = 1_000_000,
        schedule: str = "linear",
        warmup_ratio: float = 0.05,
        verbose: bool = True,
    ):
        """
        Initialize base curriculum callback.

        Parameters
        ----------
        config : Config
            Central config registry for saving/loading parameters
        env : SpotmicroEnv or VecEnv wrapper
            The training environment
        total_timesteps : int
            Estimated total training timesteps for scheduling
        schedule : str
            Curriculum schedule type: "linear" or "exponential"
        warmup_ratio : float
            Fraction of training before curriculum starts ramping (default: 0.05)
        verbose : bool
            Print curriculum updates (default: True)
        """
        super().__init__(verbose)

        self.env = env
        self.total_timesteps = total_timesteps
        self.schedule = schedule
        self.warmup_ratio = warmup_ratio

        # Internal state
        self._timesteps = 0
        self._episode_count = 0
        self._initialized = False

    def _get_unwrapped_env(self):
        """Get the underlying SpotmicroEnv from possible wrappers."""
        env = self.env
        if hasattr(env, "envs"):
            env = env.envs[0]
        if hasattr(env, "env"):
            env = env.env
        return env

    def _get_progress(self) -> float:
        """Get normalized training progress (0.0 to 1.0)."""
        return min(1.0, self._timesteps / self.total_timesteps)

    def _get_curriculum_factor(self) -> float:
        """
        Get curriculum interpolation factor with warmup.

        Returns 0.0 during warmup, then ramps from 0 to 1.
        """
        progress = self._get_progress()

        # During warmup, stay at initial difficulty
        if progress < self.warmup_ratio:
            return 0.0

        # After warmup, ramp from 0 to 1
        adjusted_progress = (progress - self.warmup_ratio) / (1.0 - self.warmup_ratio)

        if self.schedule == "linear":
            return adjusted_progress
        elif self.schedule == "exponential":
            # Slow start, fast finish
            return adjusted_progress ** 2
        else:
            return adjusted_progress

    def _interpolate(self, initial: float, final: float, factor: float) -> float:
        """Linearly interpolate between initial and final values."""
        return initial + factor * (final - initial)

    def _sync_config(self, **params) -> None:
        """Mirror serializable runtime state into the shared Config registry."""
        serializable = {
            name: value for name, value in params.items()
            if self.config.is_acceptable(value)
        }
        if not serializable:
            return

        for name, value in serializable.items():
            setattr(self, name, value)
        self.config.update(self, serializable)

    def _record_metrics(self, metrics: dict) -> None:
        """Record scalar curriculum values into the SB3 logger."""
        if not metrics or self.logger is None:
            return

        for name, value in metrics.items():
            if value is None:
                continue

            if isinstance(value, np.ndarray):
                if value.size != 1:
                    continue
                value = float(value.reshape(-1)[0])
            elif np.isscalar(value):
                value = float(value)
            else:
                continue

            self.logger.record(name, value)

    def _lazy_init(self):
        """Initialize when environment is available. Override in subclasses."""
        if self._initialized:
            return
        self._initialized = True

    @abstractmethod
    def _apply_curriculum(self, env, factor: float) -> None:
        """Apply curriculum at given factor. Must be implemented by subclasses."""
        ...

    def _on_episode_end(self, env) -> None:
        """Called at end of each episode. Override in subclasses."""
        pass

    def _on_training_start(self) -> None:
        """Called at start of training."""
        self._lazy_init()

    def _on_step(self) -> bool:
        """Called after each environment step."""
        self._lazy_init()
        self._timesteps += 1

        unwrapped = self._get_unwrapped_env()
        if unwrapped is None:
            return True

        # Apply curriculum
        factor = self._get_curriculum_factor()
        self._apply_curriculum(unwrapped, factor)

        # Check episode end
        dones = self.locals.get("dones", [False])
        if any(dones):
            self._episode_count += 1
            self._on_episode_end(unwrapped)

        return True
