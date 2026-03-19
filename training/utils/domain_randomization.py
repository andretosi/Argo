"""
Domain Randomization Utilities for Robust Training
===================================================

Provides perturbation and randomization utilities following legged_gym best practices.

Classes:
    PushPerturbation: Interval-based force perturbation during training
    FrictionRandomizer: Randomizes ground friction at episode start

Usage:
    from training.utils.domain_randomization import PushPerturbation
    from spotmicro.tools.config import Config

    cfg = Config()
    perturbation = PushPerturbation(
        config=cfg,
        push_interval_s=15.0,
        max_push_vel_xy=1.0,
        control_freq=60
    )

    # In training loop or callback:
    perturbation.maybe_apply_push(env)
"""

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spotmicro.tools.config import Config
from spotmicro.tools.configurable import configurable

if TYPE_CHECKING:
    from spotmicro.env.spotmicro_env import SpotmicroEnv


@dataclass
class PushPerturbationConfig:
    """Configuration for push perturbations (legacy dataclass for backwards compat)."""
    push_interval_s: float = 15.0
    max_push_vel_xy: float = 1.0
    push_duration_steps: int = 2
    push_interval_range: tuple = (0.8, 1.2)


@configurable
class PushPerturbation:
    """
    Applies random impulse forces to the robot at regular intervals.

    Based on ETH/NVIDIA legged_gym implementation:
    - Push every ~15 seconds by default
    - Apply force to achieve target velocity change
    - Force = mass * delta_v / dt

    Parameters
    ----------
    config : Config
        Central config registry for saving/loading parameters
    push_interval_s : float
        Time between pushes in seconds (default: 15.0)
    max_push_vel_xy : float
        Max velocity change in m/s (default: 1.0)
    push_duration_steps : int
        Number of steps to apply force (default: 2)
    push_interval_range_low : float
        Low multiplier for random interval (default: 0.8)
    push_interval_range_high : float
        High multiplier for random interval (default: 1.2)
    control_freq : int
        Control frequency in Hz (default: 60)
    robot_mass : float
        Robot mass in kg for force calculation (default: 2.5)
    """

    def __init__(
        self,
        config: Config,
        push_interval_s: float = 15.0,
        max_push_vel_xy: float = 1.0,
        push_duration_steps: int = 2,
        push_interval_range_low: float = 0.8,
        push_interval_range_high: float = 1.2,
        control_freq: int = 60,
        robot_mass: float = 2.5
    ):
        self.push_interval_s = push_interval_s
        self.max_push_vel_xy = max_push_vel_xy
        self.push_duration_steps = push_duration_steps
        self.push_interval_range = (push_interval_range_low, push_interval_range_high)
        self.control_freq = control_freq
        self.robot_mass = robot_mass

        # Calculate base interval in steps
        self._base_interval_steps = int(self.push_interval_s * control_freq)
        self._steps_since_push = 0
        self._next_push_at = self._sample_next_interval()
        self._push_steps_remaining = 0
        self._current_push_force = np.zeros(3)

    def _sample_next_interval(self) -> int:
        """Sample randomized interval for next push."""
        low, high = self.push_interval_range
        multiplier = np.random.uniform(low, high)
        return int(self._base_interval_steps * multiplier)

    def _sample_push_force(self) -> np.ndarray:
        """Sample random push force achieving target velocity change."""
        # Random direction in XY plane
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.uniform(0.5, 1.0) * self.max_push_vel_xy

        # Convert velocity to force: F = m * dv / dt
        dt = self.push_duration_steps / self.control_freq
        force_magnitude = self.robot_mass * magnitude / dt

        force = np.array([
            force_magnitude * np.cos(angle),
            force_magnitude * np.sin(angle),
            0.0  # No vertical push
        ])
        return force

    def maybe_apply_push(self, env: "SpotmicroEnv") -> bool:
        """
        Check if push should be applied and apply it.

        Parameters
        ----------
        env : SpotmicroEnv
            The training environment with _backend attribute

        Returns
        -------
        bool
            True if push was applied this step
        """
        self._steps_since_push += 1

        # Continue applying current push
        if self._push_steps_remaining > 0:
            env._backend.apply_external_force(self._current_push_force)
            self._push_steps_remaining -= 1
            return True

        # Check if time for new push
        if self._steps_since_push >= self._next_push_at:
            self._current_push_force = self._sample_push_force()
            self._push_steps_remaining = self.push_duration_steps
            self._steps_since_push = 0
            self._next_push_at = self._sample_next_interval()

            env._backend.apply_external_force(self._current_push_force)
            self._push_steps_remaining -= 1
            return True

        return False

    def reset(self):
        """Reset perturbation state for new episode."""
        self._steps_since_push = 0
        self._next_push_at = self._sample_next_interval()
        self._push_steps_remaining = 0
        self._current_push_force = np.zeros(3)


@dataclass
class FrictionConfig:
    """Configuration for friction randomization (legacy dataclass for backwards compat)."""
    friction_range: tuple = (0.5, 1.25)


@configurable
class FrictionRandomizer:
    """
    Randomizes ground friction at episode start.

    Works with any PhysicsBackend implementation (PyBullet, MuJoCo, etc).

    Parameters
    ----------
    config : Config
        Central config registry for saving/loading parameters
    friction_range_low : float
        Minimum friction coefficient (default: 0.5)
    friction_range_high : float
        Maximum friction coefficient (default: 1.25)
    """

    def __init__(
        self,
        config: Config,
        friction_range_low: float = 0.5,
        friction_range_high: float = 1.25
    ):
        self.friction_range = (friction_range_low, friction_range_high)

    def randomize(self, env: "SpotmicroEnv") -> float:
        """
        Apply random friction to terrain.

        Returns the sampled friction value.
        """
        friction = np.random.uniform(*self.friction_range)
        env._backend.set_friction(friction)
        return friction
