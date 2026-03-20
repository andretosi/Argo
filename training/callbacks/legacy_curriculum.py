"""Legacy all-in-one curriculum callback kept for backward compatibility."""

import numpy as np
from typing import TYPE_CHECKING
from dataclasses import dataclass

from stable_baselines3.common.callbacks import BaseCallback

from spotmicro.tools.config import Config
from spotmicro.tools.configurable import configurable

if TYPE_CHECKING:
    from spotmicro.env.spotmicro_env import SpotmicroEnv


@dataclass
class CurriculumStage:
    """Represents the current curriculum difficulty levels."""
    terrain_difficulty: float  # 0.0 to 1.0
    push_magnitude: float      # Current max push velocity (m/s)
    friction_range: tuple      # (min, max) friction coefficients
    progress: float            # Overall training progress 0.0 to 1.0


@configurable
class CurriculumCallback(BaseCallback):
    """
    Unified curriculum callback for smooth domain randomization progression.

    Implements legged_gym-style curriculum where difficulty starts easy and
    gradually increases:
    - Terrain: starts nearly flat, increases to full height variation
    - Forces: starts with tiny pushes, increases to full perturbation
    - Friction: starts with narrow range, widens over time

    All transitions use smooth interpolation (linear or exponential).

    Parameters
    ----------
    config : Config
        Central config registry for saving/loading parameters

    env : SpotmicroEnv
        Training environment with _backend attribute

    total_timesteps : int
        Estimated total training timesteps for scheduling

    schedule : str
        Curriculum schedule type: "linear" or "exponential"

    warmup_ratio : float
        Fraction of training before curriculum starts ramping (default: 0.05)

    === Terrain Curriculum ===
    terrain_enabled : bool
        Whether to change terrain (default: True)
    terrain_change_every_episodes : int
        Change terrain every N episodes (default: 50)
    terrain_z_max_initial : float
        Starting terrain height variation in meters (default: 0.02, nearly flat)
    terrain_z_max_final : float
        Final terrain height variation (default: 0.3)
    terrain_size : int
        Heightmap resolution (default: 256)

    === Push Force Curriculum ===
    push_enabled : bool
        Whether to apply push perturbations (default: True)
    push_interval_s : float
        Time between pushes in seconds (default: 15.0)
    push_vel_initial : float
        Starting max push velocity in m/s (default: 0.1, very gentle)
    push_vel_final : float
        Final max push velocity (default: 1.5)
    push_duration_steps : int
        Steps to apply force (default: 2)

    === Friction Curriculum ===
    friction_enabled : bool
        Whether to randomize friction (default: True)
    friction_range_initial : tuple
        Starting friction range, narrow (default: (0.9, 1.1))
    friction_range_final : tuple
        Final friction range, wide (default: (0.4, 1.5))

    verbose : bool
        Print curriculum updates (default: True)
    """
    __config_exclude__ = {"env"}

    def __init__(
        self,
        config: Config,
        env=None,
        total_timesteps: int = 1_000_000,
        schedule: str = "linear",
        warmup_ratio: float = 0.05,
        # Terrain
        terrain_enabled: bool = True,
        terrain_change_every_episodes: int = 50,
        terrain_z_max_initial: float = 0.02,
        terrain_z_max_final: float = 0.3,
        terrain_size: int = 256,
        terrain_scale: list = None,
        terrain_origin: list = None,
        # Push forces
        push_enabled: bool = True,
        push_interval_s: float = 15.0,
        push_vel_initial: float = 0.1,
        push_vel_final: float = 1.5,
        push_duration_steps: int = 2,
        push_interval_range_low: float = 0.8,
        push_interval_range_high: float = 1.2,
        # Friction
        friction_enabled: bool = True,
        friction_range_initial_low: float = 0.9,
        friction_range_initial_high: float = 1.1,
        friction_range_final_low: float = 0.4,
        friction_range_final_high: float = 1.5,
        # Misc
        verbose: bool = True,
    ):
        super().__init__(verbose)

        self.env = env
        self.total_timesteps = total_timesteps
        self.schedule = schedule
        self.warmup_ratio = warmup_ratio

        # Terrain params
        self.terrain_enabled = terrain_enabled
        self.terrain_change_every_episodes = terrain_change_every_episodes
        self.terrain_z_max_initial = terrain_z_max_initial
        self.terrain_z_max_final = terrain_z_max_final
        self.terrain_size = terrain_size
        self.terrain_scale = terrain_scale or [0.02, 0.02, 1.0]
        self.terrain_origin = terrain_origin or [0.0, 0.0, 0.0]

        # Push force params
        self.push_enabled = push_enabled
        self.push_interval_s = push_interval_s
        self.push_vel_initial = push_vel_initial
        self.push_vel_final = push_vel_final
        self.push_duration_steps = push_duration_steps
        self.push_interval_range = (push_interval_range_low, push_interval_range_high)

        # Friction params
        self.friction_enabled = friction_enabled
        self.friction_range_initial = (friction_range_initial_low, friction_range_initial_high)
        self.friction_range_final = (friction_range_final_low, friction_range_final_high)

        # Internal state
        self._timesteps = 0
        self._episode_count = 0
        self._terrain_handle = None
        self._push_count = 0
        self._last_terrain_episode = -1

        # Push state (internal tracking)
        self._push_steps_remaining = 0
        self._push_force = np.zeros(3)
        self._steps_since_push = 0
        self._next_push_at = 0

        # Robot/env properties (lazy init)
        self._robot_mass = 2.5
        self._control_freq = 60
        self._initialized = False

    def _lazy_init(self):
        """Initialize from environment when available."""
        if self._initialized:
            return

        unwrapped = self._get_unwrapped_env()
        if unwrapped is not None:
            if hasattr(unwrapped, '_backend'):
                try:
                    self._robot_mass = unwrapped._backend.get_base_mass()
                except Exception:
                    pass
            if hasattr(unwrapped, 'control_frequnecy'):
                self._control_freq = unwrapped.control_frequnecy

        # Initialize push timing
        base_interval = int(self.push_interval_s * self._control_freq)
        self._next_push_at = self._sample_interval(base_interval)

        self._initialized = True

    def _get_unwrapped_env(self):
        """Get the underlying SpotmicroEnv from wrappers."""
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

    def _sample_interval(self, base_interval: int) -> int:
        """Sample randomized push interval."""
        low, high = self.push_interval_range
        return int(base_interval * np.random.uniform(low, high))

    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum state for logging/debugging."""
        factor = self._get_curriculum_factor()

        terrain_diff = factor
        push_mag = self._interpolate(self.push_vel_initial, self.push_vel_final, factor)

        friction_low = self._interpolate(
            self.friction_range_initial[0], self.friction_range_final[0], factor
        )
        friction_high = self._interpolate(
            self.friction_range_initial[1], self.friction_range_final[1], factor
        )

        return CurriculumStage(
            terrain_difficulty=terrain_diff,
            push_magnitude=push_mag,
            friction_range=(friction_low, friction_high),
            progress=self._get_progress()
        )

    # === Push Force Logic ===

    def _get_current_push_vel(self) -> float:
        """Get current max push velocity based on curriculum."""
        factor = self._get_curriculum_factor()
        return self._interpolate(self.push_vel_initial, self.push_vel_final, factor)

    def _sample_push_force(self) -> np.ndarray:
        """Sample random push force with current curriculum magnitude."""
        max_vel = self._get_current_push_vel()

        # Random direction in XY plane (full 360 degrees)
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.uniform(0.5, 1.0) * max_vel

        # Force = mass * velocity_change / dt
        dt = self.push_duration_steps / self._control_freq
        force_magnitude = self._robot_mass * magnitude / dt

        force = np.array([
            force_magnitude * np.cos(angle),
            force_magnitude * np.sin(angle),
            0.0  # No vertical push
        ])
        return force

    def _maybe_apply_push(self, env) -> bool:
        """Apply push if it's time."""
        self._steps_since_push += 1

        # Continue applying current push
        if self._push_steps_remaining > 0:
            env._backend.apply_external_force(self._push_force)
            self._push_steps_remaining -= 1
            return True

        # Check if time for new push
        if self._steps_since_push >= self._next_push_at:
            self._push_force = self._sample_push_force()
            self._push_steps_remaining = self.push_duration_steps
            self._steps_since_push = 0

            base_interval = int(self.push_interval_s * self._control_freq)
            self._next_push_at = self._sample_interval(base_interval)

            env._backend.apply_external_force(self._push_force)
            self._push_steps_remaining -= 1
            self._push_count += 1
            return True

        return False

    def _reset_push_state(self):
        """Reset push state for new episode.

        Note: We DON'T reset _steps_since_push here - push timing persists across
        episodes so that robots get pushed every ~N seconds of training time,
        not resetting each episode.
        """
        # Only reset active push state, not the global timing
        self._push_steps_remaining = 0
        self._push_force = np.zeros(3)

    # === Friction Logic ===

    def _get_current_friction_range(self) -> tuple:
        """Get current friction range based on curriculum."""
        factor = self._get_curriculum_factor()

        low = self._interpolate(
            self.friction_range_initial[0], self.friction_range_final[0], factor
        )
        high = self._interpolate(
            self.friction_range_initial[1], self.friction_range_final[1], factor
        )
        return (low, high)

    def _randomize_friction(self, env) -> float:
        """Apply random friction within current curriculum range."""
        low, high = self._get_current_friction_range()
        friction = np.random.uniform(low, high)
        env._backend.set_friction(friction)
        return friction

    # === Terrain Logic ===

    def _get_current_z_max(self) -> float:
        """Get current terrain z_max based on curriculum."""
        factor = self._get_curriculum_factor()
        return self._interpolate(self.terrain_z_max_initial, self.terrain_z_max_final, factor)

    def _spawn_terrain(self, env) -> None:
        """Spawn new terrain with current curriculum difficulty."""
        from spotmicro.tools.TerrainTools import Heightmap

        z_max = self._get_current_z_max()
        heightmap = Heightmap.from_noise(x=self.terrain_size, y=self.terrain_size, z_max=z_max)

        backend = env._backend

        # Remove old terrain
        if self._terrain_handle is not None:
            try:
                backend.remove_terrain(self._terrain_handle)
            except Exception:
                pass

        # Spawn new
        try:
            self._terrain_handle = backend.spawn_terrain(
                heightmap_data=heightmap.data,
                scale=self.terrain_scale,
                origin=self.terrain_origin
            )
        except Exception as e:
            if self.verbose:
                print(f"[Curriculum] Terrain spawn error: {e}")

    # === Callback Methods ===

    def _on_training_start(self) -> None:
        """Called at start of training."""
        self._lazy_init()

        if self.verbose:
            print(f"[CurriculumCallback] Training started")
            print(f"  Schedule: {self.schedule}, warmup: {self.warmup_ratio*100:.0f}%")
            if self.terrain_enabled:
                print(f"  Terrain: z_max {self.terrain_z_max_initial:.3f} -> {self.terrain_z_max_final:.3f}")
            if self.push_enabled:
                print(f"  Push: {self.push_vel_initial:.2f} -> {self.push_vel_final:.2f} m/s")
            if self.friction_enabled:
                print(f"  Friction: {self.friction_range_initial} -> {self.friction_range_final}")

        # Initial terrain
        if self.terrain_enabled:
            unwrapped = self._get_unwrapped_env()
            if unwrapped is not None:
                self._spawn_terrain(unwrapped)

    def _on_step(self) -> bool:
        """Called after each environment step."""
        self._lazy_init()
        self._timesteps += 1

        unwrapped = self._get_unwrapped_env()
        if unwrapped is None:
            return True

        # Apply push perturbation
        if self.push_enabled:
            self._maybe_apply_push(unwrapped)

        # Check episode end
        dones = self.locals.get("dones", [False])
        if any(dones):
            self._episode_count += 1

            # Reset push state
            if self.push_enabled:
                self._reset_push_state()

            # Randomize friction
            if self.friction_enabled:
                self._randomize_friction(unwrapped)

            # Change terrain periodically
            if self.terrain_enabled:
                if self._episode_count - self._last_terrain_episode >= self.terrain_change_every_episodes:
                    self._spawn_terrain(unwrapped)
                    self._last_terrain_episode = self._episode_count

                    if self.verbose:
                        stage = self.get_current_stage()
                        print(f"[Curriculum] Ep {self._episode_count}: "
                              f"progress={stage.progress:.1%}, "
                              f"terrain_diff={stage.terrain_difficulty:.2f}, "
                              f"push={stage.push_magnitude:.2f}m/s, "
                              f"friction={stage.friction_range}")

        return True

    def _on_training_end(self) -> None:
        """Called at end of training."""
        if self.verbose:
            stage = self.get_current_stage()
            print(f"[CurriculumCallback] Training ended")
            print(f"  Total timesteps: {self._timesteps:,}")
            print(f"  Total episodes: {self._episode_count}")
            print(f"  Total pushes: {self._push_count}")
            print(f"  Final stage: {stage}")
