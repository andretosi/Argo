"""Push-force curriculum callback."""

import numpy as np
from spotmicro.tools.config import Config
from spotmicro.tools.configurable import configurable

from training.callbacks.curriculum_base import BaseCurriculumCallback


@configurable
class ForceCurriculumCallback(BaseCurriculumCallback):
    """Apply progressively stronger random pushes during training."""

    def __init__(
        self,
        config: Config,
        env=None,
        total_timesteps: int = 1_000_000,
        schedule: str = "linear",
        warmup_ratio: float = 0.05,
        progression_mode: str = "time",
        push_vel_initial: float = 0.1,
        push_vel_final: float = 1.5,
        push_interval_s: float = 15.0,
        push_duration_steps: int = 2,
        push_interval_range_low: float = 0.8,
        push_interval_range_high: float = 1.2,
        current_factor: float = 0.0,
        current_push_vel=None,
        verbose: bool = True,
    ):
        super().__init__(
            config=config,
            env=env,
            total_timesteps=total_timesteps,
            schedule=schedule,
            warmup_ratio=warmup_ratio,
            progression_mode=progression_mode,
            verbose=verbose,
        )

        self.push_vel_initial = push_vel_initial
        self.push_vel_final = push_vel_final
        self.push_interval_s = push_interval_s
        self.push_duration_steps = push_duration_steps
        self.push_interval_range = (push_interval_range_low, push_interval_range_high)
        self.current_factor = current_factor
        self.current_push_vel = (
            push_vel_initial if current_push_vel is None else current_push_vel
        )

        # Push state
        self._push_steps_remaining = 0
        self._current_push_force = np.zeros(3)
        self._steps_since_push = 0
        self._next_push_at = 0
        self._push_count = 0

        # Robot properties (lazy init)
        self._robot_mass = 2.5
        self._control_freq = 60

    def _lazy_init(self):
        """Initialize from environment when available."""
        if self._initialized:
            return

        unwrapped = self._get_unwrapped_env()
        if unwrapped is not None:
            if hasattr(unwrapped, "_backend"):
                try:
                    self._robot_mass = unwrapped._backend.get_base_mass()
                except Exception:
                    pass
            if hasattr(unwrapped, "control_frequnecy"):
                self._control_freq = unwrapped.control_frequnecy

        # Initialize push timing
        base_interval = int(self.push_interval_s * self._control_freq)
        self._next_push_at = self._sample_interval(base_interval)

        self._initialized = True

    def _sample_interval(self, base_interval: int) -> int:
        """Sample randomized push interval."""
        low, high = self.push_interval_range
        return int(base_interval * np.random.uniform(low, high))

    def _sample_push_force(self, max_vel: float) -> np.ndarray:
        """Sample random push force achieving target velocity change."""
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.uniform(0.5, 1.0) * max_vel

        # Force = mass * velocity_change / dt
        dt = self.push_duration_steps / self._control_freq
        force_magnitude = self._robot_mass * magnitude / dt

        force = np.array([
            force_magnitude * np.cos(angle),
            force_magnitude * np.sin(angle),
            0.0,  # No vertical push
        ])
        return force

    def _on_training_start(self) -> None:
        """Log configuration at start."""
        super()._on_training_start()

        if self.verbose:
            print(
                f"[ForceCurriculum] push_vel: {self.push_vel_initial:.2f} -> "
                f"{self.push_vel_final:.2f} m/s, interval ~{self.push_interval_s}s"
            )

    def _apply_curriculum(self, env, factor: float) -> None:
        """Apply push force at current curriculum level."""
        current_push_vel = self._interpolate(
            self.push_vel_initial, self.push_vel_final, factor
        )
        self._sync_config(
            current_factor=float(factor),
            current_push_vel=float(current_push_vel),
        )
        self._record_metrics({
            "curriculum/push_factor": factor,
            "curriculum/push_velocity": current_push_vel,
            "curriculum/push_count": self._push_count,
        })
        self._steps_since_push += 1

        # Continue applying current push
        if self._push_steps_remaining > 0:
            env._backend.apply_external_force(self._current_push_force)
            self._push_steps_remaining -= 1
            return

        # Check if time for new push
        if self._steps_since_push >= self._next_push_at:
            self._current_push_force = self._sample_push_force(current_push_vel)
            self._push_steps_remaining = self.push_duration_steps
            self._steps_since_push = 0

            base_interval = int(self.push_interval_s * self._control_freq)
            self._next_push_at = self._sample_interval(base_interval)

            env._backend.apply_external_force(self._current_push_force)
            self._push_steps_remaining -= 1
            self._push_count += 1
            self._record_metrics({
                "curriculum/push_count": self._push_count,
            })

    def step_saved_state(self, env=None) -> None:
        """Advance push perturbations using the saved curriculum intensity."""
        target_env = env or self._get_unwrapped_env()
        if target_env is None:
            return
        self._lazy_init()
        self._apply_curriculum(target_env, float(self.current_factor))

    def _on_episode_end(self, env) -> None:
        """Reset push state on episode end (but keep global timing)."""
        self._push_steps_remaining = 0
        self._current_push_force = np.zeros(3)

    def _on_training_end(self) -> None:
        """Log final stats."""
        if self.verbose:
            print(f"[ForceCurriculum] Total pushes: {self._push_count}")
