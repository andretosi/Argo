"""Motor command noise curriculum callback."""

import numpy as np
from spotmicro.tools.config import Config
from spotmicro.tools.configurable import configurable

from training.callbacks.curriculum_base import BaseCurriculumCallback


@configurable
class MotorNoiseCurriculumCallback(BaseCurriculumCallback):
    """Increase action noise gradually to harden the controller."""

    def __init__(
        self,
        config: Config,
        env=None,
        total_timesteps: int = 1_000_000,
        schedule: str = "linear",
        warmup_ratio: float = 0.05,
        progression_mode: str = "time",
        noise_initial: float = 0.0,
        noise_final: float = 0.05,
        noise_type: str = "gaussian",
        current_factor: float = 0.0,
        current_noise_std=None,
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

        self.noise_initial = noise_initial
        self.noise_final = noise_final
        self.noise_type = noise_type
        self.current_factor = current_factor

        self._current_noise_std = noise_initial if current_noise_std is None else current_noise_std
        self.current_noise_std = self._current_noise_std
        self._wrapped = False

    def _on_training_start(self) -> None:
        """Log configuration and wrap environment."""
        super()._on_training_start()
        if self.current_noise_std is not None:
            self._current_noise_std = float(self.current_noise_std)

        if self.verbose:
            print(
                f"[MotorNoiseCurriculum] noise_std: {self.noise_initial:.4f} -> "
                f"{self.noise_final:.4f} rad ({self.noise_type})"
            )

        self._wrap_env()

    def _wrap_env(self):
        """Wrap the environment step function to inject motor noise."""
        if self._wrapped:
            return

        unwrapped = self._get_unwrapped_env()
        if unwrapped is None:
            return

        original_step = unwrapped.step

        def noisy_step(action):
            # Add noise to action
            noise = self._sample_noise(action.shape)
            noisy_action = action + noise

            # Clip to action space
            if hasattr(unwrapped, "action_space"):
                low = unwrapped.action_space.low
                high = unwrapped.action_space.high
                noisy_action = np.clip(noisy_action, low, high)

            return original_step(noisy_action)

        unwrapped.step = noisy_step
        self._wrapped = True

    def _sample_noise(self, shape) -> np.ndarray:
        """Sample noise based on current curriculum level."""
        if self._current_noise_std <= 0:
            return np.zeros(shape)

        if self.noise_type == "gaussian":
            return np.random.normal(0, self._current_noise_std, shape)
        elif self.noise_type == "uniform":
            return np.random.uniform(
                -self._current_noise_std, self._current_noise_std, shape
            )
        else:
            return np.zeros(shape)

    def _apply_curriculum(self, env, factor: float) -> None:
        """Update current noise level based on curriculum."""
        self._current_noise_std = self._interpolate(
            self.noise_initial, self.noise_final, factor
        )
        self._sync_config(
            current_factor=float(factor),
            current_noise_std=float(self._current_noise_std),
        )
        self._record_metrics({
            "curriculum/motor_noise_factor": factor,
            "curriculum/motor_noise_std": self._current_noise_std,
        })

    def apply_saved_state(self) -> None:
        """Wrap the environment using the saved motor-noise snapshot."""
        self._current_noise_std = float(self.current_noise_std)
        self._wrap_env()
        self._sync_config(
            current_factor=float(self.current_factor),
            current_noise_std=float(self._current_noise_std),
        )

    def _on_training_end(self) -> None:
        """Log final stats."""
        if self.verbose:
            print(f"[MotorNoiseCurriculum] Final noise_std: {self._current_noise_std:.4f}")
