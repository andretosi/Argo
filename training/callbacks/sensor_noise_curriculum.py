"""
Sensor Noise Curriculum Callback
================================

Atomic callback for sensor noise curriculum.
Gradually increases observation noise during training for sim-to-real transfer.

Usage:
    from training.callbacks import SensorNoiseCurriculumCallback
    from spotmicro.tools.config import Config

    callback = SensorNoiseCurriculumCallback(
        config=Config(),
        env=env,
        total_timesteps=1_000_000,
        noise_scale_initial=0.0,
        noise_scale_final=1.0,
    )
"""

import numpy as np
from spotmicro.tools.config import Config
from spotmicro.tools.configurable import configurable

from training.callbacks.base_curriculum import BaseCurriculumCallback
from training.utils.noise import SensorNoise


@configurable
class SensorNoiseCurriculumCallback(BaseCurriculumCallback):
    """
    Atomic callback for sensor noise curriculum.

    Wraps observations with progressively increasing noise levels.
    Uses SensorNoise utility internally with scaling factor.

    Parameters
    ----------
    config : Config
        Central config registry
    env : SpotmicroEnv
        Training environment
    total_timesteps : int
        Estimated total training timesteps
    noise_scale_initial : float
        Starting noise scale multiplier (default: 0.0)
    noise_scale_final : float
        Final noise scale multiplier (default: 1.0)
    dof_pos_noise : float
        Base joint position noise std [rad] (default: 0.01)
    dof_vel_noise : float
        Base joint velocity noise std [rad/s] (default: 1.5)
    lin_vel_noise : float
        Base linear velocity noise std [m/s] (default: 0.1)
    ang_vel_noise : float
        Base angular velocity noise std [rad/s] (default: 0.2)
    gravity_noise : float
        Base gravity vector noise std (default: 0.05)
    height_noise : float
        Base height noise std [m] (default: 0.02)
    """

    def __init__(
        self,
        config: Config,
        env=None,
        total_timesteps: int = 1_000_000,
        schedule: str = "linear",
        warmup_ratio: float = 0.05,
        noise_scale_initial: float = 0.0,
        noise_scale_final: float = 1.0,
        dof_pos_noise: float = 0.01,
        dof_vel_noise: float = 1.5,
        lin_vel_noise: float = 0.1,
        ang_vel_noise: float = 0.2,
        gravity_noise: float = 0.05,
        height_noise: float = 0.02,
        current_factor: float = 0.0,
        current_noise_scale=None,
        verbose: bool = True,
    ):
        super().__init__(
            config=config,
            env=env,
            total_timesteps=total_timesteps,
            schedule=schedule,
            warmup_ratio=warmup_ratio,
            verbose=verbose,
        )

        self.noise_scale_initial = noise_scale_initial
        self.noise_scale_final = noise_scale_final

        # Base noise levels (will be scaled by curriculum factor)
        self.base_dof_pos_noise = dof_pos_noise
        self.base_dof_vel_noise = dof_vel_noise
        self.base_lin_vel_noise = lin_vel_noise
        self.base_ang_vel_noise = ang_vel_noise
        self.base_gravity_noise = gravity_noise
        self.base_height_noise = height_noise
        self.current_factor = current_factor

        self._current_scale = noise_scale_initial if current_noise_scale is None else current_noise_scale
        self.current_noise_scale = self._current_scale
        self._wrapped = False

    def _on_training_start(self) -> None:
        """Log configuration and wrap environment."""
        super()._on_training_start()
        if self.current_noise_scale is not None:
            self._current_scale = float(self.current_noise_scale)

        if self.verbose:
            print(
                f"[SensorNoiseCurriculum] scale: {self.noise_scale_initial:.2f} -> "
                f"{self.noise_scale_final:.2f}"
            )

        self._wrap_env()

    def _wrap_env(self):
        """Wrap the environment's _get_observation to inject sensor noise."""
        if self._wrapped:
            return

        unwrapped = self._get_unwrapped_env()
        if unwrapped is None:
            return

        if not hasattr(unwrapped, "_get_observation"):
            if self.verbose:
                print("[SensorNoiseCurriculum] Warning: env has no _get_observation method")
            return

        original_get_observation = unwrapped._get_observation

        def noisy_get_observation():
            obs = original_get_observation()
            return self._apply_noise(obs)

        unwrapped._get_observation = noisy_get_observation
        self._wrapped = True

    def _apply_noise(self, obs: np.ndarray) -> np.ndarray:
        """Apply scaled noise to observation vector."""
        if self._current_scale <= 0:
            return obs

        noisy_obs = obs.copy()
        scale = self._current_scale

        # Gravity vector (indices 0-2)
        noisy_obs[0:3] += np.random.normal(0, self.base_gravity_noise * scale, 3)
        norm = np.linalg.norm(noisy_obs[0:3])
        if norm > 1e-8:
            noisy_obs[0:3] /= norm

        # Height (index 3)
        noisy_obs[3] += np.random.normal(0, self.base_height_noise * scale)

        # Linear velocity (indices 4-6)
        noisy_obs[4:7] += np.random.normal(0, self.base_lin_vel_noise * scale, 3)

        # Angular velocity (indices 7-9)
        noisy_obs[7:10] += np.random.normal(0, self.base_ang_vel_noise * scale, 3)

        # Joint positions - current (indices 10-21)
        noisy_obs[10:22] += np.random.normal(0, self.base_dof_pos_noise * scale, 12)

        # Joint velocities - current (indices 22-33)
        noisy_obs[22:34] += np.random.normal(0, self.base_dof_vel_noise * scale, 12)

        # History positions (indices 34-45 and 58-69)
        noisy_obs[34:46] += np.random.normal(0, self.base_dof_pos_noise * scale, 12)
        if len(noisy_obs) > 69:
            noisy_obs[58:70] += np.random.normal(0, self.base_dof_pos_noise * scale, 12)

        # History velocities (indices 46-57 and 70-81)
        noisy_obs[46:58] += np.random.normal(0, self.base_dof_vel_noise * scale, 12)
        if len(noisy_obs) > 81:
            noisy_obs[70:82] += np.random.normal(0, self.base_dof_vel_noise * scale, 12)

        return noisy_obs

    def _apply_curriculum(self, env, factor: float) -> None:
        """Update current noise scale based on curriculum."""
        self._current_scale = self._interpolate(
            self.noise_scale_initial, self.noise_scale_final, factor
        )
        self._sync_config(
            current_factor=float(factor),
            current_noise_scale=float(self._current_scale),
        )
        self._record_metrics({
            "curriculum/sensor_noise_factor": factor,
            "curriculum/sensor_noise_scale": self._current_scale,
        })

    def apply_saved_state(self) -> None:
        """Wrap the environment using the saved sensor-noise snapshot."""
        self._current_scale = float(self.current_noise_scale)
        self._wrap_env()
        self._sync_config(
            current_factor=float(self.current_factor),
            current_noise_scale=float(self._current_scale),
        )

    def _on_training_end(self) -> None:
        """Log final stats."""
        if self.verbose:
            print(f"[SensorNoiseCurriculum] Final scale: {self._current_scale:.2f}")
