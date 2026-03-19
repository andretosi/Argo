"""
Friction Curriculum Callback
============================

Atomic callback for friction randomization curriculum.
Gradually widens friction range during training.

Usage:
    from training.callbacks import FrictionCurriculumCallback
    from spotmicro.tools.config import Config

    callback = FrictionCurriculumCallback(
        config=Config(),
        env=env,
        total_timesteps=1_000_000,
        friction_initial=(0.9, 1.1),
        friction_final=(0.4, 1.5),
    )
"""

import numpy as np
from spotmicro.tools.config import Config
from spotmicro.tools.configurable import configurable

from training.callbacks.base_curriculum import BaseCurriculumCallback


@configurable
class FrictionCurriculumCallback(BaseCurriculumCallback):
    """
    Atomic callback for friction randomization curriculum.

    Randomizes ground friction at episode boundaries with progressively
    widening range.
    At start: narrow range (friction_initial)
    At end: wide range (friction_final)

    Parameters
    ----------
    config : Config
        Central config registry
    env : SpotmicroEnv
        Training environment
    total_timesteps : int
        Estimated total training timesteps
    friction_initial_low : float
        Starting friction range low (default: 0.9)
    friction_initial_high : float
        Starting friction range high (default: 1.1)
    friction_final_low : float
        Final friction range low (default: 0.4)
    friction_final_high : float
        Final friction range high (default: 1.5)
    """

    def __init__(
        self,
        config: Config,
        env=None,
        total_timesteps: int = 1_000_000,
        schedule: str = "linear",
        warmup_ratio: float = 0.05,
        friction_initial_low: float = 0.9,
        friction_initial_high: float = 1.1,
        friction_final_low: float = 0.4,
        friction_final_high: float = 1.5,
        current_factor: float = 0.0,
        current_friction=None,
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

        self.friction_initial = (friction_initial_low, friction_initial_high)
        self.friction_final = (friction_final_low, friction_final_high)
        self.current_factor = current_factor
        self.current_friction = current_friction

        self._friction_count = 0

    def _on_training_start(self) -> None:
        """Log configuration at start."""
        super()._on_training_start()

        if self.verbose:
            print(
                f"[FrictionCurriculum] range: {self.friction_initial} -> "
                f"{self.friction_final}"
            )

        # Apply initial friction
        unwrapped = self._get_unwrapped_env()
        if unwrapped is not None:
            if self.current_friction is not None:
                self._randomize_friction(
                    unwrapped,
                    float(self.current_factor),
                    friction=float(self.current_friction),
                )
            else:
                self._randomize_friction(unwrapped, 0.0)

    def _apply_curriculum(self, env, factor: float) -> None:
        """No per-step application needed for friction."""
        pass

    def _on_episode_end(self, env) -> None:
        """Randomize friction at episode boundaries."""
        factor = self._get_curriculum_factor()
        self._randomize_friction(env, factor)

    def _randomize_friction(self, env, factor: float, friction=None) -> float:
        """Apply random friction within current curriculum range."""
        low = self._interpolate(self.friction_initial[0], self.friction_final[0], factor)
        high = self._interpolate(self.friction_initial[1], self.friction_final[1], factor)

        if friction is None:
            friction = np.random.uniform(low, high)

        try:
            env._backend.set_friction(friction)
            self._friction_count += 1
            self._sync_config(
                current_factor=float(factor),
                current_friction=float(friction),
            )
            self._record_metrics({
                "curriculum/friction_factor": factor,
                "curriculum/friction_value": friction,
                "curriculum/friction_low": low,
                "curriculum/friction_high": high,
            })
        except Exception as e:
            if self.verbose:
                print(f"[FrictionCurriculum] Error setting friction: {e}")

        return friction

    def apply_saved_state(self, env=None) -> None:
        """Apply the saved friction snapshot to the provided environment."""
        target_env = env or self._get_unwrapped_env()
        if target_env is None or self.current_friction is None:
            return
        self._randomize_friction(
            target_env,
            self.current_factor,
            friction=float(self.current_friction),
        )

    def _on_training_end(self) -> None:
        """Log final stats."""
        if self.verbose:
            print(f"[FrictionCurriculum] Total friction randomizations: {self._friction_count}")
