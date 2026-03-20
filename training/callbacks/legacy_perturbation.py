"""Legacy perturbation callback kept for backward compatibility."""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from spotmicro.tools.config import Config
from spotmicro.tools.configurable import configurable

from training.utils.domain_randomization import (
    PushPerturbation,
    FrictionRandomizer,
)


@configurable
class PerturbationCallback(BaseCallback):
    """
    Callback that applies perturbations and domain randomization during training.

    Features:
    - Push perturbations at configurable intervals
    - Friction randomization on episode reset

    Parameters
    ----------
    config : Config
        Central config registry for saving/loading parameters
    env : SpotmicroEnv or VecEnv wrapper
        The training environment
    enable_push : bool
        Whether to enable push perturbations (default: True)
    enable_friction_rand : bool
        Whether to enable friction randomization (default: True)
    push_interval_s : float
        Time between pushes in seconds (default: 15.0)
    max_push_vel_xy : float
        Max velocity change in m/s (default: 1.0)
    push_duration_steps : int
        Steps to apply force (default: 2)
    friction_range_low : float
        Minimum friction coefficient (default: 0.5)
    friction_range_high : float
        Maximum friction coefficient (default: 1.25)
    verbose : bool
        Print perturbation events (default: True)
    """
    __config_exclude__ = {"env"}

    def __init__(
        self,
        config: Config,
        env=None,
        enable_push: bool = True,
        enable_friction_rand: bool = True,
        push_interval_s: float = 15.0,
        max_push_vel_xy: float = 1.0,
        push_duration_steps: int = 2,
        friction_range_low: float = 0.5,
        friction_range_high: float = 1.25,
        verbose: bool = True,
    ):
        super().__init__(verbose)

        self.env = env
        self.enable_push = enable_push
        self.enable_friction_rand = enable_friction_rand
        self.push_interval_s = push_interval_s
        self.max_push_vel_xy = max_push_vel_xy
        self.push_duration_steps = push_duration_steps
        self.friction_range_low = friction_range_low
        self.friction_range_high = friction_range_high

        # Initialize perturbation handlers (deferred until env is available)
        self.push_perturbation = None
        self.friction_randomizer = None
        self._initialized = False

        # Statistics
        self._push_count = 0
        self._episode_count = 0

    def _lazy_init(self):
        """Initialize perturbation handlers when env is available."""
        if self._initialized:
            return

        if self.enable_push and self.env is not None:
            unwrapped_env = self._get_unwrapped_env()
            robot_mass = 2.5
            control_freq = 60

            if unwrapped_env is not None:
                if hasattr(unwrapped_env, '_backend'):
                    try:
                        robot_mass = unwrapped_env._backend.get_base_mass()
                    except Exception:
                        pass
                if hasattr(unwrapped_env, 'control_frequnecy'):
                    control_freq = unwrapped_env.control_frequnecy

            self.push_perturbation = PushPerturbation(
                config=self.config,
                push_interval_s=self.push_interval_s,
                max_push_vel_xy=self.max_push_vel_xy,
                push_duration_steps=self.push_duration_steps,
                control_freq=control_freq,
                robot_mass=robot_mass
            )

        if self.enable_friction_rand and self.env is not None:
            self.friction_randomizer = FrictionRandomizer(
                config=self.config,
                friction_range_low=self.friction_range_low,
                friction_range_high=self.friction_range_high
            )

        self._initialized = True

    def _get_unwrapped_env(self):
        """Get the underlying SpotmicroEnv from possible wrappers."""
        env = self.env

        # Unwrap VecEnv
        if hasattr(env, "envs"):
            env = env.envs[0]
        if hasattr(env, "env"):
            env = env.env

        return env

    def _on_training_start(self) -> None:
        """Called at start of training."""
        self._lazy_init()

        if self.verbose:
            print(f"[PerturbationCallback] Training started")
            if self.enable_push and self.push_perturbation is not None:
                print(f"  - Push perturbations: enabled (interval ~{self.push_interval_s}s)")
            if self.enable_friction_rand and self.friction_randomizer is not None:
                print(f"  - Friction randomization: enabled (range [{self.friction_range_low}, {self.friction_range_high}])")

    def _on_step(self) -> bool:
        """Called after each environment step."""
        self._lazy_init()
        unwrapped_env = self._get_unwrapped_env()

        # Apply push perturbation
        if self.enable_push and self.push_perturbation is not None and unwrapped_env is not None:
            pushed = self.push_perturbation.maybe_apply_push(unwrapped_env)
            if pushed and self.push_perturbation._push_steps_remaining == 0:
                self._push_count += 1
                if self.verbose and self._push_count % 100 == 0:
                    print(f"[PerturbationCallback] Applied push #{self._push_count}")

        # Check for episode end
        dones = self.locals.get("dones", [False])
        if any(dones):
            self._episode_count += 1

            # Reset perturbation state
            if self.push_perturbation is not None:
                self.push_perturbation.reset()

            # Randomize friction for next episode
            if self.enable_friction_rand and self.friction_randomizer is not None and unwrapped_env is not None:
                friction = self.friction_randomizer.randomize(unwrapped_env)
                if self.verbose and self._episode_count % 100 == 0:
                    print(f"[PerturbationCallback] Episode {self._episode_count}: friction={friction:.2f}")

        return True

    def _on_training_end(self) -> None:
        """Called at end of training."""
        if self.verbose:
            print(f"[PerturbationCallback] Training ended")
            print(f"  - Total pushes: {self._push_count}")
            print(f"  - Total episodes: {self._episode_count}")
