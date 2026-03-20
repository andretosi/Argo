"""Legacy terrain callbacks kept for backward compatibility."""

from typing import Any, Callable, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TerrainChangeCallback(BaseCallback):
    """
    Callback that changes terrain every N episodes during training.

    This helps the policy generalize to different terrain types by exposing
    it to varied conditions throughout training.
    """

    def __init__(
        self,
        env,
        change_every_n_episodes: int = 50,
        terrain_generator: Optional[Callable] = None,
        generator_kwargs: Optional[Dict[str, Any]] = None,
        scale: list[float] = None,
        origin: list[float] = None,
        verbose: bool = True,
    ):
        """
        Initialize the terrain change callback.

        Args:
            env: The SpotmicroEnv environment (must have _backend attribute)
            change_every_n_episodes: Change terrain after this many episodes
            terrain_generator: Function to generate heightmap (default: Heightmap.from_noise)
            generator_kwargs: Kwargs for the generator (default: {"size": 256, "z_max": 0.2})
            scale: Terrain scale [x, y, z] for the physics backend
            origin: Terrain origin [x, y, z]
            verbose: Whether to print terrain change messages
        """
        super().__init__(verbose)

        self.env = env
        self.change_every_n_episodes = change_every_n_episodes
        self.scale = scale or [0.02, 0.02, 1.0]
        self.origin = origin or [0.0, 0.0, 0.0]

        # Default generator: noise-based heightmap
        if terrain_generator is None:
            from spotmicro.tools.TerrainTools import Heightmap
            self.terrain_generator = Heightmap.from_noise
        else:
            self.terrain_generator = terrain_generator

        self.generator_kwargs = generator_kwargs or {"size": 256, "z_max": 0.2}

        # Episode tracking
        self._episode_count = 0
        self._current_terrain_handle = None
        self._terrain_change_count = 0

    def _on_training_start(self) -> None:
        """Called at the start of training. Spawn initial terrain."""
        if self.verbose:
            print(f"[TerrainCallback] Starting training with terrain changes every {self.change_every_n_episodes} episodes")
        self._spawn_new_terrain()

    def _on_step(self) -> bool:
        """
        Called after each environment step.

        Checks if an episode ended (done=True) and if it's time to change terrain.
        """
        # Check if episode ended
        # In SB3, 'dones' is available in self.locals
        dones = self.locals.get("dones", [False])

        if any(dones):
            self._episode_count += 1

            # Check if it's time to change terrain
            if self._episode_count % self.change_every_n_episodes == 0:
                self._spawn_new_terrain()

        return True  # Continue training

    def _spawn_new_terrain(self) -> None:
        """Generate and spawn a new terrain."""
        # Generate new heightmap
        heightmap = self.terrain_generator(**self.generator_kwargs)

        # Get the backend from the environment
        backend = self._get_backend()
        if backend is None:
            if self.verbose:
                print("[TerrainCallback] Warning: Could not access backend, skipping terrain change")
            return

        # Remove old terrain if exists (for PyBullet)
        if self._current_terrain_handle is not None:
            try:
                backend.remove_terrain(self._current_terrain_handle)
            except Exception:
                pass  # Ignore errors on removal

        # Spawn new terrain
        try:
            self._current_terrain_handle = backend.spawn_terrain(
                heightmap_data=heightmap.data,
                scale=self.scale,
                origin=self.origin
            )
            self._terrain_change_count += 1

            if self.verbose:
                print(f"[TerrainCallback] Changed terrain #{self._terrain_change_count} "
                      f"(episode {self._episode_count})")

        except Exception as e:
            if self.verbose:
                print(f"[TerrainCallback] Error spawning terrain: {e}")

    def _get_backend(self):
        """Get the physics backend from the environment."""
        # Handle both direct env and VecEnv wrappers
        env = self.env

        # Unwrap VecEnv if necessary
        if hasattr(env, "envs"):
            env = env.envs[0]
        if hasattr(env, "env"):
            env = env.env

        # Get backend
        if hasattr(env, "_backend"):
            return env._backend
        return None

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose:
            print(f"[TerrainCallback] Training ended. Total terrain changes: {self._terrain_change_count}")


class CurriculumTerrainCallback(TerrainChangeCallback):
    """
    Advanced callback with curriculum learning: terrain difficulty increases over time.

    Usage:
    ------
        callback = CurriculumTerrainCallback(
            env=env,
            change_every_n_episodes=50,
            initial_difficulty=0.1,    # Start easy
            final_difficulty=1.0,      # End hard
            difficulty_schedule="linear"  # or "exponential"
        )
    """

    def __init__(
        self,
        env,
        change_every_n_episodes: int = 50,
        initial_difficulty: float = 0.1,
        final_difficulty: float = 1.0,
        difficulty_schedule: str = "linear",
        total_episodes_estimate: int = 10000,
        **kwargs
    ):
        """
        Initialize curriculum terrain callback.

        Args:
            initial_difficulty: Starting z_max multiplier (0.0 = flat, 1.0 = full height)
            final_difficulty: Ending z_max multiplier
            difficulty_schedule: "linear" or "exponential"
            total_episodes_estimate: Estimated total episodes for scheduling
        """
        super().__init__(env, change_every_n_episodes, **kwargs)

        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty
        self.difficulty_schedule = difficulty_schedule
        self.total_episodes_estimate = total_episodes_estimate

        # Store base z_max from generator kwargs
        self._base_z_max = self.generator_kwargs.get("z_max", 0.3)

    def _get_current_difficulty(self) -> float:
        """Calculate current difficulty based on episode count."""
        progress = min(1.0, self._episode_count / self.total_episodes_estimate)

        if self.difficulty_schedule == "linear":
            difficulty = self.initial_difficulty + progress * (self.final_difficulty - self.initial_difficulty)
        elif self.difficulty_schedule == "exponential":
            difficulty = self.initial_difficulty * (self.final_difficulty / self.initial_difficulty) ** progress
        else:
            difficulty = self.final_difficulty

        return difficulty

    def _spawn_new_terrain(self) -> None:
        """Generate terrain with current difficulty level."""
        difficulty = self._get_current_difficulty()

        # Adjust z_max based on difficulty
        current_kwargs = self.generator_kwargs.copy()
        current_kwargs["z_max"] = self._base_z_max * difficulty

        # Generate heightmap with adjusted difficulty
        heightmap = self.terrain_generator(**current_kwargs)

        backend = self._get_backend()
        if backend is None:
            return

        # Remove old terrain
        if self._current_terrain_handle is not None:
            try:
                backend.remove_terrain(self._current_terrain_handle)
            except Exception:
                pass

        # Spawn new terrain
        try:
            self._current_terrain_handle = backend.spawn_terrain(
                heightmap_data=heightmap.data,
                scale=self.scale,
                origin=self.origin
            )
            self._terrain_change_count += 1

            if self.verbose:
                print(f"[CurriculumTerrain] Changed terrain #{self._terrain_change_count} "
                      f"(episode {self._episode_count}, difficulty {difficulty:.2f}, "
                      f"z_max={current_kwargs['z_max']:.3f})")

        except Exception as e:
            if self.verbose:
                print(f"[CurriculumTerrain] Error: {e}")
