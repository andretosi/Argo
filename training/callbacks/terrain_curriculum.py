"""
Terrain Curriculum Callback
===========================

Atomic callback for terrain difficulty curriculum.
Gradually increases terrain height variation during training.

Usage:
    from training.callbacks import TerrainCurriculumCallbackV2
    from spotmicro.tools.config import Config

    callback = TerrainCurriculumCallbackV2(
        config=Config(),
        env=env,
        total_timesteps=1_000_000,
        z_max_initial=0.02,
        z_max_final=0.3,
        change_every_episodes=50,
    )
"""

import numpy as np
from spotmicro.tools.config import Config
from spotmicro.tools.configurable import configurable

from training.callbacks.base_curriculum import BaseCurriculumCallback


@configurable
class TerrainCurriculumCallbackV2(BaseCurriculumCallback):
    """
    Atomic callback for terrain difficulty curriculum.

    Spawns terrain with progressively increasing height variation.
    At start: nearly flat terrain (z_max_initial)
    At end: full height variation (z_max_final)

    Parameters
    ----------
    config : Config
        Central config registry
    env : SpotmicroEnv
        Training environment
    total_timesteps : int
        Estimated total training timesteps
    z_max_initial : float
        Starting terrain height variation in meters (default: 0.02)
    z_max_final : float
        Final terrain height variation (default: 0.3)
    change_every_episodes : int
        Change terrain every N episodes (default: 50)
    terrain_size : int
        Heightmap resolution (default: 256)
    scale : list
        Terrain scale [x, y, z] (default: [0.02, 0.02, 1.0])
    origin : list
        Terrain origin [x, y, z] (default: [0.0, 0.0, 0.0])
    """

    def __init__(
        self,
        config: Config,
        env=None,
        total_timesteps: int = 1_000_000,
        schedule: str = "linear",
        warmup_ratio: float = 0.05,
        z_max_initial: float = 0.02,
        z_max_final: float = 0.3,
        change_every_episodes: int = 50,
        terrain_size: int = 256,
        scale: list = None,
        origin: list = None,
        current_factor: float = 0.0,
        current_z_max=None,
        current_seed=None,
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

        self.z_max_initial = z_max_initial
        self.z_max_final = z_max_final
        self.change_every_episodes = change_every_episodes
        self.terrain_size = terrain_size
        self.scale = scale or [0.02, 0.02, 1.0]
        self.origin = origin or [0.0, 0.0, 0.0]
        self.current_factor = current_factor
        self.current_z_max = z_max_initial if current_z_max is None else current_z_max
        self.current_seed = current_seed

        self._terrain_handle = None
        self._last_change_episode = -1
        self._terrain_change_count = 0

    def _on_training_start(self) -> None:
        """Spawn initial terrain at start of training."""
        super()._on_training_start()

        if self.verbose:
            print(f"[TerrainCurriculum] z_max: {self.z_max_initial:.3f} -> {self.z_max_final:.3f}")

        unwrapped = self._get_unwrapped_env()
        if unwrapped is not None:
            if self.current_seed is not None:
                self._spawn_terrain(
                    unwrapped,
                    float(self.current_factor),
                    seed=int(self.current_seed),
                )
            else:
                self._spawn_terrain(unwrapped, 0.0)

    def _apply_curriculum(self, env, factor: float) -> None:
        """No per-step application needed for terrain."""
        pass

    def _on_episode_end(self, env) -> None:
        """Change terrain periodically at episode boundaries."""
        if self._episode_count - self._last_change_episode >= self.change_every_episodes:
            factor = self._get_curriculum_factor()
            self._spawn_terrain(env, factor)
            self._last_change_episode = self._episode_count

    def _spawn_terrain(self, env, factor: float, seed=None) -> None:
        """Spawn new terrain with current difficulty level."""
        from spotmicro.tools.TerrainTools import Heightmap

        if seed is None:
            seed = int(np.random.randint(0, 1_000_000))
        z_max = self._interpolate(self.z_max_initial, self.z_max_final, factor)
        if self.current_z_max is not None and seed == self.current_seed:
            z_max = self.current_z_max
        heightmap = Heightmap.from_noise(
            x=self.terrain_size,
            y=self.terrain_size,
            z_max=z_max,
            seed=seed,
        )

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
                scale=self.scale,
                origin=self.origin,
            )
            self._terrain_change_count += 1
            self._sync_config(
                current_factor=float(factor),
                current_z_max=float(z_max),
                current_seed=int(seed),
            )

            if self.verbose:
                print(
                    f"[TerrainCurriculum] Episode {self._episode_count}: "
                    f"z_max={z_max:.3f}, factor={factor:.2f}"
                )
        except Exception as e:
            if self.verbose:
                print(f"[TerrainCurriculum] Spawn error: {e}")

    def apply_saved_state(self, env=None) -> None:
        """Respawn the saved terrain snapshot in the provided environment."""
        target_env = env or self._get_unwrapped_env()
        if target_env is None or self.current_seed is None:
            return
        self._spawn_terrain(target_env, self.current_factor, seed=int(self.current_seed))

    def _on_training_end(self) -> None:
        """Log final stats."""
        if self.verbose:
            print(f"[TerrainCurriculum] Total terrain changes: {self._terrain_change_count}")
