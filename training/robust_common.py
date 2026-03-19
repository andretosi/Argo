from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from spotmicro.devices.random_controller import RandomController
from spotmicro.env.spotmicro_env import SpotmicroEnv
from spotmicro.physics.factory import create_backend
from spotmicro.tools.TerrainTools import Heightmap
from spotmicro.tools.config import Config
from training.callbacks import (
    ForceCurriculumCallback,
    FrictionCurriculumCallback,
    MotorNoiseCurriculumCallback,
    SensorNoiseCurriculumCallback,
    TerrainCurriculumCallbackV2,
)
from training.reward_functions.robust_walking_reward import (
    RewardConfig,
    RewardState,
    reward_function,
)


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RunPaths:
    run_name: str
    run_dir: Path
    log_dir: Path
    step_checkpoint_dir: Path
    timed_checkpoint_dir: Path
    final_model_path: Path
    final_snapshot_path: Path
    initial_snapshot_path: Path


class StepSnapshotCallback(BaseCallback):
    """Save a model checkpoint and matching Config snapshot every N steps."""

    def __init__(
        self,
        save_dir: Path,
        run_name: str,
        config: Config,
        save_freq_steps: int,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.run_name = run_name
        self.config = config
        self.save_freq_steps = save_freq_steps
        self._last_saved_timestep = 0

    def _on_training_start(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.save_freq_steps <= 0:
            return True

        if self.num_timesteps - self._last_saved_timestep < self.save_freq_steps:
            return True

        self._last_saved_timestep = int(self.num_timesteps)
        save_stem = self.save_dir / f"ppo_{self.run_name}_steps_{self.num_timesteps:09d}"
        snapshot_path = save_stem.with_suffix(".yaml")

        self.model.save(save_stem)
        self.config.save(str(snapshot_path))

        if self.verbose:
            print(f"[StepCheckpoint] Saved model to: {save_stem}.zip")
            print(f"[StepCheckpoint] Saved snapshot to: {snapshot_path}")

        return True


class TimedSnapshotCallback(BaseCallback):
    """Save a model checkpoint and matching Config snapshot every N real seconds."""

    def __init__(
        self,
        save_dir: Path,
        run_name: str,
        config: Config,
        interval_seconds: int,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.run_name = run_name
        self.config = config
        self.interval_seconds = interval_seconds
        self._start_time = 0.0
        self._last_save_time = 0.0
        self._save_index = 0

    def _on_training_start(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        self._start_time = now
        self._last_save_time = now

    def _on_step(self) -> bool:
        if self.interval_seconds <= 0:
            return True

        now = time.time()
        if now - self._last_save_time < self.interval_seconds:
            return True

        self._save_index += 1
        elapsed_minutes = int((now - self._start_time) // 60)
        save_stem = self.save_dir / (
            f"ppo_{self.run_name}_timed_{self._save_index:03d}_{elapsed_minutes:04d}min"
        )
        snapshot_path = save_stem.with_suffix(".yaml")

        self.model.save(save_stem)
        self.config.save(str(snapshot_path))
        self._last_save_time = now

        if self.verbose:
            print(f"[TimedCheckpoint] Saved model to: {save_stem}.zip")
            print(f"[TimedCheckpoint] Saved snapshot to: {snapshot_path}")

        return True


def build_run_paths(run_name: str) -> RunPaths:
    run_dir = BASE_DIR / "data" / f"{run_name}_robust"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_dir = run_dir / "logs"
    step_checkpoint_dir = run_dir / "step_checkpoints"
    timed_checkpoint_dir = run_dir / "timed_checkpoints"

    log_dir.mkdir(parents=True, exist_ok=True)
    step_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timed_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_name=run_name,
        run_dir=run_dir,
        log_dir=log_dir,
        step_checkpoint_dir=step_checkpoint_dir,
        timed_checkpoint_dir=timed_checkpoint_dir,
        final_model_path=run_dir / f"ppo_{run_name}",
        final_snapshot_path=run_dir / f"ppo_{run_name}.yaml",
        initial_snapshot_path=run_dir / f"{run_name}_initial.yaml",
    )


def clipped_linear_schedule(initial_value: float, min_value: float = 1e-5):
    def schedule(progress_remaining: float) -> float:
        return max(progress_remaining * initial_value, min_value)

    return schedule


def default_reward_config() -> RewardConfig:
    return RewardConfig(
        tracking_lin_vel=1.0,
        tracking_ang_vel=0.5,
        tracking_sigma=0.25,
        feet_air_time=1.0,
        lin_vel_z=-2.0,
        ang_vel_xy=-0.05,
        orientation=-1.0,
        action_rate=-0.01,
        torques=-0.00001,
    )


def curriculum_factor_from_step(
    step: int,
    total_timesteps: int,
    warmup_ratio: float = 0.05,
    schedule: str = "linear",
) -> float:
    if total_timesteps <= 0:
        return 0.0

    progress = min(1.0, step / total_timesteps)
    if progress < warmup_ratio:
        return 0.0

    adjusted_progress = (progress - warmup_ratio) / (1.0 - warmup_ratio)
    if schedule == "exponential":
        return adjusted_progress ** 2
    return adjusted_progress


def interpolate(initial: float, final: float, factor: float) -> float:
    return initial + factor * (final - initial)


def create_robust_env(
    cfg: Config,
    *,
    backend_name: str = "pybullet",
    use_gui: bool = False,
    max_episode_len: int = 3000,
    sim_frequency: int = 240,
    control_frequency: int = 60,
    reward_config: RewardConfig | None = None,
) -> SpotmicroEnv:
    reward_cfg = reward_config or default_reward_config()
    backend = create_backend(
        backend_name,
        use_gui=use_gui,
        sim_frequency=sim_frequency,
    )
    device = RandomController(cfg)
    return SpotmicroEnv(
        backend=backend,
        device=device,
        config=cfg,
        reward_fn=reward_function,
        reward_state=RewardState(reward_cfg),
        use_gui=use_gui,
        max_episode_len=max_episode_len,
        sim_frequency=sim_frequency,
        control_frequency=control_frequency,
    )


def create_training_callbacks(
    *,
    cfg: Config,
    env: SpotmicroEnv,
    paths: RunPaths,
    total_timesteps: int,
    step_checkpoint_freq: int,
    timed_checkpoint_seconds: int,
    terrain_settings: dict,
    force_settings: dict,
    friction_settings: dict,
    motor_noise_settings: dict,
    sensor_noise_settings: dict,
    curriculum_verbose: bool = False,
    save_verbose: int = 1,
) -> tuple[CallbackList, dict]:
    terrain = TerrainCurriculumCallbackV2(
        config=cfg,
        env=env,
        total_timesteps=total_timesteps,
        verbose=curriculum_verbose,
        **terrain_settings,
    )
    force = ForceCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=total_timesteps,
        verbose=curriculum_verbose,
        **force_settings,
    )
    friction = FrictionCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=total_timesteps,
        verbose=curriculum_verbose,
        **friction_settings,
    )
    motor = MotorNoiseCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=total_timesteps,
        verbose=curriculum_verbose,
        **motor_noise_settings,
    )
    sensor = SensorNoiseCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=total_timesteps,
        verbose=curriculum_verbose,
        **sensor_noise_settings,
    )

    callback_list = [
        terrain,
        force,
        friction,
        motor,
        sensor,
    ]

    if step_checkpoint_freq > 0:
        callback_list.append(
            StepSnapshotCallback(
                save_dir=paths.step_checkpoint_dir,
                run_name=paths.run_name,
                config=cfg,
                save_freq_steps=step_checkpoint_freq,
                verbose=save_verbose,
            )
        )

    if timed_checkpoint_seconds > 0:
        callback_list.append(
            TimedSnapshotCallback(
                save_dir=paths.timed_checkpoint_dir,
                run_name=paths.run_name,
                config=cfg,
                interval_seconds=timed_checkpoint_seconds,
                verbose=save_verbose,
            )
        )

    return CallbackList(callback_list), {
        "terrain": terrain,
        "force": force,
        "friction": friction,
        "motor": motor,
        "sensor": sensor,
    }


def create_new_model(
    env: SpotmicroEnv,
    *,
    log_dir: Path,
    device: str = "cpu",
    verbose: int = 1,
    learning_rate: float = 3e-4,
    min_learning_rate: float = 1e-5,
    ent_coef: float = 0.01,
    clip_range: float = 0.2,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> PPO:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        learning_rate=clipped_linear_schedule(learning_rate, min_learning_rate),
        ent_coef=ent_coef,
        clip_range=clip_range,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        tensorboard_log=str(log_dir),
        device=device,
    )
    model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))
    return model


def load_model(
    model_stem: Path,
    env: SpotmicroEnv,
    *,
    log_dir: Path,
    device: str = "cpu",
) -> PPO:
    model = PPO.load(str(model_stem), env=env, device=device)
    model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))
    return model


def save_model_and_snapshot(model: PPO, cfg: Config, model_stem: Path) -> None:
    model.save(str(model_stem))
    cfg.save(str(model_stem.with_suffix(".yaml")))


def resolve_model_stem(paths: RunPaths, requested_model_stem: str | None) -> Path:
    if requested_model_stem:
        return Path(requested_model_stem)

    candidates = []
    candidates.extend(paths.timed_checkpoint_dir.glob("*.yaml"))
    candidates.extend(paths.step_checkpoint_dir.glob("*.yaml"))
    if paths.final_snapshot_path.exists():
        candidates.append(paths.final_snapshot_path)

    if not candidates:
        raise FileNotFoundError(
            f"No snapshot YAML found in {paths.run_dir}. "
            "Set the model path explicitly in the script."
        )

    latest_snapshot = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest_snapshot.with_suffix("")


def restore_snapshot_callbacks(cfg: Config, env: SpotmicroEnv, *, verbose: bool = False) -> dict:
    terrain = TerrainCurriculumCallbackV2(config=cfg, env=env, verbose=verbose)
    force = ForceCurriculumCallback(config=cfg, env=env, verbose=verbose)
    friction = FrictionCurriculumCallback(config=cfg, env=env, verbose=verbose)
    motor = MotorNoiseCurriculumCallback(config=cfg, env=env, verbose=verbose)
    sensor = SensorNoiseCurriculumCallback(config=cfg, env=env, verbose=verbose)

    terrain.apply_saved_state(env)
    friction.apply_saved_state(env)
    motor.apply_saved_state()
    sensor.apply_saved_state()

    return {
        "terrain": terrain,
        "force": force,
        "friction": friction,
        "motor": motor,
        "sensor": sensor,
    }


def seed_curriculum_timesteps(callbacks: dict, current_timestep: int) -> None:
    for callback in callbacks.values():
        if hasattr(callback, "_timesteps"):
            callback._timesteps = int(current_timestep)


def spawn_noise_terrain(
    env: SpotmicroEnv,
    *,
    z_max: float,
    terrain_size: int,
    scale: tuple[float, float, float],
    origin: tuple[float, float, float],
    seed: int | None = None,
) -> tuple[int, float]:
    terrain_seed = int(np.random.randint(0, 1_000_000)) if seed is None else int(seed)
    heightmap = Heightmap.from_noise(
        x=terrain_size,
        y=terrain_size,
        z_max=z_max,
        seed=terrain_seed,
    )
    env._backend.spawn_terrain(
        heightmap_data=heightmap.data,
        scale=list(scale),
        origin=list(origin),
    )
    return terrain_seed, float(z_max)


def attach_noise_callbacks_for_testing(
    cfg: Config,
    env: SpotmicroEnv,
    *,
    motor_noise_std: float,
    sensor_noise_scale: float,
    verbose: bool = False,
) -> dict:
    callbacks = {}

    if motor_noise_std > 0.0:
        motor = MotorNoiseCurriculumCallback(
            config=cfg,
            env=env,
            current_factor=1.0,
            current_noise_std=motor_noise_std,
            verbose=verbose,
        )
        motor.apply_saved_state()
        callbacks["motor"] = motor

    if sensor_noise_scale > 0.0:
        sensor = SensorNoiseCurriculumCallback(
            config=cfg,
            env=env,
            current_factor=1.0,
            current_noise_scale=sensor_noise_scale,
            verbose=verbose,
        )
        sensor.apply_saved_state()
        callbacks["sensor"] = sensor

    return callbacks


def create_force_callback_for_testing(
    cfg: Config,
    env: SpotmicroEnv,
    *,
    push_velocity: float,
    push_interval_s: float,
    push_duration_steps: int,
    verbose: bool = False,
) -> ForceCurriculumCallback | None:
    if push_velocity <= 0.0:
        return None

    return ForceCurriculumCallback(
        config=cfg,
        env=env,
        push_vel_initial=push_velocity,
        push_vel_final=push_velocity,
        push_interval_s=push_interval_s,
        push_duration_steps=push_duration_steps,
        current_factor=1.0,
        current_push_vel=push_velocity,
        verbose=verbose,
    )


def print_snapshot_summary(snapshot_callbacks: dict) -> None:
    terrain = snapshot_callbacks["terrain"]
    friction = snapshot_callbacks["friction"]
    motor = snapshot_callbacks["motor"]
    sensor = snapshot_callbacks["sensor"]
    force = snapshot_callbacks["force"]

    print("Loaded snapshot:")
    print(f"  Terrain factor: {terrain.current_factor:.3f}")
    print(f"  Terrain z_max:  {terrain.current_z_max}")
    print(f"  Terrain seed:   {terrain.current_seed}")
    print(f"  Friction:       {friction.current_friction}")
    print(f"  Motor noise:    {motor.current_noise_std}")
    print(f"  Sensor noise:   {sensor.current_noise_scale}")
    print(f"  Push velocity:  {force.current_push_vel}")


def run_policy_rollout(
    *,
    model: PPO,
    env: SpotmicroEnv,
    n_eval_episodes: int,
    deterministic: bool = True,
    force_callback: ForceCurriculumCallback | None = None,
    realtime_sleep: float = 0.0,
) -> list[float]:
    episode_rewards = []

    for episode_idx in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            if force_callback is not None:
                force_callback.step_saved_state(env)

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if realtime_sleep > 0.0:
                time.sleep(realtime_sleep)

        episode_rewards.append(float(total_reward))
        print(f"Episode {episode_idx + 1}: reward = {total_reward:.2f}")

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return episode_rewards
