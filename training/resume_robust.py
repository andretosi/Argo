import sys
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
TRAINING_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from spotmicro.devices.random_controller import RandomController
from spotmicro.env.spotmicro_env import SpotmicroEnv
from spotmicro.physics.factory import create_backend
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


# ============================================================================
# Resume Configuration
# Edit these values directly in the file, then run the script from your editor.
# ============================================================================

SOURCE_RUN_NAME = "robust_walk_v1"        # Run folder to resume from.
SOURCE_MODEL_STEM = None                  # Explicit model path without .zip, or None for latest snapshot.

RESUME_RUN_NAME = "robust_walk_v1_resume" # Folder/file prefix for the resumed run.
EXTRA_TRAINING_STEPS = 10_000             # Additional PPO timesteps to train.
NEW_TOTAL_TIMESTEPS = None                # Override the curriculum total if you do not want old+extra.

BACKEND = "pybullet"                      # Physics engine: "pybullet" or "mujoco".
USE_GUI = False                           # True to watch the resumed training live.
DEVICE = "cpu"                            # PPO device.

RUN_ENV_CHECK = False                     # True to run Stable-Baselines3 env checks before resume.
MAX_EPISODE_LEN = 3000                    # Max control steps per episode before truncation.
SIM_FREQUENCY = 240                       # Physics frequency in Hz.
CONTROL_FREQUENCY = 60                    # Policy/action frequency in Hz.

STEP_CHECKPOINT_FREQUENCY = 2_500         # Save a model+snapshot every N steps during resume.
TIMED_SNAPSHOT_SECONDS = 300              # Save a model+snapshot every N real seconds during resume.

MODEL_VERBOSE = 1                         # PPO console logging level after loading the model.
CURRICULUM_VERBOSE = False                # Print terrain/friction/noise curriculum messages.
SAVE_VERBOSE = 1                          # Print checkpoint save messages.


class StepSnapshotCallback(BaseCallback):
    def __init__(self, save_dir, save_freq_steps, config, run_name, verbose=1):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_freq_steps = save_freq_steps
        self.config = config
        self.run_name = run_name
        self.last_saved_timestep = 0

    def _on_training_start(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        if self.save_freq_steps <= 0:
            return True

        if self.num_timesteps - self.last_saved_timestep < self.save_freq_steps:
            return True

        self.last_saved_timestep = int(self.num_timesteps)
        save_path = self.save_dir / f"ppo_{self.run_name}_steps_{self.num_timesteps:09d}"
        snapshot_path = save_path.with_suffix(".yaml")

        self.model.save(str(save_path))
        self.config.save(str(snapshot_path))

        if self.verbose:
            print(f"[StepCheckpoint] Saved model to: {save_path}.zip")
            print(f"[StepCheckpoint] Saved snapshot to: {snapshot_path}")

        return True


class TimedSnapshotCallback(BaseCallback):
    def __init__(self, save_dir, interval_seconds, config, run_name, verbose=1):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.interval_seconds = interval_seconds
        self.config = config
        self.run_name = run_name
        self.start_time = 0.0
        self.last_save_time = 0.0
        self.save_index = 0

    def _on_training_start(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        self.start_time = now
        self.last_save_time = now

    def _on_step(self):
        if self.interval_seconds <= 0:
            return True

        now = time.time()
        if now - self.last_save_time < self.interval_seconds:
            return True

        self.save_index += 1
        elapsed_minutes = int((now - self.start_time) // 60)
        save_path = self.save_dir / (
            f"ppo_{self.run_name}_timed_{self.save_index:03d}_{elapsed_minutes:04d}min"
        )
        snapshot_path = save_path.with_suffix(".yaml")

        self.model.save(str(save_path))
        self.config.save(str(snapshot_path))
        self.last_save_time = now

        if self.verbose:
            print(f"[TimedCheckpoint] Saved model to: {save_path}.zip")
            print(f"[TimedCheckpoint] Saved snapshot to: {snapshot_path}")

        return True


def create_run_paths(run_name):
    run_dir = TRAINING_DIR / "data" / f"{run_name}_robust"
    log_dir = run_dir / "logs"
    step_checkpoint_dir = run_dir / "step_checkpoints"
    timed_checkpoint_dir = run_dir / "timed_checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    step_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timed_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "log_dir": log_dir,
        "step_checkpoint_dir": step_checkpoint_dir,
        "timed_checkpoint_dir": timed_checkpoint_dir,
        "initial_snapshot_path": run_dir / f"{run_name}_initial.yaml",
        "final_model_path": run_dir / f"ppo_{run_name}",
        "final_snapshot_path": run_dir / f"ppo_{run_name}.yaml",
    }


def create_reward_config():
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


def create_env(cfg):
    backend = create_backend(
        BACKEND,
        use_gui=USE_GUI,
        sim_frequency=SIM_FREQUENCY,
    )
    device = RandomController(cfg)
    reward_config = create_reward_config()

    return SpotmicroEnv(
        backend=backend,
        device=device,
        config=cfg,
        reward_fn=reward_function,
        reward_state=RewardState(reward_config),
        use_gui=USE_GUI,
        max_episode_len=MAX_EPISODE_LEN,
        sim_frequency=SIM_FREQUENCY,
        control_frequency=CONTROL_FREQUENCY,
    )


def resolve_model_stem(paths):
    if SOURCE_MODEL_STEM is not None:
        return Path(SOURCE_MODEL_STEM)

    candidates = list(paths["timed_checkpoint_dir"].glob("*.yaml"))
    candidates.extend(paths["step_checkpoint_dir"].glob("*.yaml"))

    if paths["final_snapshot_path"].exists():
        candidates.append(paths["final_snapshot_path"])

    if not candidates:
        raise FileNotFoundError(
            f"No snapshot YAML found in {paths['run_dir']}. "
            "Set SOURCE_MODEL_STEM explicitly."
        )

    latest_snapshot = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest_snapshot.with_suffix("")


def create_callbacks(cfg, env, resume_total_timesteps, paths):
    terrain_callback = TerrainCurriculumCallbackV2(
        config=cfg,
        env=env,
        total_timesteps=resume_total_timesteps,
        verbose=CURRICULUM_VERBOSE,
    )
    force_callback = ForceCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=resume_total_timesteps,
        verbose=CURRICULUM_VERBOSE,
    )
    friction_callback = FrictionCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=resume_total_timesteps,
        verbose=CURRICULUM_VERBOSE,
    )
    motor_noise_callback = MotorNoiseCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=resume_total_timesteps,
        verbose=CURRICULUM_VERBOSE,
    )
    sensor_noise_callback = SensorNoiseCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=resume_total_timesteps,
        verbose=CURRICULUM_VERBOSE,
    )

    callbacks = [
        terrain_callback,
        force_callback,
        friction_callback,
        motor_noise_callback,
        sensor_noise_callback,
    ]

    if STEP_CHECKPOINT_FREQUENCY > 0:
        callbacks.append(
            StepSnapshotCallback(
                save_dir=paths["step_checkpoint_dir"],
                save_freq_steps=STEP_CHECKPOINT_FREQUENCY,
                config=cfg,
                run_name=RESUME_RUN_NAME,
                verbose=SAVE_VERBOSE,
            )
        )

    if TIMED_SNAPSHOT_SECONDS > 0:
        callbacks.append(
            TimedSnapshotCallback(
                save_dir=paths["timed_checkpoint_dir"],
                interval_seconds=TIMED_SNAPSHOT_SECONDS,
                config=cfg,
                run_name=RESUME_RUN_NAME,
                verbose=SAVE_VERBOSE,
            )
        )

    callback_parts = {
        "terrain": terrain_callback,
        "force": force_callback,
        "friction": friction_callback,
        "motor": motor_noise_callback,
        "sensor": sensor_noise_callback,
    }

    return CallbackList(callbacks), callback_parts


def seed_callback_timesteps(callback_parts, current_timesteps):
    for callback in callback_parts.values():
        callback._timesteps = int(current_timesteps)


def save_model_and_snapshot(model, cfg, model_path):
    model.save(str(model_path))
    cfg.save(str(model_path.with_suffix(".yaml")))


def main():
    source_paths = create_run_paths(SOURCE_RUN_NAME)
    resume_paths = create_run_paths(RESUME_RUN_NAME)
    source_model_path = resolve_model_stem(source_paths)
    source_snapshot_path = source_model_path.with_suffix(".yaml")

    cfg = Config(str(source_snapshot_path))
    env = create_env(cfg)

    try:
        if RUN_ENV_CHECK:
            check_env(env, warn=True)

        model = PPO.load(str(source_model_path), env=env, device=DEVICE)
        model.set_logger(
            configure(str(resume_paths["log_dir"]), ["stdout", "csv", "tensorboard"])
        )

        current_timesteps = int(model.num_timesteps)
        if NEW_TOTAL_TIMESTEPS is None:
            resume_total_timesteps = current_timesteps + EXTRA_TRAINING_STEPS
        else:
            resume_total_timesteps = NEW_TOTAL_TIMESTEPS

        callbacks, callback_parts = create_callbacks(
            cfg,
            env,
            resume_total_timesteps,
            resume_paths,
        )
        seed_callback_timesteps(callback_parts, current_timesteps)
        cfg.save(str(resume_paths["initial_snapshot_path"]))

        print(f"Resuming from: {source_model_path}.zip")
        print(f"Snapshot file: {source_snapshot_path}")
        print(f"Saved timesteps: {current_timesteps:,}")
        print(f"Extra training steps: {EXTRA_TRAINING_STEPS:,}")
        print(f"Curriculum target total: {resume_total_timesteps:,}")
        print(f"New run directory: {resume_paths['run_dir']}")
        print(f"TensorBoard log dir: {resume_paths['log_dir']}")
        print("Watch in the dashboard: rollout/ep_rew_mean, rollout/ep_len_mean, "
              "train/approx_kl, train/value_loss, train/explained_variance, "
              "curriculum/terrain_z_max, curriculum/push_velocity, "
              "curriculum/friction_value, curriculum/motor_noise_std, "
              "curriculum/sensor_noise_scale")

        try:
            model.learn(
                total_timesteps=EXTRA_TRAINING_STEPS,
                callback=callbacks,
                log_interval=1,
                reset_num_timesteps=False,
            )
        except KeyboardInterrupt:
            interrupted_path = resume_paths["run_dir"] / f"ppo_{RESUME_RUN_NAME}_interrupted"
            save_model_and_snapshot(model, cfg, interrupted_path)
            print(f"\nResume interrupted. Saved: {interrupted_path}.zip")
            return

        save_model_and_snapshot(model, cfg, resume_paths["final_model_path"])
        print(f"\nResume complete. Saved: {resume_paths['final_model_path']}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
