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
# Training Configuration
# Edit these values directly in the file, then run the script from your editor.
# ============================================================================

RUN_NAME = "robust_walk_v1"          # Folder name and file prefix for this run.
TOTAL_STEPS = 10_000_000            # Total PPO training timesteps.

BACKEND = "pybullet"                # Physics engine: "pybullet" or "mujoco".
USE_GUI = False                     # True to watch training live, False for faster headless runs.
DEVICE = "cpu"                      # PPO device. For this MLP setup, CPU is usually best.

RUN_ENV_CHECK = False               # True to run Stable-Baselines3 env checks before training.
MAX_EPISODE_LEN = 3000              # Max control steps per episode before truncation.
SIM_FREQUENCY = 240                 # Physics frequency in Hz.
CONTROL_FREQUENCY = 60              # Policy/action frequency in Hz.

STEP_CHECKPOINT_FREQUENCY = 500_000 # Save a model+snapshot every N training steps.
TIMED_SNAPSHOT_SECONDS = 300        # Save a model+snapshot every N real seconds.

MODEL_VERBOSE = 1                   # PPO console logging level.
CURRICULUM_VERBOSE = False          # Print terrain/friction/noise curriculum messages.
SAVE_VERBOSE = 1                    # Print checkpoint save messages.

LEARNING_RATE = 3e-4                # Initial PPO learning rate.
MIN_LEARNING_RATE = 1e-5            # Lowest value reached by the linear LR schedule.
ENTROPY_COEF = 0.01                 # Exploration bonus strength.
CLIP_RANGE = 0.2                    # PPO clip range.
N_STEPS = 2048                      # Rollout size collected before each PPO update.
BATCH_SIZE = 64                     # Mini-batch size used during PPO updates.
GAMMA = 0.99                        # Discount factor for future rewards.
GAE_LAMBDA = 0.95                   # GAE smoothing factor.

TERRAIN_SETTINGS = {
    "schedule": "linear",           # How difficulty ramps up: "linear" or "exponential".
    "warmup_ratio": 0.05,           # Fraction of training kept at the easiest terrain.
    "z_max_initial": 0.02,          # Initial terrain height variation in meters.
    "z_max_final": 0.30,            # Final terrain height variation in meters.
    "change_every_episodes": 50,    # Spawn a new terrain every N episodes.
}

FORCE_SETTINGS = {
    "schedule": "linear",           # How push intensity ramps up.
    "warmup_ratio": 0.05,           # Fraction of training with the gentlest pushes.
    "push_vel_initial": 0.10,       # Initial push strength as target body velocity change [m/s].
    "push_vel_final": 1.50,         # Final push strength as target body velocity change [m/s].
    "push_interval_s": 15.0,        # Average time between push events [s].
    "push_duration_steps": 2,       # Number of control steps each push lasts.
}

FRICTION_SETTINGS = {
    "schedule": "linear",           # How the friction range widens.
    "warmup_ratio": 0.05,           # Fraction of training with the narrowest friction range.
    "friction_initial_low": 0.9,    # Initial minimum ground friction.
    "friction_initial_high": 1.1,   # Initial maximum ground friction.
    "friction_final_low": 0.4,      # Final minimum ground friction.
    "friction_final_high": 1.5,     # Final maximum ground friction.
}

MOTOR_NOISE_SETTINGS = {
    "schedule": "linear",           # How actuator noise ramps up.
    "warmup_ratio": 0.05,           # Fraction of training with the cleanest actuators.
    "noise_initial": 0.0,           # Initial motor noise standard deviation [rad].
    "noise_final": 0.05,            # Final motor noise standard deviation [rad].
    "noise_type": "gaussian",       # Noise distribution: "gaussian" or "uniform".
}

SENSOR_NOISE_SETTINGS = {
    "schedule": "linear",           # How observation noise ramps up.
    "warmup_ratio": 0.05,           # Fraction of training with the cleanest observations.
    "noise_scale_initial": 0.0,     # Initial global multiplier for sensor noise.
    "noise_scale_final": 1.0,       # Final global multiplier for sensor noise.
    "dof_pos_noise": 0.01,          # Joint position noise standard deviation [rad].
    "dof_vel_noise": 1.5,           # Joint velocity noise standard deviation [rad/s].
    "lin_vel_noise": 0.1,           # Base linear velocity noise [m/s].
    "ang_vel_noise": 0.2,           # Base angular velocity noise [rad/s].
}


class StepSnapshotCallback(BaseCallback):
    def __init__(self, save_dir, save_freq_steps, config, verbose=1):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_freq_steps = save_freq_steps
        self.config = config
        self.last_saved_timestep = 0

    def _on_training_start(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        if self.save_freq_steps <= 0:
            return True

        if self.num_timesteps - self.last_saved_timestep < self.save_freq_steps:
            return True

        self.last_saved_timestep = int(self.num_timesteps)
        save_path = self.save_dir / f"ppo_{RUN_NAME}_steps_{self.num_timesteps:09d}"
        snapshot_path = save_path.with_suffix(".yaml")

        self.model.save(str(save_path))
        self.config.save(str(snapshot_path))

        if self.verbose:
            print(f"[StepCheckpoint] Saved model to: {save_path}.zip")
            print(f"[StepCheckpoint] Saved snapshot to: {snapshot_path}")

        return True


class TimedSnapshotCallback(BaseCallback):
    def __init__(self, save_dir, interval_seconds, config, verbose=1):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.interval_seconds = interval_seconds
        self.config = config
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
            f"ppo_{RUN_NAME}_timed_{self.save_index:03d}_{elapsed_minutes:04d}min"
        )
        snapshot_path = save_path.with_suffix(".yaml")

        self.model.save(str(save_path))
        self.config.save(str(snapshot_path))
        self.last_save_time = now

        if self.verbose:
            print(f"[TimedCheckpoint] Saved model to: {save_path}.zip")
            print(f"[TimedCheckpoint] Saved snapshot to: {snapshot_path}")

        return True


def create_run_paths():
    run_dir = TRAINING_DIR / "data" / f"{RUN_NAME}_robust"
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
        "initial_snapshot_path": run_dir / f"{RUN_NAME}_initial.yaml",
        "final_model_path": run_dir / f"ppo_{RUN_NAME}",
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


def clipped_linear_schedule(initial_value, min_value=1e-5):
    def schedule(progress_remaining):
        return max(progress_remaining * initial_value, min_value)

    return schedule


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


def create_callbacks(cfg, env, paths):
    terrain_callback = TerrainCurriculumCallbackV2(
        config=cfg,
        env=env,
        total_timesteps=TOTAL_STEPS,
        verbose=CURRICULUM_VERBOSE,
        **TERRAIN_SETTINGS,
    )
    force_callback = ForceCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=TOTAL_STEPS,
        verbose=CURRICULUM_VERBOSE,
        **FORCE_SETTINGS,
    )
    friction_callback = FrictionCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=TOTAL_STEPS,
        verbose=CURRICULUM_VERBOSE,
        **FRICTION_SETTINGS,
    )
    motor_noise_callback = MotorNoiseCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=TOTAL_STEPS,
        verbose=CURRICULUM_VERBOSE,
        **MOTOR_NOISE_SETTINGS,
    )
    sensor_noise_callback = SensorNoiseCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=TOTAL_STEPS,
        verbose=CURRICULUM_VERBOSE,
        **SENSOR_NOISE_SETTINGS,
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
                verbose=SAVE_VERBOSE,
            )
        )

    if TIMED_SNAPSHOT_SECONDS > 0:
        callbacks.append(
            TimedSnapshotCallback(
                save_dir=paths["timed_checkpoint_dir"],
                interval_seconds=TIMED_SNAPSHOT_SECONDS,
                config=cfg,
                verbose=SAVE_VERBOSE,
            )
        )

    return CallbackList(callbacks)


def create_model(env, log_dir):
    model = PPO(
        "MlpPolicy",
        env,
        verbose=MODEL_VERBOSE,
        learning_rate=clipped_linear_schedule(LEARNING_RATE, MIN_LEARNING_RATE),
        ent_coef=ENTROPY_COEF,
        clip_range=CLIP_RANGE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        tensorboard_log=str(log_dir),
        device=DEVICE,
    )
    model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))
    return model


def save_model_and_snapshot(model, cfg, model_path):
    model.save(str(model_path))
    cfg.save(str(model_path.with_suffix(".yaml")))


def main():
    paths = create_run_paths()
    cfg = Config()
    env = create_env(cfg)

    try:
        if RUN_ENV_CHECK:
            check_env(env, warn=True)

        callbacks = create_callbacks(cfg, env, paths)
        cfg.save(str(paths["initial_snapshot_path"]))

        model = create_model(env, paths["log_dir"])

        print(f"Training run: {RUN_NAME}")
        print(f"Total steps: {TOTAL_STEPS:,}")
        print(f"Run directory: {paths['run_dir']}")
        print(f"TensorBoard log dir: {paths['log_dir']}")
        print(f"Step checkpoints: every {STEP_CHECKPOINT_FREQUENCY:,} steps")
        print(f"Timed snapshots: every {TIMED_SNAPSHOT_SECONDS} real seconds")
        print("Watch in the dashboard: rollout/ep_rew_mean, rollout/ep_len_mean, "
              "train/approx_kl, train/value_loss, train/explained_variance, "
              "curriculum/terrain_z_max, curriculum/push_velocity, "
              "curriculum/friction_value, curriculum/motor_noise_std, "
              "curriculum/sensor_noise_scale")

        try:
            model.learn(
                total_timesteps=TOTAL_STEPS,
                callback=callbacks,
                log_interval=1,
                reset_num_timesteps=True,
            )
        except KeyboardInterrupt:
            interrupted_path = paths["run_dir"] / f"ppo_{RUN_NAME}_interrupted"
            save_model_and_snapshot(model, cfg, interrupted_path)
            print(f"\nTraining interrupted. Saved: {interrupted_path}.zip")
            return

        save_model_and_snapshot(model, cfg, paths["final_model_path"])
        print(f"\nTraining complete. Saved: {paths['final_model_path']}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
