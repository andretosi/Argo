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
    CompetenceCurriculumCallback,
    ForceCurriculumCallback,
    FrictionCurriculumCallback,
    MotorNoiseCurriculumCallback,
    SensorNoiseCurriculumCallback,
    TerrainCurriculumCallback,
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
TOTAL_STEPS = 10_000_000           # Total PPO training timesteps.

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

CONTROLLER_SETTINGS = {
    "p_base2still": 0.05,           # Initial command mix: brief pauses.
    "p_base2walk": 0.70,            # Initial command mix: mostly walking.
    "p_base2turn": 0.25,            # Initial command mix: allow turning from the start.
    "p_still2walk": 0.80,           # After standing, usually ask for walking again.
    "p_still2turn": 0.20,           # After standing, sometimes ask for a turn.
    "p_walk2still": 0.20,           # End a walk segment with a short pause sometimes.
    "p_walk2turn": 0.80,            # End a walk segment with a turn most of the time.
    "p_turn2still": 0.10,           # End a turn with a pause only occasionally.
    "p_turn2walk": 0.90,            # After turning, usually return to walking.
    "v_mean": (0.0, 0.0),           # Center the walking command around zero for omni-directionality.
    "v_var": (0.35, 0.25),          # Spread forward/backward and lateral commands.
    "v_steps_mean": 220,            # Keep each walk command for a few seconds.
    "v_steps_var": 30,              # Small variation in walk-command duration.
    "w_mean": 0.0,                  # No preferred turn direction.
    "w_var": 0.35,                  # Moderate yaw-rate variation.
    "w_radius_mean": 0.10,          # Favor near in-place turning over wide arcs.
    "w_radius_var": 0.15,           # Still allow some curved turns.
    "w_steps_mean": 120,            # Keep turning commands long enough to learn them.
    "w_steps_var": 18,              # Small variation in turn-command duration.
    "s_steps_mean": 25,             # Pauses stay short.
    "s_steps_var": 6,               # Small variation in pause length.
}

TERRAIN_SETTINGS = {
    "schedule": "exponential",      # Ramp difficulty slowly at first, then more later.
    "warmup_ratio": 0.10,           # Stay easy for longer to stabilize the base gait first.
    "z_max_initial": 0.01,          # Start with almost-flat terrain.
    "z_max_final": 0.10,            # Final roughness kept moderate for smoother omni walking.
    "change_every_episodes": 80,    # Give the policy longer to adapt before changing terrain.
}

FORCE_SETTINGS = {
    "schedule": "exponential",      # Keep pushes mild for most of training.
    "warmup_ratio": 0.25,           # Learn command tracking before meaningful pushes start.
    "push_vel_initial": 0.0,        # Start with no pushes at all.
    "push_vel_final": 0.80,         # Final pushes stay moderate to protect gait smoothness.
    "push_interval_s": 20.0,        # Push less frequently than before.
    "push_duration_steps": 2,       # Number of control steps each push lasts.
}

FRICTION_SETTINGS = {
    "schedule": "exponential",      # Widen friction range gradually.
    "warmup_ratio": 0.15,           # Keep friction close to nominal early on.
    "friction_initial_low": 0.85,   # Initial minimum ground friction.
    "friction_initial_high": 1.15,  # Initial maximum ground friction.
    "friction_final_low": 0.65,     # Final minimum ground friction.
    "friction_final_high": 1.25,    # Final maximum ground friction.
}

MOTOR_NOISE_SETTINGS = {
    "schedule": "exponential",      # Add actuator noise only after the gait is stable.
    "warmup_ratio": 0.30,           # Delay actuator noise more than terrain or friction.
    "noise_initial": 0.0,           # Initial motor noise standard deviation [rad].
    "noise_final": 0.02,            # Final motor noise standard deviation [rad].
    "noise_type": "gaussian",       # Noise distribution: "gaussian" or "uniform".
}

SENSOR_NOISE_SETTINGS = {
    "schedule": "exponential",      # Delay observation noise until command tracking works.
    "warmup_ratio": 0.25,           # Keep observations clean early on.
    "noise_scale_initial": 0.0,     # Initial global multiplier for sensor noise.
    "noise_scale_final": 0.5,       # Final global multiplier for sensor noise.
    "dof_pos_noise": 0.01,          # Joint position noise standard deviation [rad].
    "dof_vel_noise": 1.5,           # Joint velocity noise standard deviation [rad/s].
    "lin_vel_noise": 0.1,           # Base linear velocity noise [m/s].
    "ang_vel_noise": 0.2,           # Base angular velocity noise [rad/s].
}

COMPETENCE_SETTINGS = {
    "progression_mode": "competence",   # Use competence instead of elapsed timesteps for curriculum progress.
    "ema_alpha": 0.10,                  # EMA smoothing factor for the competence score.
    "threshold": 0.70,                  # Competence EMA needed before unlocking harder difficulty.
    "advance_step": 0.05,               # Increase shared curriculum progress by this amount when unlocked.
    "min_episodes_between_advances": 10,# Wait this many episodes before the next difficulty increase.
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
        tracking_lin_vel=1.25,
        tracking_ang_vel=0.75,
        tracking_sigma=0.25,
        feet_air_time=0.5,
        lin_vel_z=-2.0,
        ang_vel_xy=-0.1,
        orientation=-1.5,
        base_height=-1.5,
        action_rate=-0.02,
        torques=-0.00002,
        dof_acc=-5e-7,
        power=-0.0002,
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
    device = RandomController(cfg, **CONTROLLER_SETTINGS)
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
    competence_callback = CompetenceCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=TOTAL_STEPS,
        ema_alpha=COMPETENCE_SETTINGS["ema_alpha"],
        threshold=COMPETENCE_SETTINGS["threshold"],
        advance_step=COMPETENCE_SETTINGS["advance_step"],
        min_episodes_between_advances=COMPETENCE_SETTINGS["min_episodes_between_advances"],
        verbose=False,
    )
    terrain_callback = TerrainCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=TOTAL_STEPS,
        progression_mode=COMPETENCE_SETTINGS["progression_mode"],
        verbose=CURRICULUM_VERBOSE,
        **TERRAIN_SETTINGS,
    )
    force_callback = ForceCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=TOTAL_STEPS,
        progression_mode=COMPETENCE_SETTINGS["progression_mode"],
        verbose=CURRICULUM_VERBOSE,
        **FORCE_SETTINGS,
    )
    friction_callback = FrictionCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=TOTAL_STEPS,
        progression_mode=COMPETENCE_SETTINGS["progression_mode"],
        verbose=CURRICULUM_VERBOSE,
        **FRICTION_SETTINGS,
    )
    motor_noise_callback = MotorNoiseCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=TOTAL_STEPS,
        progression_mode=COMPETENCE_SETTINGS["progression_mode"],
        verbose=CURRICULUM_VERBOSE,
        **MOTOR_NOISE_SETTINGS,
    )
    sensor_noise_callback = SensorNoiseCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=TOTAL_STEPS,
        progression_mode=COMPETENCE_SETTINGS["progression_mode"],
        verbose=CURRICULUM_VERBOSE,
        **SENSOR_NOISE_SETTINGS,
    )

    callbacks = [
        competence_callback,
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
              "competence/ema, competence/progress, "
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
