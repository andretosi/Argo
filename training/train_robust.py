from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.env_checker import check_env

from robust_common import (
    build_run_paths,
    create_new_model,
    create_robust_env,
    create_training_callbacks,
    default_reward_config,
    save_model_and_snapshot,
)
from spotmicro.tools.config import Config


# ============================================================================
# Training Configuration
# Edit these values directly in the file, then run the script from your editor.
# ============================================================================

RUN_NAME = "robust_walk_v1"
TOTAL_STEPS = 10_000

BACKEND = "pybullet"
USE_GUI = False
DEVICE = "cpu"

RUN_ENV_CHECK = False
MAX_EPISODE_LEN = 3000
SIM_FREQUENCY = 240
CONTROL_FREQUENCY = 60

STEP_CHECKPOINT_FREQUENCY = 2_500
TIMED_SNAPSHOT_SECONDS = 300

MODEL_VERBOSE = 1
CURRICULUM_VERBOSE = False
SAVE_VERBOSE = 1

LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 1e-5
ENTROPY_COEF = 0.01
CLIP_RANGE = 0.2
N_STEPS = 2048
BATCH_SIZE = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95

TERRAIN_SETTINGS = {
    "schedule": "linear",
    "warmup_ratio": 0.05,
    "z_max_initial": 0.02,
    "z_max_final": 0.30,
    "change_every_episodes": 50,
}

FORCE_SETTINGS = {
    "schedule": "linear",
    "warmup_ratio": 0.05,
    "push_vel_initial": 0.10,
    "push_vel_final": 1.50,
    "push_interval_s": 15.0,
    "push_duration_steps": 2,
}

FRICTION_SETTINGS = {
    "schedule": "linear",
    "warmup_ratio": 0.05,
    "friction_initial_low": 0.9,
    "friction_initial_high": 1.1,
    "friction_final_low": 0.4,
    "friction_final_high": 1.5,
}

MOTOR_NOISE_SETTINGS = {
    "schedule": "linear",
    "warmup_ratio": 0.05,
    "noise_initial": 0.0,
    "noise_final": 0.05,
    "noise_type": "gaussian",
}

SENSOR_NOISE_SETTINGS = {
    "schedule": "linear",
    "warmup_ratio": 0.05,
    "noise_scale_initial": 0.0,
    "noise_scale_final": 1.0,
    "dof_pos_noise": 0.01,
    "dof_vel_noise": 1.5,
    "lin_vel_noise": 0.1,
    "ang_vel_noise": 0.2,
}


def main() -> None:
    paths = build_run_paths(RUN_NAME)
    cfg = Config()
    reward_config = default_reward_config()

    env = create_robust_env(
        cfg,
        backend_name=BACKEND,
        use_gui=USE_GUI,
        max_episode_len=MAX_EPISODE_LEN,
        sim_frequency=SIM_FREQUENCY,
        control_frequency=CONTROL_FREQUENCY,
        reward_config=reward_config,
    )

    try:
        if RUN_ENV_CHECK:
            check_env(env, warn=True)

        callbacks, _ = create_training_callbacks(
            cfg=cfg,
            env=env,
            paths=paths,
            total_timesteps=TOTAL_STEPS,
            step_checkpoint_freq=STEP_CHECKPOINT_FREQUENCY,
            timed_checkpoint_seconds=TIMED_SNAPSHOT_SECONDS,
            terrain_settings=TERRAIN_SETTINGS,
            force_settings=FORCE_SETTINGS,
            friction_settings=FRICTION_SETTINGS,
            motor_noise_settings=MOTOR_NOISE_SETTINGS,
            sensor_noise_settings=SENSOR_NOISE_SETTINGS,
            curriculum_verbose=CURRICULUM_VERBOSE,
            save_verbose=SAVE_VERBOSE,
        )

        cfg.save(str(paths.initial_snapshot_path))

        model = create_new_model(
            env,
            log_dir=paths.log_dir,
            device=DEVICE,
            verbose=MODEL_VERBOSE,
            learning_rate=LEARNING_RATE,
            min_learning_rate=MIN_LEARNING_RATE,
            ent_coef=ENTROPY_COEF,
            clip_range=CLIP_RANGE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
        )

        print(f"Training run: {RUN_NAME}")
        print(f"Total steps: {TOTAL_STEPS:,}")
        print(f"Run directory: {paths.run_dir}")
        print(f"Step checkpoints: every {STEP_CHECKPOINT_FREQUENCY:,} steps")
        print(f"Timed snapshots: every {TIMED_SNAPSHOT_SECONDS} real seconds")

        try:
            model.learn(
                total_timesteps=TOTAL_STEPS,
                callback=callbacks,
                log_interval=1,
                reset_num_timesteps=True,
            )
        except KeyboardInterrupt:
            interrupted_model = paths.run_dir / f"ppo_{RUN_NAME}_interrupted"
            save_model_and_snapshot(model, cfg, interrupted_model)
            print(f"\nTraining interrupted. Saved: {interrupted_model}.zip")
            return

        save_model_and_snapshot(model, cfg, paths.final_model_path)
        print(f"\nTraining complete. Saved: {paths.final_model_path}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
