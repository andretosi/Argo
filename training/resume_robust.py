from __future__ import annotations

from stable_baselines3.common.env_checker import check_env

from robust_common import (
    build_run_paths,
    create_robust_env,
    create_training_callbacks,
    default_reward_config,
    load_model,
    resolve_model_stem,
    save_model_and_snapshot,
    seed_curriculum_timesteps,
)
from spotmicro.tools.config import Config


# ============================================================================
# Resume Configuration
# Edit these values directly in the file, then run the script from your editor.
# ============================================================================

SOURCE_RUN_NAME = "robust_walk_v1"
SOURCE_MODEL_STEM = None

RESUME_RUN_NAME = "robust_walk_v1_resume"
EXTRA_TRAINING_STEPS = 10_000
NEW_TOTAL_TIMESTEPS = None

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


def main() -> None:
    source_paths = build_run_paths(SOURCE_RUN_NAME)
    resume_paths = build_run_paths(RESUME_RUN_NAME)
    source_model_stem = resolve_model_stem(source_paths, SOURCE_MODEL_STEM)
    source_snapshot_path = source_model_stem.with_suffix(".yaml")

    cfg = Config(str(source_snapshot_path))
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

        model = load_model(
            source_model_stem,
            env,
            log_dir=resume_paths.log_dir,
            device=DEVICE,
        )

        current_timesteps = int(model.num_timesteps)
        resume_total_timesteps = NEW_TOTAL_TIMESTEPS
        if resume_total_timesteps is None:
            resume_total_timesteps = current_timesteps + EXTRA_TRAINING_STEPS

        callbacks, callback_parts = create_training_callbacks(
            cfg=cfg,
            env=env,
            paths=resume_paths,
            total_timesteps=resume_total_timesteps,
            step_checkpoint_freq=STEP_CHECKPOINT_FREQUENCY,
            timed_checkpoint_seconds=TIMED_SNAPSHOT_SECONDS,
            terrain_settings={},
            force_settings={},
            friction_settings={},
            motor_noise_settings={},
            sensor_noise_settings={},
            curriculum_verbose=CURRICULUM_VERBOSE,
            save_verbose=SAVE_VERBOSE,
        )
        seed_curriculum_timesteps(callback_parts, current_timesteps)

        cfg.save(str(resume_paths.initial_snapshot_path))

        print(f"Resuming from: {source_model_stem}.zip")
        print(f"Snapshot file: {source_snapshot_path}")
        print(f"Saved timesteps: {current_timesteps:,}")
        print(f"Extra training steps: {EXTRA_TRAINING_STEPS:,}")
        print(f"Curriculum target total: {resume_total_timesteps:,}")
        print(f"New run directory: {resume_paths.run_dir}")

        try:
            model.learn(
                total_timesteps=EXTRA_TRAINING_STEPS,
                callback=callbacks,
                log_interval=1,
                reset_num_timesteps=False,
            )
        except KeyboardInterrupt:
            interrupted_model = resume_paths.run_dir / f"ppo_{RESUME_RUN_NAME}_interrupted"
            save_model_and_snapshot(model, cfg, interrupted_model)
            print(f"\nResume interrupted. Saved: {interrupted_model}.zip")
            return

        save_model_and_snapshot(model, cfg, resume_paths.final_model_path)
        print(f"\nResume complete. Saved: {resume_paths.final_model_path}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
