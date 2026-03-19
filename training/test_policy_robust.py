from __future__ import annotations

from robust_common import (
    attach_noise_callbacks_for_testing,
    build_run_paths,
    create_force_callback_for_testing,
    create_robust_env,
    curriculum_factor_from_step,
    default_reward_config,
    interpolate,
    load_model,
    resolve_model_stem,
    run_policy_rollout,
    spawn_noise_terrain,
)
from spotmicro.tools.config import Config


# ============================================================================
# Policy Test Configuration
# Edit these values directly in the file, then run the script from your editor.
# ============================================================================

RUN_NAME = "robust_walk_v1"
MODEL_STEM = None

BACKEND = "pybullet"
USE_GUI = True
DEVICE = "cpu"

MAX_EPISODE_LEN = 3000
SIM_FREQUENCY = 240
CONTROL_FREQUENCY = 60

N_EVAL_EPISODES = 3
DETERMINISTIC = True
REALTIME_SLEEP = 1.0 / 240.0

# Terrain mode options:
# - "flat"            -> leave the default flat ground
# - "manual"          -> spawn a terrain with the values below
# - "random"          -> random terrain with the values below
# - "curriculum_step" -> terrain difficulty based on the chosen training step
TERRAIN_MODE = "flat"

TERRAIN_SIZE = 256
TERRAIN_SCALE = (0.02, 0.02, 1.0)
TERRAIN_ORIGIN = (0.0, 0.0, 0.0)

MANUAL_TERRAIN_Z_MAX = 0.12
MANUAL_TERRAIN_SEED = 12345

TRAIN_TOTAL_TIMESTEPS = 10_000_000
CURRICULUM_STEP = 100_000
CURRICULUM_WARMUP_RATIO = 0.05
CURRICULUM_SCHEDULE = "linear"
RANDOMIZE_CURRICULUM_TERRAIN = True
CURRICULUM_TERRAIN_SEED = 12345

CURRICULUM_TERRAIN_Z_MAX_INITIAL = 0.02
CURRICULUM_TERRAIN_Z_MAX_FINAL = 0.30

# Flat/default test values.
# Keep everything explicit so it is obvious what the policy is facing.
GROUND_FRICTION = 1.0
MOTOR_NOISE_STD = 0.0
SENSOR_NOISE_SCALE = 0.0
PUSH_VELOCITY = 0.0
PUSH_INTERVAL_S = 15.0
PUSH_DURATION_STEPS = 2

# Optional: when using "curriculum_step", also derive the other perturbations
# from the same curriculum factor instead of using the flat/default values above.
APPLY_FULL_CURRICULUM_STATE = False

CURRICULUM_FRICTION_INITIAL = (0.9, 1.1)
CURRICULUM_FRICTION_FINAL = (0.4, 1.5)
CURRICULUM_MOTOR_NOISE_INITIAL = 0.0
CURRICULUM_MOTOR_NOISE_FINAL = 0.05
CURRICULUM_SENSOR_NOISE_INITIAL = 0.0
CURRICULUM_SENSOR_NOISE_FINAL = 1.0
CURRICULUM_PUSH_VELOCITY_INITIAL = 0.10
CURRICULUM_PUSH_VELOCITY_FINAL = 1.50


def main() -> None:
    paths = build_run_paths(RUN_NAME)
    model_stem = resolve_model_stem(paths, MODEL_STEM)

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
        model = load_model(
            model_stem,
            env,
            log_dir=paths.log_dir,
            device=DEVICE,
        )

        friction_value = GROUND_FRICTION
        motor_noise_std = MOTOR_NOISE_STD
        sensor_noise_scale = SENSOR_NOISE_SCALE
        push_velocity = PUSH_VELOCITY
        terrain_seed = None
        terrain_z_max = None

        if TERRAIN_MODE == "manual":
            terrain_seed, terrain_z_max = spawn_noise_terrain(
                env,
                z_max=MANUAL_TERRAIN_Z_MAX,
                terrain_size=TERRAIN_SIZE,
                scale=TERRAIN_SCALE,
                origin=TERRAIN_ORIGIN,
                seed=MANUAL_TERRAIN_SEED,
            )
        elif TERRAIN_MODE == "random":
            terrain_seed, terrain_z_max = spawn_noise_terrain(
                env,
                z_max=MANUAL_TERRAIN_Z_MAX,
                terrain_size=TERRAIN_SIZE,
                scale=TERRAIN_SCALE,
                origin=TERRAIN_ORIGIN,
                seed=None,
            )
        elif TERRAIN_MODE == "curriculum_step":
            factor = curriculum_factor_from_step(
                CURRICULUM_STEP,
                TRAIN_TOTAL_TIMESTEPS,
                warmup_ratio=CURRICULUM_WARMUP_RATIO,
                schedule=CURRICULUM_SCHEDULE,
            )
            terrain_z_max = interpolate(
                CURRICULUM_TERRAIN_Z_MAX_INITIAL,
                CURRICULUM_TERRAIN_Z_MAX_FINAL,
                factor,
            )
            terrain_seed, terrain_z_max = spawn_noise_terrain(
                env,
                z_max=terrain_z_max,
                terrain_size=TERRAIN_SIZE,
                scale=TERRAIN_SCALE,
                origin=TERRAIN_ORIGIN,
                seed=None if RANDOMIZE_CURRICULUM_TERRAIN else CURRICULUM_TERRAIN_SEED,
            )

            if APPLY_FULL_CURRICULUM_STATE:
                friction_low = interpolate(
                    CURRICULUM_FRICTION_INITIAL[0],
                    CURRICULUM_FRICTION_FINAL[0],
                    factor,
                )
                friction_high = interpolate(
                    CURRICULUM_FRICTION_INITIAL[1],
                    CURRICULUM_FRICTION_FINAL[1],
                    factor,
                )
                friction_value = 0.5 * (friction_low + friction_high)
                motor_noise_std = interpolate(
                    CURRICULUM_MOTOR_NOISE_INITIAL,
                    CURRICULUM_MOTOR_NOISE_FINAL,
                    factor,
                )
                sensor_noise_scale = interpolate(
                    CURRICULUM_SENSOR_NOISE_INITIAL,
                    CURRICULUM_SENSOR_NOISE_FINAL,
                    factor,
                )
                push_velocity = interpolate(
                    CURRICULUM_PUSH_VELOCITY_INITIAL,
                    CURRICULUM_PUSH_VELOCITY_FINAL,
                    factor,
                )

            print(f"Curriculum step: {CURRICULUM_STEP:,}")
            print(f"Curriculum factor: {factor:.3f}")

        env._backend.set_friction(friction_value)
        attach_noise_callbacks_for_testing(
            cfg,
            env,
            motor_noise_std=motor_noise_std,
            sensor_noise_scale=sensor_noise_scale,
            verbose=False,
        )
        force_callback = create_force_callback_for_testing(
            cfg,
            env,
            push_velocity=push_velocity,
            push_interval_s=PUSH_INTERVAL_S,
            push_duration_steps=PUSH_DURATION_STEPS,
            verbose=False,
        )

        print(f"Testing model: {model_stem}.zip")
        print(f"Terrain mode: {TERRAIN_MODE}")
        print(f"Terrain seed: {terrain_seed}")
        print(f"Terrain z_max: {terrain_z_max}")
        print(f"Ground friction: {friction_value}")
        print(f"Motor noise std: {motor_noise_std}")
        print(f"Sensor noise scale: {sensor_noise_scale}")
        print(f"Push velocity: {push_velocity}")

        run_policy_rollout(
            model=model,
            env=env,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=DETERMINISTIC,
            force_callback=force_callback,
            realtime_sleep=REALTIME_SLEEP if USE_GUI else 0.0,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
