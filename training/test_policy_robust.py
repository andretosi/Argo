import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
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
from spotmicro.tools.TerrainTools import Heightmap
from spotmicro.tools.config import Config
from training.callbacks import ForceCurriculumCallback, MotorNoiseCurriculumCallback, SensorNoiseCurriculumCallback
from training.reward_functions.robust_walking_reward import (
    RewardConfig,
    RewardState,
    reward_function,
)


# ============================================================================
# Policy Test Configuration
# Edit these values directly in the file, then run the script from your editor.
# ============================================================================

RUN_NAME = "robust_walk_v1"        # Run folder to test.
MODEL_STEM = None                  # Explicit model path without .zip, or None for latest snapshot.

BACKEND = "pybullet"               # Physics engine used for evaluation.
USE_GUI = True                     # True to visualize the policy.
DEVICE = "cpu"                     # PPO device for policy inference.

MAX_EPISODE_LEN = 3000             # Max control steps per evaluation episode.
SIM_FREQUENCY = 240                # Physics frequency in Hz.
CONTROL_FREQUENCY = 60             # Policy/action frequency in Hz.

N_EVAL_EPISODES = 3                # Number of rollout episodes to run.
DETERMINISTIC = True               # True for deterministic policy actions.
REALTIME_SLEEP = 1.0 / 240.0       # Extra sleep per control step when GUI is on.

# Terrain mode options:
# - "flat"            -> keep the default flat ground
# - "manual"          -> spawn terrain with the manual values below
# - "random"          -> spawn random terrain with the manual z_max below
# - "curriculum_step" -> use the terrain difficulty of a chosen training step
TERRAIN_MODE = "flat"              # "flat", "manual", "random", or "curriculum_step".

TERRAIN_SIZE = 256                 # Heightmap resolution for generated terrains.
TERRAIN_SCALE = (0.02, 0.02, 1.0)  # Terrain scaling [x, y, z].
TERRAIN_ORIGIN = (0.0, 0.0, 0.0)   # Terrain origin [x, y, z].

MANUAL_TERRAIN_Z_MAX = 0.12        # Terrain height variation used in "manual" mode.
MANUAL_TERRAIN_SEED = 12345        # Terrain seed used in "manual" mode.

TRAIN_TOTAL_TIMESTEPS = 10_000_000 # Total training length assumed by the curriculum.
CURRICULUM_STEP = 100_000          # Virtual training step used in "curriculum_step" mode.
CURRICULUM_WARMUP_RATIO = 0.10     # Fraction of training kept at the easiest difficulty.
CURRICULUM_SCHEDULE = "exponential" # "linear" or "exponential" curriculum interpolation.
RANDOMIZE_CURRICULUM_TERRAIN = True # True for a new terrain seed at that curriculum step.
CURRICULUM_TERRAIN_SEED = 12345    # Terrain seed used when the curriculum terrain is fixed.

CURRICULUM_TERRAIN_Z_MAX_INITIAL = 0.01 # Terrain height variation at training start.
CURRICULUM_TERRAIN_Z_MAX_FINAL = 0.10   # Terrain height variation at training end.

# Default test conditions.
# Keeping everything explicit makes it easy to understand what the policy sees.
GROUND_FRICTION = 1.0              # Fixed ground friction used for the test.
MOTOR_NOISE_STD = 0.0              # Fixed motor noise standard deviation [rad].
SENSOR_NOISE_SCALE = 0.0           # Fixed sensor-noise multiplier.
PUSH_VELOCITY = 0.0                # Fixed push strength as target body velocity change [m/s].
PUSH_INTERVAL_S = 20.0             # Average time between pushes [s] when pushes are enabled.
PUSH_DURATION_STEPS = 2            # Number of control steps each push lasts.

# If True, "curriculum_step" also adjusts friction and perturbations using the
# same curriculum factor. If False, those stay on the explicit defaults above.
APPLY_FULL_CURRICULUM_STATE = False # If True, also derive friction/noise/pushes from CURRICULUM_STEP.

CURRICULUM_FRICTION_INITIAL = (0.85, 1.15) # Friction range at training start.
CURRICULUM_FRICTION_FINAL = (0.65, 1.25)   # Friction range at training end.
CURRICULUM_MOTOR_NOISE_INITIAL = 0.0      # Motor noise at training start [rad].
CURRICULUM_MOTOR_NOISE_FINAL = 0.02       # Motor noise at training end [rad].
CURRICULUM_SENSOR_NOISE_INITIAL = 0.0     # Sensor-noise scale at training start.
CURRICULUM_SENSOR_NOISE_FINAL = 0.5       # Sensor-noise scale at training end.
CURRICULUM_PUSH_VELOCITY_INITIAL = 0.0    # Push strength at training start [m/s].
CURRICULUM_PUSH_VELOCITY_FINAL = 0.80     # Push strength at training end [m/s].

CONTROLLER_SETTINGS = {
    "p_base2still": 0.05,                  # Initial command mix: brief pauses.
    "p_base2walk": 0.70,                   # Initial command mix: mostly walking.
    "p_base2turn": 0.25,                   # Initial command mix: allow turning from the start.
    "p_still2walk": 0.80,                  # After standing, usually ask for walking again.
    "p_still2turn": 0.20,                  # After standing, sometimes ask for a turn.
    "p_walk2still": 0.20,                  # End a walk segment with a short pause sometimes.
    "p_walk2turn": 0.80,                   # End a walk segment with a turn most of the time.
    "p_turn2still": 0.10,                  # End a turn with a pause only occasionally.
    "p_turn2walk": 0.90,                   # After turning, usually return to walking.
    "v_mean": (0.0, 0.0),                  # Center the walking command around zero for omni-directionality.
    "v_var": (0.35, 0.25),                 # Spread forward/backward and lateral commands.
    "v_steps_mean": 220,                   # Keep each walk command for a few seconds.
    "v_steps_var": 30,                     # Small variation in walk-command duration.
    "w_mean": 0.0,                         # No preferred turn direction.
    "w_var": 0.35,                         # Moderate yaw-rate variation.
    "w_radius_mean": 0.10,                 # Favor near in-place turning over wide arcs.
    "w_radius_var": 0.15,                  # Still allow some curved turns.
    "w_steps_mean": 120,                   # Keep turning commands long enough to learn them.
    "w_steps_var": 18,                     # Small variation in turn-command duration.
    "s_steps_mean": 25,                    # Pauses stay short.
    "s_steps_var": 6,                      # Small variation in pause length.
}


def create_run_paths():
    run_dir = TRAINING_DIR / "data" / f"{RUN_NAME}_robust"
    return {
        "run_dir": run_dir,
        "log_dir": run_dir / "logs",
        "step_checkpoint_dir": run_dir / "step_checkpoints",
        "timed_checkpoint_dir": run_dir / "timed_checkpoints",
        "final_snapshot_path": run_dir / f"ppo_{RUN_NAME}.yaml",
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


def resolve_model_stem(paths):
    if MODEL_STEM is not None:
        return Path(MODEL_STEM)

    candidates = list(paths["timed_checkpoint_dir"].glob("*.yaml"))
    candidates.extend(paths["step_checkpoint_dir"].glob("*.yaml"))

    if paths["final_snapshot_path"].exists():
        candidates.append(paths["final_snapshot_path"])

    if not candidates:
        raise FileNotFoundError(
            f"No snapshot YAML found in {paths['run_dir']}. "
            "Set MODEL_STEM explicitly."
        )

    latest_snapshot = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest_snapshot.with_suffix("")


def curriculum_factor_from_step(step, total_timesteps, warmup_ratio=0.05, schedule="linear"):
    if total_timesteps <= 0:
        return 0.0

    progress = min(1.0, step / total_timesteps)
    if progress < warmup_ratio:
        return 0.0

    adjusted_progress = (progress - warmup_ratio) / (1.0 - warmup_ratio)
    if schedule == "exponential":
        return adjusted_progress ** 2
    return adjusted_progress


def interpolate(initial, final, factor):
    return initial + factor * (final - initial)


def spawn_noise_terrain(env, z_max, seed=None):
    if seed is None:
        seed = int(np.random.randint(0, 1_000_000))

    heightmap = Heightmap.from_noise(
        x=TERRAIN_SIZE,
        y=TERRAIN_SIZE,
        z_max=z_max,
        seed=seed,
    )
    env._backend.spawn_terrain(
        heightmap_data=heightmap.data,
        scale=list(TERRAIN_SCALE),
        origin=list(TERRAIN_ORIGIN),
    )
    return int(seed), float(z_max)


def attach_noise_callbacks(cfg, env, motor_noise_std, sensor_noise_scale):
    if motor_noise_std > 0.0:
        motor_noise_callback = MotorNoiseCurriculumCallback(
            config=cfg,
            env=env,
            current_factor=1.0,
            current_noise_std=motor_noise_std,
            verbose=False,
        )
        motor_noise_callback.apply_saved_state()

    if sensor_noise_scale > 0.0:
        sensor_noise_callback = SensorNoiseCurriculumCallback(
            config=cfg,
            env=env,
            current_factor=1.0,
            current_noise_scale=sensor_noise_scale,
            verbose=False,
        )
        sensor_noise_callback.apply_saved_state()


def create_force_callback(cfg, env, push_velocity):
    if push_velocity <= 0.0:
        return None

    return ForceCurriculumCallback(
        config=cfg,
        env=env,
        push_vel_initial=push_velocity,
        push_vel_final=push_velocity,
        push_interval_s=PUSH_INTERVAL_S,
        push_duration_steps=PUSH_DURATION_STEPS,
        current_factor=1.0,
        current_push_vel=push_velocity,
        verbose=False,
    )


def run_policy_rollout(model, env, force_callback):
    episode_rewards = []

    for episode_index in range(N_EVAL_EPISODES):
        observation, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            if force_callback is not None:
                force_callback.step_saved_state(env)

            action, _ = model.predict(observation, deterministic=DETERMINISTIC)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if USE_GUI and REALTIME_SLEEP > 0.0:
                time.sleep(REALTIME_SLEEP)

        episode_rewards.append(float(total_reward))
        print(f"Episode {episode_index + 1}: reward = {total_reward:.2f}")

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def main():
    paths = create_run_paths()
    model_path = resolve_model_stem(paths)

    cfg = Config()
    env = create_env(cfg)

    try:
        model = PPO.load(str(model_path), env=env, device=DEVICE)
        model.set_logger(configure(str(paths["log_dir"]), ["stdout", "csv", "tensorboard"]))

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
                seed=MANUAL_TERRAIN_SEED,
            )
        elif TERRAIN_MODE == "random":
            terrain_seed, terrain_z_max = spawn_noise_terrain(
                env,
                z_max=MANUAL_TERRAIN_Z_MAX,
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
        attach_noise_callbacks(cfg, env, motor_noise_std, sensor_noise_scale)
        force_callback = create_force_callback(cfg, env, push_velocity)

        print(f"Testing model: {model_path}.zip")
        print(f"Terrain mode: {TERRAIN_MODE}")
        print(f"Terrain seed: {terrain_seed}")
        print(f"Terrain z_max: {terrain_z_max}")
        print(f"Ground friction: {friction_value}")
        print(f"Motor noise std: {motor_noise_std}")
        print(f"Sensor noise scale: {sensor_noise_scale}")
        print(f"Push velocity: {push_velocity}")

        run_policy_rollout(model, env, force_callback)
    finally:
        env.close()


if __name__ == "__main__":
    main()
