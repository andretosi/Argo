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
# Snapshot Test Configuration
# Edit these values directly in the file, then run the script from your editor.
# ============================================================================

RUN_NAME = "robust_walk_v1"        # Run folder to inspect.
SNAPSHOT_MODEL_STEM = None         # Explicit model path without .zip, or None for latest snapshot.

BACKEND = "pybullet"               # Physics engine used for the replay.
USE_GUI = True                     # True to visualize the saved snapshot.
DEVICE = "cpu"                     # PPO device for policy inference.

MAX_EPISODE_LEN = 3000             # Max control steps per evaluation episode.
SIM_FREQUENCY = 240                # Physics frequency in Hz.
CONTROL_FREQUENCY = 60             # Policy/action frequency in Hz.

N_EVAL_EPISODES = 3                # Number of rollout episodes to run.
DETERMINISTIC = True               # True for deterministic policy actions.
REALTIME_SLEEP = 1.0 / 240.0       # Extra sleep per control step when GUI is on.

CONTROLLER_SETTINGS = {
    "p_base2still": 0.05,          # Initial command mix: brief pauses.
    "p_base2walk": 0.70,           # Initial command mix: mostly walking.
    "p_base2turn": 0.25,           # Initial command mix: allow turning from the start.
    "p_still2walk": 0.80,          # After standing, usually ask for walking again.
    "p_still2turn": 0.20,          # After standing, sometimes ask for a turn.
    "p_walk2still": 0.20,          # End a walk segment with a short pause sometimes.
    "p_walk2turn": 0.80,           # End a walk segment with a turn most of the time.
    "p_turn2still": 0.10,          # End a turn with a pause only occasionally.
    "p_turn2walk": 0.90,           # After turning, usually return to walking.
    "v_mean": (0.0, 0.0),          # Center the walking command around zero for omni-directionality.
    "v_var": (0.35, 0.25),         # Spread forward/backward and lateral commands.
    "v_steps_mean": 220,           # Keep each walk command for a few seconds.
    "v_steps_var": 30,             # Small variation in walk-command duration.
    "w_mean": 0.0,                 # No preferred turn direction.
    "w_var": 0.35,                 # Moderate yaw-rate variation.
    "w_radius_mean": 0.10,         # Favor near in-place turning over wide arcs.
    "w_radius_var": 0.15,          # Still allow some curved turns.
    "w_steps_mean": 120,           # Keep turning commands long enough to learn them.
    "w_steps_var": 18,             # Small variation in turn-command duration.
    "s_steps_mean": 25,            # Pauses stay short.
    "s_steps_var": 6,              # Small variation in pause length.
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
    if SNAPSHOT_MODEL_STEM is not None:
        return Path(SNAPSHOT_MODEL_STEM)

    candidates = list(paths["timed_checkpoint_dir"].glob("*.yaml"))
    candidates.extend(paths["step_checkpoint_dir"].glob("*.yaml"))

    if paths["final_snapshot_path"].exists():
        candidates.append(paths["final_snapshot_path"])

    if not candidates:
        raise FileNotFoundError(
            f"No snapshot YAML found in {paths['run_dir']}. "
            "Set SNAPSHOT_MODEL_STEM explicitly."
        )

    latest_snapshot = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest_snapshot.with_suffix("")


def restore_snapshot(cfg, env):
    terrain_callback = TerrainCurriculumCallbackV2(config=cfg, env=env, verbose=False)
    force_callback = ForceCurriculumCallback(config=cfg, env=env, verbose=False)
    friction_callback = FrictionCurriculumCallback(config=cfg, env=env, verbose=False)
    motor_noise_callback = MotorNoiseCurriculumCallback(config=cfg, env=env, verbose=False)
    sensor_noise_callback = SensorNoiseCurriculumCallback(config=cfg, env=env, verbose=False)

    terrain_callback.apply_saved_state(env)
    friction_callback.apply_saved_state(env)
    motor_noise_callback.apply_saved_state()
    sensor_noise_callback.apply_saved_state()

    return {
        "terrain": terrain_callback,
        "force": force_callback,
        "friction": friction_callback,
        "motor": motor_noise_callback,
        "sensor": sensor_noise_callback,
    }


def print_snapshot_summary(snapshot_callbacks):
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


def run_policy_rollout(model, env, snapshot_callbacks):
    episode_rewards = []

    for episode_index in range(N_EVAL_EPISODES):
        observation, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            snapshot_callbacks["force"].step_saved_state(env)
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
    snapshot_path = model_path.with_suffix(".yaml")

    cfg = Config(str(snapshot_path))
    env = create_env(cfg)

    try:
        snapshot_callbacks = restore_snapshot(cfg, env)
        model = PPO.load(str(model_path), env=env, device=DEVICE)
        model.set_logger(configure(str(paths["log_dir"]), ["stdout", "csv", "tensorboard"]))

        print(f"Testing snapshot: {model_path}.zip")
        print_snapshot_summary(snapshot_callbacks)
        run_policy_rollout(model, env, snapshot_callbacks)
    finally:
        env.close()


if __name__ == "__main__":
    main()
