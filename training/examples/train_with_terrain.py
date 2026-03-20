"""
Training with Dynamic Terrain Changes
======================================

This example shows how to train a walking policy with terrain that changes
periodically during training. This helps the robot generalize to different
terrain types.

The key feature of Fucina is that the SAME CODE works with both PyBullet
and MuJoCo backends - just change the `engine` parameter!

Usage:
------
    # Train with PyBullet (fast, good for development)
    python train_with_terrain.py --engine pybullet --episodes 100000

    # Train with MuJoCo (more accurate physics)
    python train_with_terrain.py --engine mujoco --episodes 100000

    # With GUI for debugging
    python train_with_terrain.py --engine pybullet --gui --episodes 1000
"""

import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.env_checker import check_env

# Fucina imports
from spotmicro.physics.factory import create_backend
from spotmicro.env.spotmicro_env import SpotmicroEnv
from spotmicro.tools.config import Config
from spotmicro.tools.TerrainTools import Heightmap
from spotmicro.devices.random_controller import RandomController

# Training callbacks
from training.callbacks.legacy_terrain import TerrainChangeCallback, CurriculumTerrainCallback


# ============================================================================
# REWARD FUNCTION
# ============================================================================
# Simple reward function for walking - customize as needed

class RewardState:
    """Tracks state between reward computations."""
    def __init__(self):
        self.previous_position = None

    def populate(self, env):
        self.previous_position = env.agent.state.base_position.copy()


def reward_function(env, action) -> tuple[float, dict]:
    """
    Reward function that encourages forward walking.

    Returns:
        tuple: (reward, info_dict)
    """
    agent = env.agent
    state = agent.state

    # Forward velocity reward (x direction)
    forward_vel = state.linear_velocity[0]
    forward_reward = forward_vel * 2.0

    # Stability penalty (minimize roll/pitch)
    roll, pitch, _ = state.roll_pitch_yaw
    stability_penalty = -0.5 * (abs(roll) + abs(pitch))

    # Energy penalty (minimize joint velocities)
    joint_vel_penalty = -0.01 * sum(abs(v) for v in state.joint_velocities)

    # Total reward
    reward = forward_reward + stability_penalty + joint_vel_penalty

    # Info dict for logging
    info = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty,
        "joint_vel_penalty": joint_vel_penalty,
    }

    return reward, info


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train walking policy with dynamic terrain")

    # Engine selection (THE KILLER FEATURE!)
    parser.add_argument(
        "--engine",
        type=str,
        choices=["pybullet", "mujoco"],
        default="pybullet",
        help="Physics engine to use (pybullet or mujoco)"
    )

    # Training parameters
    parser.add_argument("--total-steps", type=int, default=5_000_000, help="Total training steps")
    parser.add_argument("--terrain-change-episodes", type=int, default=50,
                        help="Change terrain every N episodes")

    # Terrain parameters
    parser.add_argument("--terrain-type", type=str, choices=["fixed", "random", "curriculum"],
                        default="random", help="Terrain variation strategy")
    parser.add_argument("--terrain-difficulty", type=float, default=0.3,
                        help="Max terrain height (z_max)")

    # Other options
    parser.add_argument("--gui", action="store_true", help="Show GUI (slower)")
    parser.add_argument("--run-name", type=str, default="terrain_walk", help="Name for this run")

    return parser.parse_args()


def main():
    args = parse_args()

    # ========================================================================
    # SETUP DIRECTORIES
    # ========================================================================
    run_dir = Path(f"runs/{args.run_name}_{args.engine}")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"

    print(f"=" * 60)
    print(f"Training with Fucina")
    print(f"=" * 60)
    print(f"Engine:      {args.engine}")
    print(f"Terrain:     {args.terrain_type}")
    print(f"Total steps: {args.total_steps:,}")
    print(f"Output:      {run_dir}")
    print(f"=" * 60)

    # ========================================================================
    # CREATE BACKEND (PyBullet or MuJoCo - same API!)
    # ========================================================================
    backend = create_backend(
        engine=args.engine,
        use_gui=args.gui,
        sim_frequency=240
    )

    # ========================================================================
    # CREATE ENVIRONMENT
    # ========================================================================
    config = Config()  # Empty config, use defaults
    device = RandomController(config)

    env = SpotmicroEnv(
        backend=backend,
        device=device,
        config=config,
        reward_fn=reward_function,
        reward_state=RewardState(),
        use_gui=args.gui,
    )

    # Validate environment
    check_env(env, warn=True)

    # ========================================================================
    # SETUP TERRAIN CALLBACK
    # ========================================================================
    if args.terrain_type == "fixed":
        # No terrain changes
        terrain_callback = None
        print("[Terrain] Using fixed flat terrain")

    elif args.terrain_type == "random":
        # Random terrain changes every N episodes
        terrain_callback = TerrainChangeCallback(
            env=env,
            change_every_n_episodes=args.terrain_change_episodes,
            terrain_generator=Heightmap.from_noise,
            generator_kwargs={
                "size": 256,
                "z_max": args.terrain_difficulty,
            },
            scale=[0.02, 0.02, 1.0],  # Scale for physics
            origin=[0.0, 0.0, 0.0],
            verbose=True,
        )
        print(f"[Terrain] Random terrain, changing every {args.terrain_change_episodes} episodes")
        print(f"[Terrain] Max height: {args.terrain_difficulty}m")

    elif args.terrain_type == "curriculum":
        # Gradually increasing difficulty
        terrain_callback = CurriculumTerrainCallback(
            env=env,
            change_every_n_episodes=args.terrain_change_episodes,
            initial_difficulty=0.1,  # Start with 10% of max height
            final_difficulty=1.0,    # End with 100% of max height
            difficulty_schedule="linear",
            total_episodes_estimate=args.total_steps // 1000,  # Rough estimate
            terrain_generator=Heightmap.from_noise,
            generator_kwargs={
                "size": 256,
                "z_max": args.terrain_difficulty,
            },
            verbose=True,
        )
        print(f"[Terrain] Curriculum terrain, difficulty 10% -> 100%")

    # ========================================================================
    # SETUP CALLBACKS
    # ========================================================================
    checkpoint_callback = CheckpointCallback(
        save_freq=args.total_steps // 10,
        save_path=str(checkpoint_dir),
        name_prefix=f"ppo_{args.run_name}"
    )

    # Combine callbacks
    callbacks = [checkpoint_callback]
    if terrain_callback is not None:
        callbacks.append(terrain_callback)

    callback_list = CallbackList(callbacks)

    # ========================================================================
    # CREATE AND TRAIN MODEL
    # ========================================================================
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        ent_coef=0.002,
        clip_range=0.1,
        tensorboard_log=str(log_dir),
    )

    print(f"\nStarting training...")
    model.learn(
        total_timesteps=args.total_steps,
        callback=callback_list,
    )

    # ========================================================================
    # SAVE FINAL MODEL
    # ========================================================================
    final_path = run_dir / f"ppo_{args.run_name}_final"
    model.save(str(final_path))
    print(f"\nModel saved to: {final_path}")

    # Cleanup
    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
