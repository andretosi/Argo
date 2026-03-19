#!/usr/bin/env python3
"""
Test Robust Training with Atomic Curriculum Callbacks
======================================================

Quick test to verify the atomic curriculum callbacks work properly.
Runs a short training session and checks that curriculum progresses.

Usage:
    python test_robust_training.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_checker import check_env

from spotmicro.env.spotmicro_env import SpotmicroEnv
from spotmicro.physics.factory import create_backend
from spotmicro.devices.random_controller import RandomController
from spotmicro.tools.config import Config

# Atomic curriculum callbacks
from training.callbacks import (
    TerrainCurriculumCallbackV2,
    ForceCurriculumCallback,
    FrictionCurriculumCallback,
    MotorNoiseCurriculumCallback,
    SensorNoiseCurriculumCallback,
)

# Reward function
from training.reward_functions.robust_walking_reward import (
    reward_function,
    RewardState,
    RewardConfig,
)


def test_robust_training(total_steps: int = 50_000, use_gui: bool = False):
    """
    Test robust training with atomic curriculum callbacks.

    Parameters
    ----------
    total_steps : int
        Total training timesteps (default: 50k for quick test)
    use_gui : bool
        Whether to show GUI
    """
    print("=" * 60)
    print("Testing Robust Training with Atomic Curriculum Callbacks")
    print("=" * 60)

    # Config
    cfg = Config()
    reward_config = RewardConfig()

    # Create environment
    print("\n[1/5] Creating environment...")
    backend = create_backend("pybullet", use_gui=use_gui)
    device = RandomController(cfg)

    env = SpotmicroEnv(
        backend=backend,
        device=device,
        config=cfg,
        reward_fn=reward_function,
        reward_state=RewardState(reward_config),
        use_gui=use_gui,
        max_episode_len=500,  # Short episodes for testing
    )

    # Verify environment
    check_env(env, warn=True)
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")

    # Create atomic curriculum callbacks
    print("\n[2/5] Creating atomic curriculum callbacks...")

    terrain_callback = TerrainCurriculumCallbackV2(
        config=cfg,
        env=env,
        total_timesteps=total_steps,
        z_max_initial=0.02,
        z_max_final=0.15,  # Lower for testing
        change_every_episodes=10,  # More frequent for testing
        verbose=True,
    )

    force_callback = ForceCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=total_steps,
        push_vel_initial=0.1,
        push_vel_final=0.8,
        push_interval_s=5.0,  # More frequent for testing
        verbose=True,
    )

    friction_callback = FrictionCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=total_steps,
        friction_initial_low=0.9,
        friction_initial_high=1.1,
        friction_final_low=0.6,
        friction_final_high=1.3,
        verbose=True,
    )

    motor_noise_callback = MotorNoiseCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=total_steps,
        noise_initial=0.0,
        noise_final=0.03,
        verbose=True,
    )

    sensor_noise_callback = SensorNoiseCurriculumCallback(
        config=cfg,
        env=env,
        total_timesteps=total_steps,
        noise_scale_initial=0.0,
        noise_scale_final=0.5,  # Moderate for testing
        verbose=True,
    )

    callbacks = CallbackList([
        terrain_callback,
        force_callback,
        friction_callback,
        motor_noise_callback,
        sensor_noise_callback,
    ])

    print("  Created 5 atomic callbacks:")
    print("    - TerrainCurriculumCallbackV2")
    print("    - ForceCurriculumCallback")
    print("    - FrictionCurriculumCallback")
    print("    - MotorNoiseCurriculumCallback")
    print("    - SensorNoiseCurriculumCallback")

    # Create model
    print("\n[3/5] Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        device='cpu',
    )
    print(f"  Policy: {model.policy}")

    # Train
    print(f"\n[4/5] Training for {total_steps:,} steps...")
    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        progress_bar=False,
    )

    # Evaluate
    print("\n[5/5] Evaluating trained model...")
    episode_rewards = []
    for ep in range(5):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        steps = 0

        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1

        episode_rewards.append(ep_reward)
        print(f"  Episode {ep + 1}: reward = {ep_reward:.2f}, steps = {steps}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\n  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Cleanup
    env.close()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"  Total training steps: {total_steps:,}")
    print(f"  Terrain changes: {terrain_callback._terrain_change_count}")
    print(f"  Push events: {force_callback._push_count}")
    print(f"  Friction randomizations: {friction_callback._friction_count}")
    print(f"  Final motor noise: {motor_noise_callback._current_noise_std:.4f}")
    print(f"  Final sensor noise scale: {sensor_noise_callback._current_scale:.2f}")
    print(f"  Mean eval reward: {mean_reward:.2f}")

    # Check curriculum progression
    success = True
    if terrain_callback._terrain_change_count == 0:
        print("  WARNING: No terrain changes occurred!")
        success = False
    if force_callback._push_count == 0:
        print("  WARNING: No push events occurred!")
        success = False
    if friction_callback._friction_count == 0:
        print("  WARNING: No friction randomizations occurred!")
        success = False

    if success:
        print("\n  All curriculum callbacks working correctly!")

    return success, mean_reward


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000, help="Total training steps")
    parser.add_argument("--gui", action="store_true", help="Show GUI")
    args = parser.parse_args()

    success, reward = test_robust_training(args.steps, args.gui)
    sys.exit(0 if success else 1)
