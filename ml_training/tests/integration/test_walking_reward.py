"""The walking reward on the real environment: tracking, progress direction, effort."""

import unittest

import numpy as np

from commands import FixedController
from spotmicro.env import SpotmicroEnv
from training.rewards import make_reward


def make_env(mode: str) -> SpotmicroEnv:
    env = SpotmicroEnv(FixedController(mode), make_reward({"walking": {}}))
    env.set_num_steps(10_000_000)  # progress fully faded in
    env.reset(seed=0)
    return env


def step(env: SpotmicroEnv, push: float = 0.0, steps: int = 1) -> dict[str, float]:
    """Step at the homing pose, pushing the base along the robot's heading (+: head first)."""
    for _ in range(steps):
        yaw = env.agent.state.roll_pitch_yaw[2]
        if push:
            env.sim.apply_external_force(push * np.array([np.cos(yaw), np.sin(yaw), 0.0]))
        env.step(env.agent.default_actions.astype(np.float32))
    return env.episode_reward_info[-1]


class TestWalkingReward(unittest.TestCase):
    def test_standing_still_on_a_zero_command_is_tracked(self) -> None:
        env = make_env("still")
        info = step(env, steps=60)
        self.assertGreater(info["linear_vel_reward"], 0.9 * 12.0)
        self.assertEqual(info["progress_reward"], 0.0)
        self.assertLess(info["effort_penalty"], 0.0)  # the servos hold the robot up
        env.close()

    def test_progress_has_the_sign_of_motion_along_the_command(self) -> None:
        for push, sign in ((15.0, 1), (-15.0, -1)):
            with self.subTest(push=push):
                env = make_env("walk")
                step(env, steps=30)
                info = step(env, push=push, steps=10)
                self.assertEqual(np.sign(info["progress_reward"]), sign)
                env.close()

    def test_moving_at_the_command_beats_standing(self) -> None:
        env = make_env("walk")  # 0.2 m/s forward
        standing = step(env, steps=30)["linear_vel_reward"]
        moving = max(step(env, push=8.0)["linear_vel_reward"] for _ in range(40))
        self.assertGreater(moving, standing)
        env.close()


if __name__ == "__main__":
    unittest.main()
