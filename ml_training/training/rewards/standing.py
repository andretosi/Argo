"""Standing reward: hold every joint at its homing position, with little effort."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from training.rewards.base import Reward, RewardConfig, normalized_effort

if TYPE_CHECKING:
    from spotmicro.env import SpotmicroEnv


class StandingReward(Reward):
    """Per joint, a downward parabola centred on the homing position, averaged."""

    name = "standing"

    @dataclass
    class Params(RewardConfig):
        position_weight: float = 2.0
        effort_weight: float = -1.0
        parabola_steepness: float = -5.0  # curvature of the per-joint parabola
        position_reward_min: float = -0.5  # per-joint reward is clipped to [min, 1]

    params: Params

    def __init__(self, params: Params | None = None) -> None:
        super().__init__(params)
        self.homing_positions: np.ndarray | None = None

    def reset(self, env: "SpotmicroEnv") -> None:
        self.homing_positions = np.array([float(j.homing_position) for j in env.agent.motor_joints])

    def compute(self, env: "SpotmicroEnv", action: np.ndarray) -> dict[str, float]:
        p = self.params
        diffs = env.agent.state.joint_positions - self.homing_positions
        rewards = np.clip(p.parabola_steepness * np.square(diffs) + 1.0, p.position_reward_min, 1.0)
        reward_dict = {
            "action_reward": p.position_weight * np.mean(rewards),
            "effort_penalty": p.effort_weight * normalized_effort(env),
        }
        return reward_dict
