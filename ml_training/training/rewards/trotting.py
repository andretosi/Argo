"""Trotting reward: walk at the commanded velocity, starting from a standing policy.

Some weights change over training: they start at ``<name>`` and move by ``<name>_change``
as ``fade_in`` rises from 0 to 1 after ``<name>_fade_start`` environment steps.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from training.rewards.base import Reward, RewardConfig, normalized_effort

if TYPE_CHECKING:
    from spotmicro.env import SpotmicroEnv


def fade_in(current_step: int, start: int, scale: float = 2.0) -> float:
    """0 before ``start``, then rises towards 1 (time constant: 1M steps / scale).

    Returns a NumPy float64 on purpose: multiplied with float32 terms it keeps them in
    float64 (a plain Python float would demote them to float32 under NumPy 2 promotion).
    """
    return np.clip(1.0 - np.exp(-scale * (current_step - start) / 1_000_000), 0.0, 1.0)


class TrottingReward(Reward):
    name = "trotting"

    @dataclass
    class Params(RewardConfig):
        linear_vel: float = 12.5
        linear_vel_change: float = -2.5
        linear_vel_fade_start: int = 8_000_000
        linear_vel_error_gain: float = 1.75
        linear_vel_min: float = -1.0
        height: float = -3.0
        stabilization: float = -3.0
        action_rate: float = -1.0
        action_rate_change: float = -1.5
        action_rate_fade_start: int = 5_000_000
        drift: float = -2.0
        angular_vel: float = -1.5
        effort: float = -0.5
        effort_change: float = -2.0
        effort_fade_start: int = 8_000_000
        deviation: float = -0.5

    params: Params

    def compute(self, env: "SpotmicroEnv", action: np.ndarray) -> dict[str, float]:
        p = self.params
        agent = env.agent
        roll, pitch, _ = agent.state.roll_pitch_yaw

        # Commanded velocities (robot frame), from the command device
        vx, vy, w = agent.command_input.as_array
        target_lin_velocity = np.array([vx, vy, 0.0])
        target_ang_velocity = np.array([0.0, 0.0, w])

        lin_vel_error = np.linalg.norm(target_lin_velocity - agent.state.linear_velocity) ** 2
        ang_vel_error = (
            np.linalg.norm(
                (target_ang_velocity - agent.state.angular_velocity) / agent.max_angular_velocity
            )
            ** 2
        )
        deviation_penalty = (
            np.linalg.norm(agent.state.joint_positions - agent.homing_positions) ** 2
        )
        height_penalty = (agent.state.base_position[2] - env.target_body_to_feet_height) ** 2
        stabilization_penalty = roll**2 + pitch**2
        # Velocity perpendicular to the commanded direction (all of it when the command is zero)
        target_norm_sq = np.dot(target_lin_velocity, target_lin_velocity)
        if target_norm_sq < 1e-12:
            perp_velocity = agent.state.linear_velocity
        else:
            perp_velocity = (
                agent.state.linear_velocity
                - (np.dot(agent.state.linear_velocity, target_lin_velocity) / target_norm_sq)
                * target_lin_velocity
            )
        action_rate = np.mean(action - agent.previous_action) ** 2

        lin_vel_reward = max(1 - p.linear_vel_error_gain * lin_vel_error, p.linear_vel_min)
        drift_penalty = np.linalg.norm(perp_velocity) ** 2
        steps = env.num_steps

        reward_dict = {
            "linear_vel_reward": (
                p.linear_vel + p.linear_vel_change * fade_in(steps, p.linear_vel_fade_start)
            )
            * lin_vel_reward,
            "height_penalty": p.height * min(height_penalty, 1.0),
            "stabilization_penalty": p.stabilization * min(stabilization_penalty, 1.0),
            "action_rate_penalty": (
                p.action_rate + p.action_rate_change * fade_in(steps, p.action_rate_fade_start)
            )
            * action_rate,
            "drift_penalty": p.drift * drift_penalty,
            "angular_vel_penalty": p.angular_vel * ang_vel_error,
            "effort_penalty": (p.effort + p.effort_change * fade_in(steps, p.effort_fade_start))
            * normalized_effort(env),
            "deviation_penalty": p.deviation * deviation_penalty,
        }
        return reward_dict
