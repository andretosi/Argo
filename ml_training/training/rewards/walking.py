"""Walking reward: follow the velocity command, upright, smoothly and with little effort.

The command (vx, vy, w) is read in the robot frame, in m/s and rad/s, as in robust_walking.
Every term is defined for any command, including standing still (zero command).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from training.rewards.base import Reward, RewardConfig, normalized_effort

if TYPE_CHECKING:
    from spotmicro.env import SpotmicroEnv


def fade_in(current_step: int, start: int, scale: float = 2.0) -> float:
    """0 before ``start``, then rises towards 1 (time constant: 1M steps / scale)."""
    if current_step < start:
        return 0.0
    return 1.0 - np.exp(-scale * (current_step - start) / 1_000_000)


class WalkingReward(Reward):
    name = "walking"

    @dataclass
    class Params(RewardConfig):
        alive: float = 1.5  # per step, while the episode runs
        # Velocity tracking: weight * max(1 - (error / tolerance)^2, min), error in m/s
        linear_vel: float = 12.0
        linear_vel_tolerance: float = 0.2  # [m/s]: the term is 0 at this error
        linear_vel_min: float = 0.0  # below 0, a robot that can't walk yet prefers to fall
        # Progress: speed along the commanded direction [m/s], clipped, faded in over training
        progress: float = 2.0
        progress_clip: float = 0.5  # [m/s]
        progress_fade_start: int = 100_000  # [env steps]
        angular_vel: float = -0.2  # x (commanded - actual yaw rate)^2, in (rad/s)^2
        action_rate: float = -2.0  # x mean squared action change
        height: float = -3.0  # x (body-to-feet height - target)^2, in m^2
        target_height: float = 0.21  # [m]
        stabilization: float = -3.0  # x (roll^2 + pitch^2)
        effort: float = -1.5  # x mean squared torque, as a fraction of max torque
        deviation: float = 0.0  # x squared distance of the joints from the homing pose
        vertical_motion: float = -0.5  # x vertical speed^2

    params: Params

    def __init__(self, params: Params | None = None) -> None:
        super().__init__(params)
        self.prev_base_position = np.zeros(3)

    def reset(self, env: "SpotmicroEnv") -> None:
        self.prev_base_position = env.agent.state.base_position.copy()

    def compute(self, env: "SpotmicroEnv", action: np.ndarray) -> dict[str, float]:
        p = self.params
        agent = env.agent
        state = agent.state
        cmd_vx, cmd_vy, cmd_w = agent.command_input.as_array
        cmd_xy = np.array([cmd_vx, cmd_vy])
        roll, pitch, yaw = state.roll_pitch_yaw

        lin_vel_error = np.linalg.norm(cmd_xy - state.linear_velocity[:2])
        lin_vel_reward = max(1.0 - (lin_vel_error / p.linear_vel_tolerance) ** 2, p.linear_vel_min)

        position = state.base_position
        velocity = (position[:2] - self.prev_base_position[:2]) * env.control_frequency
        self.prev_base_position = position.copy()
        cmd_speed = np.linalg.norm(cmd_xy)
        if cmd_speed > 1e-6:
            c, s = np.cos(yaw), np.sin(yaw)
            direction = np.array([c * cmd_vx - s * cmd_vy, s * cmd_vx + c * cmd_vy]) / cmd_speed
            progress = float(np.clip(velocity @ direction, -p.progress_clip, p.progress_clip))
        else:
            progress = 0.0

        ang_vel_error = (cmd_w - state.angular_velocity[2]) ** 2
        height_error = (agent.get_body_to_feet_height_projected() - p.target_height) ** 2
        action_rate = np.mean((action - agent.previous_action) ** 2)
        deviation = np.sum((state.joint_positions - agent.homing_positions) ** 2)

        reward_dict = {
            "alive": p.alive,
            "linear_vel_reward": p.linear_vel * lin_vel_reward,
            "progress_reward": p.progress
            * fade_in(env.num_steps, p.progress_fade_start)
            * progress,
            "angular_vel_penalty": p.angular_vel * ang_vel_error,
            "action_rate_penalty": p.action_rate * action_rate,
            "height_penalty": p.height * min(height_error, 1.0),
            "stabilization_penalty": p.stabilization * min(roll**2 + pitch**2, 1.0),
            "effort_penalty": p.effort * normalized_effort(env),
            "deviation_penalty": p.deviation * deviation,
            "vertical_motion_penalty": p.vertical_motion * state.linear_velocity[2] ** 2,
        }
        return reward_dict
