"""Robust walking reward, legged_gym style.

Based on ETH RSL legged_gym (https://github.com/leggedrobotics/legged_gym): exponential
velocity tracking, feet air time, and penalties on body motion, orientation, height and
action smoothness. Positive weights are rewards, negative weights are penalties.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from training.rewards.base import Reward, RewardConfig

if TYPE_CHECKING:
    from spotmicro.env import SpotmicroEnv


class RobustWalkingReward(Reward):
    """Track the commanded velocity while staying stable and smooth."""

    name = "robust_walking"

    @dataclass
    class Params(RewardConfig):
        # === Tracking rewards ===
        tracking_lin_vel: float = 1.0  # Track commanded linear velocity
        tracking_ang_vel: float = 0.5  # Track commanded angular velocity
        tracking_sigma: float = 0.25  # Exponential tracking sharpness

        # === Gait rewards ===
        feet_air_time: float = 1.0  # Reward feet spending time in air
        feet_air_time_threshold: float = 0.5  # Seconds threshold for full reward

        # === Base motion penalties ===
        lin_vel_z: float = -2.0  # Penalize vertical body velocity
        ang_vel_xy: float = -0.05  # Penalize roll/pitch angular velocity

        # === Orientation penalties ===
        orientation: float = -1.0  # Penalize non-flat orientation
        base_height: float = -1.0  # Penalize deviation from target height
        target_height: float = 0.2  # Target body-to-feet height [m]

        # === Smoothness penalties ===
        action_rate: float = -0.01  # Penalize action changes
        torques: float = -0.00001  # Penalize torque usage
        dof_acc: float = -2.5e-7  # Penalize joint acceleration

        # === Safety penalties ===
        dof_pos_limits: float = -0.5  # Penalize approaching joint limits

        # === Energy efficiency ===
        power: float = -0.0001  # Penalize mechanical power

        # === Survival ===
        alive_bonus: float = 0.1  # Small bonus for staying alive

    params: Params

    def __init__(self, params: Params | None = None) -> None:
        super().__init__(params)
        self.prev_joint_velocities = np.zeros(0)  # set in reset()
        self.feet_air_time = np.zeros(4)  # Time each foot has been in the air
        self.feet_contact_prev: set[int] = set()
        self.dt = 1 / 60

    def reset(self, env: "SpotmicroEnv") -> None:
        self.prev_joint_velocities = env.agent.state.joint_velocities.copy()
        self.feet_air_time = np.zeros(4)
        self.feet_contact_prev = env.agent.state.feet_contacts.copy()
        self.dt = 1.0 / env.control_frequency
        joints = env.agent.motor_joints
        self._foot_ids = [j.link_id for j in joints if j.type == "foot"]
        low = np.array([j.limits[0] for j in joints])
        high = np.array([j.limits[1] for j in joints])
        margin = 0.1 * (high - low)
        self._soft_low, self._soft_high = low + margin, high - margin

    def compute(self, env: "SpotmicroEnv", action: np.ndarray) -> dict[str, float]:
        p = self.params
        agent = env.agent
        reward_dict: dict[str, float] = {}

        lin_vel = agent.state.linear_velocity  # robot frame
        ang_vel = agent.state.angular_velocity  # robot frame
        roll, pitch, _ = agent.state.roll_pitch_yaw
        joint_pos = agent.state.joint_positions
        joint_vel = agent.state.joint_velocities
        feet_contacts = agent.state.feet_contacts
        cmd_vx, cmd_vy, cmd_wz = tuple(agent.command_input.as_array)

        # === Tracking rewards (exponential, legged_gym style), in the robot frame ===
        cmd_body_xy = np.array([cmd_vx, cmd_vy], dtype=np.float32)
        lin_vel_error = np.sum((lin_vel[:2] - cmd_body_xy) ** 2)
        reward_dict["tracking_lin_vel"] = p.tracking_lin_vel * np.exp(
            -lin_vel_error / p.tracking_sigma
        )
        ang_vel_error = (ang_vel[2] - cmd_wz) ** 2
        reward_dict["tracking_ang_vel"] = p.tracking_ang_vel * np.exp(
            -ang_vel_error / p.tracking_sigma
        )

        # === Feet air time: rewarded when a foot lands ===
        air_time_reward = 0.0
        for i, foot_id in enumerate(self._foot_ids):
            if foot_id in feet_contacts:
                if foot_id not in self.feet_contact_prev:
                    air_time_reward += min(self.feet_air_time[i], p.feet_air_time_threshold)
                self.feet_air_time[i] = 0
            else:
                self.feet_air_time[i] += self.dt
        reward_dict["feet_air_time"] = p.feet_air_time * air_time_reward

        # === Base motion and orientation penalties ===
        reward_dict["lin_vel_z"] = p.lin_vel_z * lin_vel[2] ** 2
        reward_dict["ang_vel_xy"] = p.ang_vel_xy * np.sum(ang_vel[:2] ** 2)
        reward_dict["orientation"] = p.orientation * (roll**2 + pitch**2)
        body_height = agent.get_body_to_feet_height_projected()
        height_error = (body_height - p.target_height) ** 2
        reward_dict["base_height"] = p.base_height * min(height_error, 1.0)

        # === Smoothness penalties ===
        action_diff = action - agent.previous_action
        reward_dict["action_rate"] = p.action_rate * np.sum(action_diff**2)
        torques = agent.state.joint_efforts
        reward_dict["torques"] = p.torques * np.sum(torques**2)
        joint_acc = (joint_vel - self.prev_joint_velocities) / self.dt
        reward_dict["dof_acc"] = p.dof_acc * np.sum(joint_acc**2)

        # === Energy penalty (power = torque * velocity) ===
        reward_dict["power"] = p.power * np.sum(np.abs(torques * joint_vel))

        # === Joint limit penalties (within 10% of the range) ===
        below = np.maximum(self._soft_low - joint_pos, 0.0)
        above = np.maximum(joint_pos - self._soft_high, 0.0)
        limit_penalty = float(np.sum(below**2 + above**2))
        reward_dict["dof_pos_limits"] = p.dof_pos_limits * limit_penalty

        reward_dict["alive_bonus"] = p.alive_bonus

        self.prev_joint_velocities = joint_vel.copy()
        self.feet_contact_prev = feet_contacts.copy()

        return reward_dict
