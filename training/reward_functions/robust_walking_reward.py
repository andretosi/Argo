"""
Robust Walking Reward Function
==============================

Implements legged_gym-style reward components with proper weighting
for sim-to-real transfer.

Based on:
- ETH RSL legged_gym (https://github.com/leggedrobotics/legged_gym)
- NVIDIA Isaac Gym examples

Key components:
- Velocity tracking (exponential reward)
- Feet air time bonus
- Orientation penalties
- Smoothness penalties (action rate, torques, joint acceleration)

Usage:
    from training.reward_functions.robust_walking_reward import (
        reward_function, RewardState, RewardConfig
    )

    config = RewardConfig(tracking_sigma=0.25)
    state = RewardState(config)

    env = SpotmicroEnv(..., reward_fn=reward_function, reward_state=state)
"""

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spotmicro.env.spotmicro_env import SpotmicroEnv


@dataclass
class RewardConfig:
    """
    Configuration for reward function weights and parameters.

    Weights follow legged_gym defaults with adjustments for SpotMicro.
    Positive weights = rewards, negative weights = penalties.
    """
    # === Tracking rewards ===
    tracking_lin_vel: float = 1.0         # Track commanded linear velocity
    tracking_ang_vel: float = 0.5         # Track commanded angular velocity
    tracking_sigma: float = 0.25          # Exponential tracking sharpness

    # === Gait rewards ===
    feet_air_time: float = 1.0            # Reward feet spending time in air
    feet_air_time_threshold: float = 0.5  # Seconds threshold for full reward

    # === Base motion penalties ===
    lin_vel_z: float = -2.0               # Penalize vertical body velocity
    ang_vel_xy: float = -0.05             # Penalize roll/pitch angular velocity

    # === Orientation penalties ===
    orientation: float = -1.0             # Penalize non-flat orientation
    base_height: float = -1.0             # Penalize deviation from target height
    target_height: float = 0.2            # Target body-to-feet height [m]

    # === Smoothness penalties ===
    action_rate: float = -0.01            # Penalize action changes
    torques: float = -0.00001             # Penalize torque usage
    dof_acc: float = -2.5e-7              # Penalize joint acceleration

    # === Safety penalties ===
    dof_pos_limits: float = -0.5          # Penalize approaching joint limits

    # === Energy efficiency ===
    power: float = -0.0001                # Penalize mechanical power

    # === Survival ===
    alive_bonus: float = 0.1              # Small bonus for staying alive


class RewardState:
    """
    Maintains state between timesteps for reward calculation.

    Tracks:
    - Previous joint velocities (for acceleration penalty)
    - Feet contact history (for air time calculation)
    - Previous base position
    """

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()

        # State variables
        self.prev_joint_velocities: np.ndarray | None = None
        self.prev_base_position: np.ndarray | None = None
        self.feet_air_time: np.ndarray = np.zeros(4)  # Time each foot in air
        self.feet_contact_prev: set = set()
        self.dt: float = 1/60  # Will be updated from env

    def populate(self, env: "SpotmicroEnv") -> None:
        """
        Initialize state from environment on reset.

        Called automatically by SpotmicroEnv.reset().
        """
        # Get joint velocities - may be empty on first call before physics settles
        joint_vel = env.agent.state.joint_velocities
        n_joints = len(env.agent.motor_joints)
        if len(joint_vel) != n_joints:
            joint_vel = np.zeros(n_joints)
        self.prev_joint_velocities = joint_vel.copy()
        self.prev_base_position = env.agent.state.base_position.copy()
        self.feet_air_time = np.zeros(4)
        self.feet_contact_prev = env.agent.state.feet_contacts.copy()
        self.dt = 1.0 / env.control_frequnecy


def reward_function(env: "SpotmicroEnv", action: np.ndarray) -> tuple[float, dict]:
    """
    Compute reward using legged_gym-style components.

    Parameters
    ----------
    env : SpotmicroEnv
        The environment instance
    action : np.ndarray
        The action taken (normalized joint positions)

    Returns
    -------
    tuple[float, dict]
        Total reward and dictionary of individual components
    """
    state = env.reward_state
    config = state.config
    agent = env.agent

    reward_dict = {}

    # === Get current state ===
    base_pos = agent.state.base_position
    lin_vel = agent.state.linear_velocity      # In body frame
    ang_vel = agent.state.angular_velocity     # In body frame
    roll, pitch, _ = agent.state.roll_pitch_yaw
    joint_pos = agent.state.joint_positions
    joint_vel = agent.state.joint_velocities
    feet_contacts = agent.state.feet_contacts

    # Get commands
    cmd_vx, cmd_vy, cmd_wz = tuple(agent.controller.input.as_array)

    # === Tracking rewards (exponential - legged_gym style) ===
    # Linear velocity tracking in the robot/body frame
    cmd_body_xy = np.array([cmd_vx, cmd_vy], dtype=np.float32)
    lin_vel_error = np.sum((lin_vel[:2] - cmd_body_xy) ** 2)
    reward_dict["tracking_lin_vel"] = config.tracking_lin_vel * np.exp(
        -lin_vel_error / config.tracking_sigma
    )

    # Angular velocity tracking (yaw rate)
    ang_vel_error = (ang_vel[2] - cmd_wz) ** 2
    reward_dict["tracking_ang_vel"] = config.tracking_ang_vel * np.exp(
        -ang_vel_error / config.tracking_sigma
    )

    # === Feet air time reward ===
    # Get foot link IDs (feet are the last joints in each leg)
    foot_joints = [j for j in agent.motor_joints if j.type == "foot"]
    foot_ids = [j.link_id for j in foot_joints]

    # Update air time and calculate reward
    air_time_reward = 0.0
    for i, foot_id in enumerate(foot_ids):
        if foot_id in feet_contacts:
            # Foot is on ground - give reward based on accumulated air time
            if foot_id not in state.feet_contact_prev:
                # Just landed - reward for air time
                air_time_bonus = min(state.feet_air_time[i], config.feet_air_time_threshold)
                air_time_reward += air_time_bonus
            state.feet_air_time[i] = 0
        else:
            # Foot is in air - accumulate time
            state.feet_air_time[i] += state.dt

    reward_dict["feet_air_time"] = config.feet_air_time * air_time_reward

    # === Base motion penalties ===
    # Vertical velocity penalty
    reward_dict["lin_vel_z"] = config.lin_vel_z * lin_vel[2] ** 2

    # Roll/pitch angular velocity penalty
    reward_dict["ang_vel_xy"] = config.ang_vel_xy * np.sum(ang_vel[:2] ** 2)

    # === Orientation penalties ===
    # Penalize non-flat orientation
    reward_dict["orientation"] = config.orientation * (roll ** 2 + pitch ** 2)

    # Base height penalty
    body_height = agent.get_body_to_feet_height_projected()
    height_error = (body_height - config.target_height) ** 2
    reward_dict["base_height"] = config.base_height * min(height_error, 1.0)

    # === Smoothness penalties ===
    # Action rate (change in action)
    action_diff = action - agent.previous_action
    reward_dict["action_rate"] = config.action_rate * np.sum(action_diff ** 2)

    # Torque penalty
    torques = np.array([j.effort for j in agent.motor_joints])
    reward_dict["torques"] = config.torques * np.sum(torques ** 2)

    # Joint acceleration penalty
    if state.prev_joint_velocities is not None:
        joint_acc = (joint_vel - state.prev_joint_velocities) / state.dt
        reward_dict["dof_acc"] = config.dof_acc * np.sum(joint_acc ** 2)
    else:
        reward_dict["dof_acc"] = 0.0

    # === Energy penalty (power = torque * velocity) ===
    power = np.sum(np.abs(torques * joint_vel))
    reward_dict["power"] = config.power * power

    # === Joint limit penalties ===
    limit_penalty = 0.0
    for i, joint in enumerate(agent.motor_joints):
        low, high = joint.limits
        margin = 0.1 * (high - low)  # 10% margin
        if joint_pos[i] < low + margin:
            limit_penalty += (low + margin - joint_pos[i]) ** 2
        elif joint_pos[i] > high - margin:
            limit_penalty += (joint_pos[i] - (high - margin)) ** 2
    reward_dict["dof_pos_limits"] = config.dof_pos_limits * limit_penalty

    # === Alive bonus ===
    reward_dict["alive_bonus"] = config.alive_bonus

    # === Update state for next step ===
    state.prev_joint_velocities = joint_vel.copy()
    state.prev_base_position = base_pos.copy()
    state.feet_contact_prev = feet_contacts.copy()

    # === Compute total reward ===
    total_reward = sum(reward_dict.values())

    # Log components for monitoring
    env.log_rewards(reward_dict)

    return total_reward, reward_dict
