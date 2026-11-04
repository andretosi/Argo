"""Jumping reward: a periodic crouch-and-launch cycle with clean landings."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from training.rewards.base import Reward, RewardConfig, normalized_effort

if TYPE_CHECKING:
    from spotmicro.env import SpotmicroEnv


class JumpingReward(Reward):
    """Track a target body height that dips (crouch) then rises (launch) every ``period`` s.

    Note: the phase clock (``step_count``) is not reset between episodes, as in the original.
    """

    name = "jumping"

    @dataclass
    class Params(RewardConfig):
        # Jump cycle
        period: float = 1.5  # seconds per crouch + launch cycle
        control_dt: float = 1 / 60  # seconds per step, for the phase clock
        standing_height: float = 0.257  # default standing height [m]
        crouch_depth: float = 0.1  # [m]
        # Component weights
        height_tracking: float = 1.5
        velocity_thrust: float = 2.0
        airtime: float = 3.0
        orientation: float = 1.0
        pitch_rate_control: float = 1.0
        stability: float = 1.0
        precision: float = 0.5
        effort: float = 0.2
        # Component shapes
        height_sharpness: float = 15.0
        thrust_gain: float = 2.0  # reward per m/s of upward velocity in the launch phase
        airtime_bonus: float = 5.0
        orientation_sharpness: float = 5.0
        pitch_rate_sharpness: float = 1.0
        stability_sharpness: float = 0.1
        landing_phase: float = 0.8  # fraction of the cycle after which landing stability counts
        precision_sharpness: float = 10.0
        alive_height: float = 0.15  # body above this height is "alive"
        alive_bonus: float = 2.0
        fallen_penalty: float = -5.0

    params: Params

    def __init__(self, params: Params | None = None) -> None:
        super().__init__(params)
        self.step_count = 0
        self.home_xy = (0.0, 0.0)

    def compute(self, env: "SpotmicroEnv", action: np.ndarray) -> dict[str, float]:
        p = self.params
        state = env.agent.state

        # Phase within the jump cycle
        self.step_count += 1
        t = (self.step_count * p.control_dt) % p.period
        if t < p.period / 2:
            # Crouch: smooth dip down and back up to standing height
            z_target = p.standing_height - p.crouch_depth * (
                0.5 * (1 - np.cos(2 * np.pi * t / (p.period / 2)))
            )
        else:
            # Launch: reach for maximum extension
            z_target = p.standing_height + p.crouch_depth

        z_actual = state.base_position[2]
        reward_height = np.exp(-p.height_sharpness * (z_actual - z_target) ** 2)

        v_z = state.linear_velocity[2]
        reward_velocity = p.thrust_gain * v_z if (t > p.period / 2 and v_z > 0) else 0.0

        is_airborne = 1.0 if len(state.feet_contacts) == 0 else 0.0
        reward_airtime = p.airtime_bonus * is_airborne

        roll, pitch, _ = state.roll_pitch_yaw
        reward_orientation = np.exp(-p.orientation_sharpness * (roll**2 + pitch**2))

        effort_penalty = -1 * normalized_effort(env)

        dist_from_home = np.linalg.norm(np.array(state.base_position[:2]) - self.home_xy)
        reward_precision = np.exp(-p.precision_sharpness * dist_from_home**2)

        pitch_rate = state.angular_velocity[1]
        reward_pitch_rate = np.exp(-p.pitch_rate_sharpness * (pitch_rate**2))

        alive = p.alive_bonus if z_actual > p.alive_height else p.fallen_penalty

        reward_stability = 0.0
        if t > p.landing_phase * p.period and not is_airborne:
            # Reward low base + joint velocities (standing still after landing)
            qvel = np.concatenate(
                [state.linear_velocity, state.angular_velocity, state.joint_velocities]
            )
            reward_stability = np.exp(-p.stability_sharpness * np.linalg.norm(qvel))

        reward_dict = {
            "height_tracking": p.height_tracking * reward_height,
            "velocity_thrust": p.velocity_thrust * reward_velocity,
            "airtime": p.airtime * reward_airtime,
            "orientation": p.orientation * reward_orientation,
            "pitch_rate_control": p.pitch_rate_control * reward_pitch_rate,
            "alive_bonus": alive,
            "stability": p.stability * reward_stability,
            "precision": p.precision * reward_precision,
            "effort_penalty": p.effort * effort_penalty,
        }
        return reward_dict
