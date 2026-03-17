import numpy as np
from spotmicro.env.spotmicro_env import SpotmicroEnv
from pybullet import getEulerFromQuaternion

class RewardState:
    def __init__(self):
        self.prev_contacts = set()
        self.a = -5.0   # controls steepness of parabola
        self.step_count = 0

        # will hold homing positions in raw joint space
        self.homing_positions = None
    
    def populate(self, env: SpotmicroEnv):
        # Store raw homing positions for each motor joint
        self.homing_positions = np.array([
            float(j.homing_position) for j in env.agent.motor_joints
        ])
        return
    
    def increment_step(self):
        self.step_count += 1

    def get_sim_time(self, dt=1/60.):
        # Returns the elapsed "physics time" in seconds
        return self.step_count * dt

import numpy as np
import time

def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:
    # 1. Timing and Phase Constants
    # Adjust T to control how fast the "jump" cycle is (e.g., 1.0 second)
    T = 1.0 
    env.reward_state.increment_step()
    t = env.reward_state.get_sim_time() % T
    
    z_standing = 0.257  # Default standing height for SpotMicro
    crouch_depth = 0.1 # How deep to go
    
    # 2. Define Target Trajectory (Sinusoidal Nudge)
    # This creates a smooth "U" shape for the crouch and a "Launch" phase
    if t < T/2:
        # Crouch Phase: Smoothly descend and ascend back to standing
        # Using 1 - cos creates a smooth dip
        z_target = z_standing - crouch_depth * (0.5 * (1 - np.cos(2 * np.pi * t / (T/2))))
    else:
        # Flight/Extension Phase: Nudge it to reach for the stars
        # We set target high to encourage max extension/jump
        z_target = z_standing + crouch_depth 

    # 3. State Extraction
    current_pos = env.agent.state.base_position
    z_actual = current_pos[2]
    
    # 4. Reward Components
    # Height Tracking: Using a Gaussian-style reward for the nudge
    reward_height = np.exp(-15.0 * (z_actual - z_target)**2)
    
    # Upward Velocity: Big reward for moving UP during the second half of the phase
    v_z = env.agent.state.linear_velocity[2]
    reward_velocity = 0.0
    if t > T/2 and v_z > 0:
        reward_velocity = 2.0 * v_z 

    # Airtime: Reward for having no feet on the ground
    contacts = env.agent.state.feet_contacts
    is_airborne = 1.0 if sum(contacts) == 0 else 0.0
    reward_airtime = 5.0 * is_airborne

    # Orientation: Keep it level so it doesn't backflip into oblivion
    roll, pitch, _ = getEulerFromQuaternion(env.agent.state.base_orientation)
    reward_orientation = np.exp(-5.0 * (roll**2 + pitch**2))

    # 5. Effort Penalty (Your existing logic)
    efforts = np.array([j.effort for j in env.agent.motor_joints])
    max_torque = np.array([j.max_torque for j in env.agent.motor_joints])
    # Reduced weight on effort so it doesn't become "lazy"
    effort_penalty = -1 * np.mean((efforts / max_torque) ** 2)

    # === Final reward ===
    reward_dict = {
        "height_tracking": 1.0 * reward_height,
        "velocity_thrust": 1.0 * reward_velocity,
        "airtime": 1.5 * reward_airtime,
        "orientation": 0.5 * reward_orientation,
        "effort_penalty": 0.05 * effort_penalty,
    }

    total_reward = sum(reward_dict.values())
    env.log_rewards(reward_dict)
    
    return total_reward, reward_dict
