import pybullet
import numpy as np
from src.spotmicro.env.spotmicro_env_mujoco import SpotmicroEnv

class RewardState:
    def __init__(self):
        self.homing_positions = None
        self.a = -5.0

    def populate(self, env: SpotmicroEnv):
        self.homing_positions = np.array([
            float(j.homing_position) for j in env.agent.motor_joints
        ])
        return
    
def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:
    # Parabola centered at homing position
    diffs = env.agent.state.joint_positions - env.reward_state.homing_positions
    rewards = env.reward_state.a * np.square(diffs) + 1.0

    # Clip to desired range
    rewards = np.clip(rewards, -0.5, 1.0)

    pos_reward = np.mean(rewards)

    # === NEW: Torque penalty ===
    efforts = np.array([j.effort for j in env.agent.motor_joints])
    max_torque = np.array([j.max_torque for j in env.agent.motor_joints])
    normalized_effort = np.mean((efforts / max_torque) ** 2)  # normalized quadratic cost

    # === Final reward ===
    reward_dict = {
        "action_reward": 2.0 * pos_reward,
        "effort_penalty": -1 * normalized_effort,
    }

    total_reward = sum(reward_dict.values())

    env.log_rewards(reward_dict)
    return total_reward, reward_dict