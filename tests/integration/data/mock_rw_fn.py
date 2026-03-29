"""
This function here is more complex than it should for the purpose it serves here (just basic testng).
This is because it was copy-pasted from an already developed reward function.
In any case, this way we test a wide array of attributes (thiugh there's no guarantee we are testing every single one)
"""


import pybullet
import numpy as np
# Importa la classe SpotmicroEnv, che definisce l'ambiente di simulazione del robot.
from spotmicro.env.spotmicro_env_mujoco import SpotmicroEnv

# classe per memorizzare lo stato tra un timestep e l'altro
class RewardState:
    def __init__(self):
        # Insieme per memorizzare i contatti precedenti dei piedi del robot. Attualmente non utilizzato.
        self.prev_contacts = set() # INUTILIZZATO
        # Posizione precedente della base del robot, inizializzata a zero.
        self.prev_base_position = np.array([0.0, 0.0, 0.0]) # placeholder che viene aggiornato al primo step

    def populate(self, env: SpotmicroEnv): # INUTILIZZATO
        # Questo metodo dovrebbe aggiornare lo stato della ricompensa con le informazioni dell'ambiente, ma attualmente è vuoto.
        return

# Funzione per aumentare gradualmente un valore da 0 a 1, utile per introdurre gradualmente componenti della reward.
def fade_in(current_step, start, scale=2.0): # INUTILIZZATO
    # Se il passo corrente è inferiore a quello di inizio, ritorna 0.
    if current_step < start:
        return 0.0
    # Calcola un valore che cresce esponenzialmente verso 1.
    return 1.0 - np.exp(-scale * (current_step - start) / 1_000_000)

# Funzione principale che calcola la ricompensa totale per un dato stato e azione.
def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:
    # Extract roll, pitch, yaw
    roll, pitch, _ = env.agent.state.roll_pitch_yaw

    # Input velocities in robot frame
    vx_i, vy_i, w_i = tuple(env.agent.controller.input.as_array)
    # Target linear velocity in world frame
    target_linear_velocity = np.array(
        pybullet.getMatrixFromQuaternion(env.agent.state.base_orientation)
    ).reshape(3,3) @ np.array([vx_i, vy_i, 0.0])
    target_angular_velocity = np.array([w_i, 0.0, 0.0])

    # --- Linear velocity error ---
    target_norm = np.linalg.norm(target_linear_velocity)
    if target_norm < 1e-8:
        lin_vel_sq_perc_error = 0.0
    else:
        lin_vel_sq_perc_error = (
            np.linalg.norm(target_linear_velocity - env.agent.state.linear_velocity) / target_norm
        ) ** 2

    lin_vel_reward = max(1.0 - (lin_vel_sq_perc_error / 0.3) ** 2, -0.5)

    # --- Angular velocity error ---
    max_ang_vel = env.agent.max_angular_velocity
    if max_ang_vel < 1e-8:
        ang_vel_error = 0.0
    else:
        ang_vel_error = np.linalg.norm(
            (target_angular_velocity - env.agent.state.angular_velocity) / max_ang_vel
        ) ** 2

    # --- Perpendicular drift penalty ---
    if target_norm < 1e-8:
        perp_velocity = env.agent.state.linear_velocity
    else:
        proj = (np.dot(env.agent.state.linear_velocity, target_linear_velocity) /
                (target_norm ** 2)) * target_linear_velocity
        perp_velocity = env.agent.state.linear_velocity - proj
    drift_penalty = np.linalg.norm(perp_velocity) ** 2

    # --- Progress along target ---
    delta_pos = env.agent.state.base_position - env.reward_state.prev_base_position
    if target_norm < 1e-8:
        progress = 0.0
    else:
        progress = np.dot(delta_pos, target_linear_velocity) / target_norm
        progress = np.clip(progress / env.sim_frequency, -0.5, 0.5)

    # --- Other components ---
    deviation_penalty = np.linalg.norm(env.agent.state.joint_positions - env.agent.homing_positions) ** 2
    body_to_feet_height = env.agent.get_body_to_feet_height_projected()
    height_penalty = (body_to_feet_height - env.target_body_to_feet_height) ** 2
    action_rate = np.mean(action - env.agent.previous_action) ** 2
    vertical_velocity_sq = env.agent.state.linear_velocity[2] ** 2
    stabilization_penalty = roll ** 2 + pitch ** 2
    total_normalized_effort = np.sum([(j.effort / j.max_torque) ** 2 for j in env.agent.motor_joints]) / len(env.agent.motor_joints)

    # --- Compose reward dict ---
    reward_dict = {
        "linear_vel_reward": 12 * lin_vel_reward,
        "progress_reward": 1 * progress,
        "angular_vel_penalty": -5 * ang_vel_error,
        "drift_penalty": -6 * drift_penalty,
        "action_rate_penalty": -2 * action_rate,
        "height_penalty": -3 * min(height_penalty, 1.0),
        "stabilization_penalty": -3 * min(stabilization_penalty, 1.0),
        "effort_penalty": -1.5 * total_normalized_effort,
        "deviation_penalty": -0.0 * deviation_penalty,
        "vertical_motion_penalty": -0.5 * vertical_velocity_sq,
    }

    total_reward = sum(reward_dict.values())

    # Log reward components for monitoring
    env.log_rewards(reward_dict)

    return total_reward, reward_dict
