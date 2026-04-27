import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import mujoco

from spotmicro.env.spotmicro_env import SpotmicroEnv
from spotmicro.physics.factory import create_backend
from spotmicro.devices.fixed_controller import FixedController
from spotmicro.tools.config import Config
from reward_function import reward_function, RewardState
from stable_baselines3.common.env_checker import check_env

# ========= CONFIG TEST DI CARICO ==========
BODY_TARGET = "base_link"  # Nome del corpo del baricentro
RAMP_RATE = 60.0            # Newton aggiunti al secondo (incremento graduale)
MAX_DROP_PCT = 0.075        # Smetti quando scende del 20% rispetto all'altezza iniziale
# ==========================================

# ========= ENV ==========
cfg = Config()
dev = FixedController("still")
backend = create_backend("mujoco", use_gui=True)
env = SpotmicroEnv(
    backend,
    dev,
    cfg,
    reward_function,
    RewardState(),
    use_gui=True
)

# ========= MODEL ==========
run = "stand"
model = PPO.load(f"ppo_{run}")

# ========= LOGGING STORAGE ==========
history = {
    'time': [],
    'torques': [],
    'height': [],
    'applied_force': []
}

# Accesso diretto al backend MuJoCo per limiti e dati fisici
model_mj = env._backend.model
data_mj = env._backend.data
ctrl_limits = model_mj.actuator_ctrlrange
body_id = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, BODY_TARGET)

obs, _ = env.reset()
initial_height = data_mj.xpos[body_id][2]
applied_f_z = 0.0
force_active = True

print(f"Rollout iniziato. Altezza iniziale: {initial_height:.3f}m")
print(f"Soglia di stop (20% drop): {initial_height * (1 - MAX_DROP_PCT):.3f}m")

try:
    for step in range(2000):
        # 1. Applicazione Forza Verticale Graduale
        if force_active:
            # Incremento basato sul tempo di simulazione
            applied_f_z += RAMP_RATE * model_mj.opt.timestep
            data_mj.xfrc_applied[body_id][2] = -applied_f_z
            
            # Controllo altezza attuale
            current_height = data_mj.xpos[body_id][2]
            drop_pct = (initial_height - current_height) / initial_height
            
            if drop_pct >= MAX_DROP_PCT:
                print(f"!!! CEDIMENTO RAGGIUNTO a {applied_f_z:.1f} Newton")
                force_active = False
                data_mj.xfrc_applied[body_id][2] = 0.0 # Rilascia per vedere il recupero
        
        # 2. Step della Policy
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. Logging
        history['time'].append(data_mj.time)
        history['torques'].append(np.copy(data_mj.actuator_force))
        history['height'].append(data_mj.xpos[body_id][2])
        history['applied_force'].append(applied_f_z if force_active else 0.0)

        time.sleep(1/120.) # Playback accelerato per i test
        if terminated or truncated: break

finally:
    env.close()
    
    # ========= PLOT DEI RISULTATI ==========
    torques_array = np.array(history['torques'])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot Altezza e Forza
    ax1.plot(history['time'], history['height'], color='blue', label='Altezza COM (m)')
    ax1_force = ax1.twinx()
    ax1_force.plot(history['time'], history['applied_force'], color='red', linestyle='--', label='Forza Applicata (N)')
    ax1.set_ylabel("Altezza (m)")
    ax1_force.set_ylabel("Forza (N)")
    ax1.set_title("Risposta del Robot al Carico Verticale")
    ax1.legend(loc='upper left')
    ax1_force.legend(loc='upper right')

    # Plot Coppie Motori (primi 3 per esempio)
    for i in range(3):
        ax2.plot(history['time'], torques_array[:, i], label=f'Motore {i}')
        ax2.axhline(y=ctrl_limits[i, 1], color='r', linestyle=':', alpha=0.3)
    
    ax2.set_ylabel("Torque (Nm)")
    ax2.set_xlabel("Tempo (s)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()