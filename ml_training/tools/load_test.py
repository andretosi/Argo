"""Load test for a standing policy.

Pushes the base down with a force that grows over time until the robot sags by
MAX_DROP_PCT of its initial height, then releases it to watch the recovery, and plots the
height, the applied force and the motor torques.

From ml_training/::

    python tools/load_test.py                                   # training/data/standing/final.zip
    python tools/load_test.py training/data/<run>/final.zip
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from stable_baselines3 import PPO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from commands.fixed_controller import FixedController
from training.builders import load_run, make_env

# ========= LOAD TEST SETTINGS ==========
BODY_TARGET = "base_link"  # body the force is applied to
RAMP_RATE = 60.0  # force increase [N per simulated second]
MAX_DROP_PCT = 0.075  # release when the base has dropped this fraction of its initial height
# ==========================================

# ========= MODEL AND ENV (the model's run config, commanded to stand still) ==========
model_path = sys.argv[1] if len(sys.argv) > 1 else ROOT_DIR / "training/data/standing/final.zip"
model = PPO.load(str(model_path), device="cpu")
env = make_env(load_run(model_path), gui=True, command=FixedController("still"))

# ========= LOGGING STORAGE ==========
history = {"time": [], "torques": [], "height": [], "applied_force": []}

# Direct access to the MuJoCo model and data, for limits and physical quantities
model_mj = env.sim.model
data_mj = env.sim.data
ctrl_limits = model_mj.actuator_ctrlrange
body_id = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, BODY_TARGET)

obs, _ = env.reset()
initial_height = data_mj.xpos[body_id][2]
applied_f_z = 0.0
force_active = True

print(f"Rollout started. Initial height: {initial_height:.3f} m")
print(f"Release threshold ({MAX_DROP_PCT:.1%} drop): {initial_height * (1 - MAX_DROP_PCT):.3f} m")

try:
    for step in range(2000):
        # 1. Gradually increasing downward force (the env clears it after every step)
        if force_active:
            # Grows with simulated time: one control step per loop iteration
            applied_f_z += RAMP_RATE / env.control_frequency
            data_mj.xfrc_applied[body_id][2] = -applied_f_z

            # Current height
            current_height = data_mj.xpos[body_id][2]
            drop_pct = (initial_height - current_height) / initial_height

            if drop_pct >= MAX_DROP_PCT:
                print(f"!!! Collapse threshold reached at {applied_f_z:.1f} N")
                force_active = False
                data_mj.xfrc_applied[body_id][2] = 0.0  # release to watch the recovery

        # 2. Policy step
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # 3. Logging
        history["time"].append(data_mj.time)
        history["torques"].append(np.copy(data_mj.actuator_force))
        history["height"].append(data_mj.xpos[body_id][2])
        history["applied_force"].append(applied_f_z if force_active else 0.0)

        time.sleep(1 / 120.0)  # playback at twice real time
        if terminated or truncated:
            break

finally:
    env.close()

    # ========= PLOTS ==========
    torques_array = np.array(history["torques"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Height and force
    ax1.plot(history["time"], history["height"], color="blue", label="Base height (m)")
    ax1_force = ax1.twinx()
    ax1_force.plot(
        history["time"],
        history["applied_force"],
        color="red",
        linestyle="--",
        label="Applied force (N)",
    )
    ax1.set_ylabel("Height (m)")
    ax1_force.set_ylabel("Force (N)")
    ax1.set_title("Robot response to a vertical load")
    ax1.legend(loc="upper left")
    ax1_force.legend(loc="upper right")

    # Motor torques (the first three, as an example)
    for i in range(3):
        ax2.plot(history["time"], torques_array[:, i], label=f"Motor {i}")
        ax2.axhline(y=ctrl_limits[i, 1], color="r", linestyle=":", alpha=0.3)

    ax2.set_ylabel("Torque (Nm)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
