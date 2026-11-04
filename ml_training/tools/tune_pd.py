"""Tune the PD gains of one joint interactively in the MuJoCo viewer.

The robot is held in mid-air and one joint follows a sine trajectory under a PD torque
controller; adjust Kp/Kv live with the keyboard and watch the tracking.
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# ================= Configuration =================
DATA_DIR = Path(__file__).resolve().parents[1] / "robot_models"
INPUT_XML = str(DATA_DIR / "spotmicroai.mujoco.xml")
OUTPUT_XML = str(DATA_DIR / "spotmicro_motor_tuned.xml")

TARGET_JOINT_NAME = "front_left_shoulder"
TARGET_ACTUATOR_NAME = "front_left_shoulder_ctrl"

# Initial PD gains
Kp = 60.0
Kv = 3.0
stored_params = []
quit_flag = False


# ================= 1. Parser XML =================
def prepare_mujoco_xml(input_path: str, output_path: str) -> None:
    print(f"Generating XML from {input_path}...")
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Remove the free joint so the robot hangs in mid-air
    for body in root.iter("body"):
        for fj in body.findall("freejoint"):
            body.remove(fj)
            print(
                f" -> Removed the free joint of body '{body.get('name')}' (robot fixed in mid-air)."
            )

    # Replace the <position> actuators with torque <motor>s
    actuator_sec = root.find("actuator")
    if actuator_sec is not None:
        for pos in actuator_sec.findall("position"):
            motor = ET.Element("motor")
            motor.set("name", pos.get("name"))
            motor.set("joint", pos.get("joint"))

            # The position actuator's forcerange becomes the motor's ctrlrange (max torque)
            forcerange = pos.get("forcerange")
            if forcerange:
                motor.set("ctrlrange", forcerange)
                motor.set("ctrllimited", "true")

            # Swap the element in the tree
            list_of_actuators = list(actuator_sec)
            idx = list_of_actuators.index(pos)
            actuator_sec.insert(idx, motor)
            actuator_sec.remove(pos)

    tree.write(output_path)
    print(f" -> New XML saved to: {output_path}\n")


# ================= 2. PD controller =================
def pd_controller(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    # Find the joint and actuator ids
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, TARGET_JOINT_NAME)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, TARGET_ACTUATOR_NAME)

    if joint_id == -1 or actuator_id == -1:
        return  # joint not found: do nothing

    # Indices into the state arrays
    qpos_idx = model.jnt_qposadr[joint_id]
    qvel_idx = model.jnt_dofadr[joint_id]

    # Sine reference trajectory
    amplitude = 0.5  # [rad]
    freq = 1.0  # [Hz]
    omega = 2 * np.pi * freq

    q_des = amplitude * np.sin(omega * data.time)
    v_des = amplitude * omega * np.cos(omega * data.time)

    # Current state
    q = data.qpos[qpos_idx]
    v = data.qvel[qvel_idx]

    # PD control law
    tau = Kp * (q_des - q) + Kv * (v_des - v)

    # Drive only the target actuator (the others stay at 0)
    data.ctrl[actuator_id] = tau


# ================= 3. Keyboard =================
def key_callback(keycode: int) -> None:
    global Kp, Kv, quit_flag
    char = chr(keycode).lower()

    step_kp = 5.0
    step_kv = 0.5

    if char == "w":
        Kp += step_kp
    elif char == "e":
        Kp = max(0.0, Kp - step_kp)
    elif char == "r":
        Kv += step_kv
    elif char == "t":
        Kv = max(0.0, Kv - step_kv)
    elif char == "s":
        stored_params.append({"Kp": Kp, "Kv": Kv})
        print(f"[STORED] Kp={Kp:.1f}, Kv={Kv:.1f}")
    elif char == "p":
        print("\n--- Stored gains ---")
        for i, p in enumerate(stored_params):
            print(f" Set {i + 1}: Kp={p['Kp']:.1f}, Kv={p['Kv']:.1f}")
        print("--------------------\n")
    elif char == "q":
        quit_flag = True

    if char in ["w", "e", "r", "t"]:
        print(f"LIVE TUNING -> Kp: {Kp:.1f} | Kv: {Kv:.1f}")


# ================= Main loop =================
if __name__ == "__main__":
    # 1. Prepare the model file
    if not os.path.exists(INPUT_XML):
        print(f"ERROR: {INPUT_XML} not found. Check the path.")
        sys.exit()

    prepare_mujoco_xml(INPUT_XML, OUTPUT_XML)

    # 2. Load it in MuJoCo
    model = mujoco.MjModel.from_xml_path(OUTPUT_XML)
    data = mujoco.MjData(model)

    # Optional: uncomment to disable gravity and test the joint dynamics alone
    # model.opt.gravity[:] = [0, 0, 0]

    print("\n" + "=" * 40)
    print(" KEYBOARD (in the MuJoCo viewer):")
    print(" [W] Kp up       |  [E] Kp down")
    print(" [R] Kv up       |  [T] Kv down")
    print(" [S] Store gains |  [P] Print stored gains")
    print(" [Q] Quit and print the report")
    print("=" * 40 + "\n")

    # 3. Run the simulation
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        mujoco.set_mjcb_control(pd_controller)

        while viewer.is_running() and not quit_flag:
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            # Keep (roughly) real time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("\nSimulation finished. Stored gains:")
    if not stored_params:
        print("None stored during this session.")
    else:
        for i, p in enumerate(stored_params):
            print(f" - Set {i + 1}: Kp={p['Kp']:.1f}, Kv={p['Kv']:.1f}")
