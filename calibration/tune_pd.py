import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# ================= Configurazione =================
# Sostituisci con il percorso reale del tuo file XML
INPUT_XML = "../src/spotmicro/data/spotmicroai.mujoco.xml" 
OUTPUT_XML = "../src/spotmicro/data/spotmicro_motor_tuned.xml"

TARGET_JOINT_NAME = "front_left_shoulder"
TARGET_ACTUATOR_NAME = "front_left_shoulder_ctrl"

# Parametri PD iniziali
Kp = 60.0
Kv = 3.0
stored_params = []
quit_flag = False

# ================= 1. Parser XML =================
def prepare_mujoco_xml(input_path, output_path):
    print(f"Generazione file XML da {input_path}...")
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Rimuovi il freejoint per bloccare il robot a mezz'aria
    for body in root.iter('body'):
        for fj in body.findall('freejoint'):
            body.remove(fj)
            print(f" -> Rimosso freejoint dal body '{body.get('name')}' (Robot bloccato a mezz'aria).")

    # Sostituisci <position> con <motor>
    actuator_sec = root.find('actuator')
    if actuator_sec is not None:
        for pos in actuator_sec.findall('position'):
            motor = ET.Element('motor')
            motor.set('name', pos.get('name'))
            motor.set('joint', pos.get('joint'))
            
            # Il forcerange del position control diventa il ctrlrange del motore (coppia max)
            forcerange = pos.get('forcerange')
            if forcerange:
                motor.set('ctrlrange', forcerange)
                motor.set('ctrllimited', 'true')
            
            # Rimpiazza l'elemento nell'albero
            list_of_actuators = list(actuator_sec)
            idx = list_of_actuators.index(pos)
            actuator_sec.insert(idx, motor)
            actuator_sec.remove(pos)
            
    tree.write(output_path)
    print(f" -> Nuovo XML salvato in: {output_path}\n")

# ================= 2. Controller PD =================
def pd_controller(model, data):
    global Kp, Kv
    
    # Trova gli ID del giunto e dell'attuatore
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, TARGET_JOINT_NAME)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, TARGET_ACTUATOR_NAME)
    
    if joint_id == -1 or actuator_id == -1:
        return # Giunto non trovato, ignora

    # Mappatura agli indirizzi degli array di stato
    qpos_idx = model.jnt_qposadr[joint_id]
    qvel_idx = model.jnt_dofadr[joint_id]
    
    # Genera la traiettoria sinusoidale
    amplitude = 0.5   # Ampiezza in radianti
    freq = 1.0        # Frequenza in Hz
    omega = 2 * np.pi * freq
    
    q_des = amplitude * np.sin(omega * data.time)
    v_des = amplitude * omega * np.cos(omega * data.time)
    
    # Leggi lo stato attuale
    q = data.qpos[qpos_idx]
    v = data.qvel[qvel_idx]
    
    # Legge di controllo PD
    tau = Kp * (q_des - q) + Kv * (v_des - v)
    
    # Applica solo all'attuatore target (gli altri a 0)
    data.ctrl[actuator_id] = tau

# ================= 3. Gestione Tastiera =================
def key_callback(keycode):
    global Kp, Kv, stored_params, quit_flag
    char = chr(keycode).lower()
    
    step_kp = 5.0
    step_kv = 0.5
    
    if char == 'w': Kp += step_kp
    elif char == 'e': Kp = max(0.0, Kp - step_kp)
    elif char == 'r': Kv += step_kv
    elif char == 't': Kv = max(0.0, Kv - step_kv)
    elif char == 's':
        stored_params.append({'Kp': Kp, 'Kv': Kv})
        print(f"[STORED] Salvati Kp={Kp:.1f}, Kv={Kv:.1f}")
    elif char == 'p':
        print("\n--- Parametri Memorizzati ---")
        for i, p in enumerate(stored_params):
            print(f" Set {i+1}: Kp={p['Kp']:.1f}, Kv={p['Kv']:.1f}")
        print("-----------------------------\n")
    elif char == 'q':
        quit_flag = True
        
    if char in ['w', 'e', 'r', 't']:
        print(f"LIVE TUNING -> Kp: {Kp:.1f} | Kv: {Kv:.1f}")

# ================= Main Loop =================
if __name__ == "__main__":
    # 1. Prepara il file
    if not os.path.exists(INPUT_XML):
        print(f"ERRORE: File {INPUT_XML} non trovato. Controlla il path.")
        exit()
        
    prepare_mujoco_xml(INPUT_XML, OUTPUT_XML)
    
    # 2. Carica MuJoCo
    model = mujoco.MjModel.from_xml_path(OUTPUT_XML)
    data = mujoco.MjData(model)
    
    # Disabilita la gravità globale se vuoi testare solo la pura dinamica del giunto 
    # (Opzionale, decommenta la riga sotto se serve)
    # model.opt.gravity[:] = [0, 0, 0]
    
    print("\n" + "="*40)
    print(" CONTROLLI TASTIERA (MuJoCo Viewer):")
    print(" [W] Aumenta Kp  |  [E] Riduci Kp")
    print(" [R] Aumenta Kv  |  [T] Riduci Kv")
    print(" [S] Salva Set   |  [P] Stampa Set salvati")
    print(" [Q] Esci e Stampa report")
    print("="*40 + "\n")

    # 3. Avvia Simulazione
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        mujoco.set_mjcb_control(pd_controller)
        
        while viewer.is_running() and not quit_flag:
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Mantieni il real-time (circa)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
    # Uscita pulita
    print("\nSimulazione terminata. Report finale dei parametri:")
    if not stored_params:
        print("Nessun parametro salvato durante la sessione.")
    else:
        for i, p in enumerate(stored_params):
            print(f" - Set {i+1}: Kp={p['Kp']:.1f}, Kv={p['Kv']:.1f}")