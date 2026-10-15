
from __future__ import annotations
import threading
import numpy as np
import time
import rclpy # type: ignore
from rclpy.executors import MultiThreadedExecutor # type: ignore
from dataclasses import dataclass, field

from gz.transport13 import Node as GzNode
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.world_stats_pb2 import WorldStatistics

from backend import PhysicsBackend, JointInfo

# Importa il tuo nodo esistente
from RobotControllerNode import RobotControllerNode  # ← adatta il path

WORLD_NAME = "empty"  # ← nome del world nel tuo SDF

def _quaternion_from_euler(roll, pitch, yaw):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def _quaternion_to_euler(q):
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2, sinp) if np.abs(sinp) >= 1 else np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def _quaternion_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

def pause_simulation(gz_node: GzNode, pause : bool):
    """Mette in pausa la simulazione."""
    req = WorldControl()
    req.pause = pause
    gz_node.request(
        f"/world/{WORLD_NAME}/control",
        req, WorldControl, Boolean, 500
    )
    if pause:
        print("Simulazione messa in pausa")
    else:
        print("Simulazione avviata")

def is_paused(gz_node: GzNode) -> bool:
    """Legge lo stato della simulazione da gz-transport."""
    msg = gz_node.request(
        f"/world/{WORLD_NAME}/stats",
        WorldStatistics,
        WorldStatistics,
        500
    )
    return msg.paused if msg else None

class GazeboBackend(PhysicsBackend):
    def __init__(self):

    # ── 1. Init ROS2 ──────────────────────────────────────
        if not rclpy.ok():
            rclpy.init()

        # ── 2. Istanzia il TUO nodo ───────────────────────────
        #    Nota: il timer demo_loop non serve qui, puoi
        #    commentarlo nel costruttore del tuo nodo se vuoi.
        self._robot_node = RobotControllerNode()

        # ── 3. Executor: fa girare il tuo nodo in background ──
        #    Senza questo i callback (joint_state_callback,
        #    imu_callback) non verrebbero mai chiamati e
        #    last_known_state / last_orientation resterebbero
        #    sempre vuoti.
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._robot_node)

        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            daemon=True   # si chiude automaticamente col programma
        )
        self._spin_thread.start()

        # ── 4. Gz-transport: solo per lo step ─────────────────
        self._gz_node = GzNode()

    # ══════════════════════════════════════════════════════════
    #  engine_name
    # ══════════════════════════════════════════════════════════
    @property
    def engine_name(self) -> str:
        return "gazebo_harmonic"

    # ══════════════════════════════════════════════════════════
    #  Lifecycle
    # ══════════════════════════════════════════════════════════
    def load_model(self, model_path: str, position: np.ndarray, orientation: np.ndarray) -> None:
        pass  # il modello è già spawnato dal launch file

    def close(self) -> None:
        self._executor.shutdown()
        rclpy.shutdown()

    def step(self) -> None:
        """
        Avanza la simulazione di esattamente 1 timestep usando
        il servizio gz-transport /world/<name>/control.
        Equivalente di pybullet.stepSimulation().
        """
        req = WorldControl()
        req.multi_step = 1          # numero di step da eseguire

        success, _, = self._gz_node.request(
            f"/world/{WORLD_NAME}/control",
            req,
            WorldControl,
            Boolean,
            500,
        )
        #pause_simulation(self._gz_node, True)  # rimette in pausa dopo lo step
        if not success:
            self._ros_node.get_logger().warn("gz step: richiesta fallita o timeout")
    
    def sync_viewer(self) -> None:
        """Update the GUI viewer (no-op when headless)."""
        ...

    # ── Robot queries ──────────────────────────────────────────
    def get_base_position(self) -> np.ndarray:
        """Return (3,) world-frame position of the robot base."""
        pos = self._robot_node.get_base_position_odom()
        return np.array(pos)

    def get_base_orientation(self) -> np.ndarray:
        """Return (4,) quaternion (w,x,y,z) of the robot base."""
        #qui puoi scegliere se usare l'IMU o l'odometria
        #w, x, y, z = self._robot_node.get_base_position_IMU()
        w, x, y, z = self._robot_node.get_base_orientation_odom()
        return np.array([w, x, y, z])

    def get_base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (linear_vel (3,), angular_vel (3,)) in world frame."""
        lin, ang = self._robot_node.get_base_velocity_odom()
        return np.array(lin), np.array(ang)

    # ══════════════════════════════════════════════════════════
    #  Joint state  ← last_known_state of the robot node
    # ══════════════════════════════════════════════════════════
    def get_joint_state(self, joint_id: int) -> tuple[float, float, float]:
        """
        Return (position, velocity, effort) of a single joint.

        Il tuo nodo salva i dati in un dizionario per nome.
        Qui mappiamo joint_id → nome usando joint_names.
        """
        if joint_id >= len(self._robot_node.joint_names):
            return 0.0, 0.0, 0.0

        name = self._robot_node.joint_names[joint_id]
        data = self._robot_node.get_joint_info(name)

        if data is None:
            return 0.0, 0.0, 0.0

        return data['position'], data['velocity'], data['effort']

    def get_joint_infos(self) -> list[JointInfo]:
        """
        Costruisce la lista di JointInfo direttamente da
        joint_names del tuo nodo (senza parsare l'URDF).
        I limiti sono placeholder — aggiungili se li hai.
        """
        return [
            JointInfo(
                name       = name,
                joint_id   = i,
                link_id    = i,
                joint_type = "revolute",
                limits     = (-3.14, 3.14),  # tobe updated if you have specific limits
            )
            for i, name in enumerate(self._robot_node.joint_names)
        ]

    def get_link_position(self, link_id: int) -> np.ndarray:
        """Return (3,) world-frame position of the given link."""
        return np.zeros(3)  # placeholder, implement if needed
    
    # ─ Utility functions ───────────────────────────────────────
    def get_rotation_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        """Convert a (4,) quaternion to a (3,3) rotation matrix."""
        return _quaternion_to_matrix(quaternion)

    def euler_from_quaternion(self, quaternion: np.ndarray) -> np.ndarray:
        """Convert a (4,) quaternion to (roll, pitch, yaw)."""
        return _quaternion_to_euler(quaternion)

    def quaternion_from_euler(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert (roll, pitch, yaw) to a (4,) quaternion."""
        return _quaternion_from_euler(roll, pitch, yaw)
    
    # ══════════════════════════════════════════════════════════
    #  Joint control  ← move_joints() of the robot node
    # ══════════════════════════════════════════════════════════
    def apply_joint_controls(
        self,
        joint_ids:    list[int],
        positions:    list[float],
        max_torques:  list[float],
    ) -> None:
        """
        Delega direttamente a move_joints() del tuo nodo.
        Nota: move_joints() manda TUTTI i 12 giunti insieme,
        quindi costruiamo un array completo partendo dallo
        stato corrente e sovrascrivendo solo i joint_ids richiesti.
        """
        # Parte dallo stato attuale per i giunti non toccati
        current = [
            self._robot_node.last_known_state
                .get(name, {})
                .get('position', 0.0)
            for name in self._robot_node.joint_names
        ]

        # Sovrascrive solo i giunti richiesti
        for jid, pos in zip(joint_ids, positions):
            current[jid] = pos

        # Usa il tuo metodo esistente — duration=0 → immediato
        self._robot_node.move_joints(current, duration_sec=0.1)

    def reset_robot(
        self,
        position:        np.ndarray,
        orientation:     np.ndarray,
        joint_ids:       list[int],
        joint_positions: list[float],
    ) -> None:
        """Reset base pose, zero velocities, and set joint homing."""
        
        self.apply_joint_controls(joint_ids, joint_positions, [])

        pose_req = Pose()
        pose_req.name = "differential_drive_robot"  # ← nome modello nel SDF
        pose_req.position.x = float(position[0])
        pose_req.position.y = float(position[1])
        pose_req.position.z = float(position[2]) # height target in gazebo: 0.25506095
        w, x, y, z = orientation
        pose_req.orientation.w = float(w)
        pose_req.orientation.x = float(x)
        pose_req.orientation.y = float(y)
        pose_req.orientation.z = float(z)
        
        self._gz_node.request(
            f"/world/{WORLD_NAME}/set_pose",
            pose_req, Pose, Boolean, 500,
        )
        

    # ── Contact detection ──────────────────────────────────────
    def get_feet_contacts(self, foot_link_ids: list[int]) -> set[int]:
        """Return the set of foot link IDs currently in contact with the ground."""
        pass # placeholder, implement if needed

def test_reset_robot():
    gazebo = GazeboBackend()
    print("Backend avviato, attendo inizializzazione...")
    time.sleep(2.0)

    # ── Parametri di reset ────────────────────────────────────
    # Posizione: x=0, y=0, z=0.3 (leggermente sollevato da terra)
    position = np.array([0.0, 0.0, 0.3])

    # Orientazione: robot dritto, nessuna rotazione (quaternione identità)
    # formato (w, x, y, z)
    orientation = np.array([1.0, 0.0, 0.0, 0.0])

    # Tutti e 12 i giunti
    joint_ids = list(range(12))

    # Posizione di homing: tutti i giunti a 0.0
    joint_positions = [0.0] * 12

    # ── Snapshot PRIMA del reset ──────────────────────────────
    print("\n── Stato PRIMA del reset ───────────────────────────")
    for i in range(12):
        pos, vel, eff = gazebo.get_joint_state(i)
        name = gazebo._robot_node.joint_names[i]
        print(f"  {name:<30} pos={pos:+.4f}")

    # ── Esegui il reset ───────────────────────────────────────
    print("\nEseguo reset_robot()...")
    gazebo.reset_robot(
        position=position,
        orientation=orientation,
        joint_ids=joint_ids,
        joint_positions=joint_positions
    )

    time.sleep(1.0)  # aspetta che Gazebo applichi il reset

    # ── Snapshot DOPO il reset ────────────────────────────────
    print("\n── Stato DOPO il reset ─────────────────────────────")
    for i in range(12):
        pos, vel, eff = gazebo.get_joint_state(i)
        name = gazebo._robot_node.joint_names[i]
        print(f"  {name:<30} pos={pos:+.4f}  (atteso: 0.0000)")

    # ── Verifica posizione base ───────────────────────────────
    base_pos = gazebo.get_base_position()
    print(f"\nPosizione base: {base_pos}  (attesa: [0.0, 0.0, 0.3])")

    gazebo.close()

def test_apply_joint_controls():
    gazebo = GazeboBackend()
    print("Engine started...")

    time.sleep(1.0)
    
    for i in range(40):
        pause_simulation(gazebo._gz_node, False)
        joint_state = gazebo.get_joint_state(0)
        base_orientation = gazebo.get_base_orientation()
        base_position = gazebo.get_base_position()
        base_ang_vel, base_lin_vel = gazebo.get_base_velocity()

        #print(f"Joint 0 state: pos={joint_state[0]:.2f}, vel={joint_state[1]:.2f}, eff={joint_state[2]:.2f}")
        print(f"Base position: x={base_position[0]:.2f}, y={base_position[1]:.2f}, z={base_position[2]:.2f}")
        print(f"Base orientation: w={base_orientation[0]:.2f}, x={base_orientation[1]:.2f}, y={base_orientation[2]:.2f}, z={base_orientation[3]:.2f}")
        print(f"Base velocity: linear={base_lin_vel}, angular={base_ang_vel}")
        
        if i % 2 == 0:
            # Posizione A: tutti a 0.0
            targets = [0.0] * 12
            gazebo.apply_joint_controls(
                joint_ids=list(range(12)),
                positions=targets,
                max_torques=[1.0] * 12
            )
        else:
            # Posizione B: tutti a 0.3
            targets = [0.3] * 12
            gazebo.apply_joint_controls(
                joint_ids=list(range(12)),
                positions=targets,
                max_torques=[1.0] * 12
            )
        
        time.sleep(1.1)  # aspetta 1.1 secondi tra i comandi
        pause_simulation(gazebo._gz_node, True)
        time.sleep(1)
        
    
    gazebo.close()
    
def test_reset():##test del reset
    gazebo = GazeboBackend()
    print("Engine started...")
    
    time.sleep(1.0)

    # ── Parametri di reset ────────────────────────────────────
    # Posizione: x=0, y=0, z=0.3 (leggermente sollevato da terra)
    position = np.array([1.0, 2.0, 0.26])

    # Orientazione: robot dritto, nessuna rotazione (quaternione identità)
    # formato (w, x, y, z)
    orientation = np.array([1.0, 0.0, 0.0, 0.0])

    # Tutti e 12 i giunti
    joint_ids = list(range(12))

    # Posizione di homing: tutti i giunti a 0.0
    joint_positions = [0.0] * 12

    # ── Snapshot PRIMA del reset ──────────────────────────────
    print("\n── Stato PRIMA del reset ───────────────────────────")
    for i in range(12):
        pos, vel, eff = gazebo.get_joint_state(i)
        name = gazebo._robot_node.joint_names[i]
        print(f"  {name:<30} pos={pos:+.4f}")

    # ── Esegui il reset ───────────────────────────────────────
    print("\nEseguo reset_robot()...")
    gazebo.reset_robot(
        position=position,
        orientation=orientation,
        joint_ids=joint_ids,
        joint_positions=joint_positions
    )

    time.sleep(2.0)  # aspetta che Gazebo applichi il reset

    # ── Snapshot DOPO il reset ────────────────────────────────
    print("\n── Stato DOPO il reset ─────────────────────────────")
    for i in range(12):
        pos, vel, eff = gazebo.get_joint_state(i)
        name = gazebo._robot_node.joint_names[i]
        print(f"  {name:<30} pos={pos:+.4f}  (atteso: 0.0000)")

    # ── Verifica posizione base ───────────────────────────────
    base_pos = gazebo.get_base_position()
    print(f"\nPosizione base: {base_pos}  (attesa: [0.0, 0.0, 0.3])")

    gazebo.close()

def test_step2():
    gazebo = GazeboBackend()
    print("Backend avviato, attendo inizializzazione...")
    time.sleep(2.0)
    print("\nInvio comando target (0.5 rad su tutti i giunti)...")
    pause_simulation(gazebo._gz_node, True)
    time.sleep(0.5)
    """gazebo.apply_joint_controls(
        joint_ids=list(range(12)),
        positions=[0.5] * 12,
        max_torques=[1.0] * 12
    )"""
    time.sleep(5.0)

    # ── Esegui N step e osserva il movimento ──────────────────
    print("\n── Step individuali ────────────────────────────────")
    N_STEPS = 400
    positions = []

    for i in range(N_STEPS):
        gazebo.step()
        print("Palle")
        #pos, vel, eff = gazebo.get_joint_state(0)
        #positions.append(pos)
        #print(f"  step {i+1:02d} → pos={pos:+.6f}  vel={vel:+.6f}")
    #is_paused = is_paused(gazebo._gz_node)
    #print(f"Simulazione in pausa: {is_paused}")

def test_step():
    gazebo = GazeboBackend()
    print("Backend avviato, attendo inizializzazione...")
    time.sleep(2.0)

    # ── Verifica che i dati arrivino ──────────────────────────
    if not gazebo._robot_node.last_known_state:
        print("⚠️  Nessun dato da /joint_states — controlla che Gazebo sia attivo")
        gazebo.close()
        return

    print("✅ Dati ricevuti correttamente")

    # ── Metti in pausa la simulazione ─────────────────────────
    # Lo facciamo sempre, sia che sia già in pausa sia che non lo sia.
    # Se era già in pausa non cambia nulla.

    # resetto la posizione del robot
    print("\nEseguo reset_robot()...")
    position = np.array([1.0, 2.0, 0.26])
    orientation = np.array([1.0, 0.0, 0.0, 0.0])
    joint_ids = list(range(12))
    joint_positions = [0.0] * 12
    gazebo.reset_robot(
        position=position,
        orientation=orientation,
        joint_ids=joint_ids,
        joint_positions=joint_positions
    )
    time.sleep(3.0)  # aspetta che Gazebo applichi il reset

    print("\nMetto in pausa la simulazione...")
    pause_simulation(gazebo._gz_node, True)
    time.sleep(0.5)

    # ── Snapshot PRIMA degli step ─────────────────────────────
    joint_name = gazebo._robot_node.joint_names[0]
    pos_before = gazebo._robot_node.last_known_state.get(joint_name, {}).get('position', None)
    print(f"\nGiunto monitorato: '{joint_name}'")
    print(f"Posizione PRIMA degli step: {pos_before}")

    # ── Invia un comando target ───────────────────────────────
    # Comando a 0.5 rad su tutti i giunti
    print("\nInvio comando target (0.5 rad su tutti i giunti)...")
    gazebo.apply_joint_controls(
        joint_ids=list(range(12)),
        positions=[0.5] * 12,
        max_torques=[1.0] * 12
    )

    # ── Esegui N step e osserva il movimento ──────────────────
    print("\n── Step individuali ────────────────────────────────")
    N_STEPS = 40
    positions = []

    for i in range(N_STEPS):
        gazebo.step()
        pos, vel, eff = gazebo.get_joint_state(0)
        positions.append(pos)
        print(f"  step {i+1:02d} → pos={pos:+.6f}  vel={vel:+.6f}")
        time.sleep(0.1)

    # ── Analisi risultati ─────────────────────────────────────
    print("\n── Analisi ─────────────────────────────────────────")

    if pos_before is None:
        print("⚠️  Nessun dato iniziale disponibile")
    elif all(p == positions[0] for p in positions):
        print("⚠️  La posizione NON cambia tra gli step")
        print("   Possibili cause:")
        print("   1. Il servizio /world/control non funziona — verifica con:")
        print(f"      gz service -l | grep {WORLD_NAME}")
        print("   2. Il controller non sta applicando i comandi")
    else:
        delta = abs(positions[-1] - positions[0])
        print(f"✅ La posizione cambia tra gli step (delta totale: {delta:.6f} rad)")

        # Verifica che il movimento sia nella direzione giusta
        if positions[-1] > positions[0]:
            print(f"✅ Il giunto si sta muovendo verso il target (0.5 rad)")
        else:
            print(f"ℹ️  Il giunto si sta muovendo in direzione opposta al target")

    gazebo.close()

def main(args=None):
    test_apply_joint_controls()
    

    
if __name__ == '__main__':
    main()