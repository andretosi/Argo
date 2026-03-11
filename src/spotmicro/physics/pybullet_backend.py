import numpy as np
import pybullet
import pybullet_data
from importlib.resources import files

from src.spotmicro.physics.backend import PhysicsBackend, JointInfo


class PybulletBackend(PhysicsBackend):
    """PhysicsBackend implementation backed by PyBullet."""

    def __init__(self, use_gui: bool = False, sim_frequency: int = 240):
        self._client = None
        self._robot_id = None
        self._terrain_id = None
        self._use_gui = use_gui
        self._sim_frequency = sim_frequency

    @property
    def engine_name(self) -> str:
        return "pybullet"

    # ── Lifecycle ──────────────────────────────────────────────
    def load_model(self, model_path: str, position: np.ndarray, orientation: np.ndarray) -> None:
        if self._client is None:
            self._client = pybullet.connect(pybullet.GUI if self._use_gui else pybullet.DIRECT)
            pybullet.resetDebugVisualizerCamera(
                cameraDistance=1.2, cameraYaw=45, cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.2],
            )

        pybullet.resetSimulation(physicsClientId=self._client)
        pybullet.setGravity(0, 0, -9.81, physicsClientId=self._client)
        pybullet.setTimeStep(1 / self._sim_frequency, physicsClientId=self._client)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load ground plane
        self._terrain_id = pybullet.loadURDF("plane.urdf", physicsClientId=self._client)

        pybullet.changeDynamics(
            bodyUniqueId=self._terrain_id, linkIndex=-1,
            lateralFriction=1.0, spinningFriction=0.0,
            rollingFriction=0.0, restitution=0.0,
            physicsClientId=self._client,
        )

        self._robot_id = pybullet.loadURDF(
            model_path,
            basePosition=position.tolist(),
            baseOrientation=orientation.tolist(),
            physicsClientId=self._client,
        )

    def close(self) -> None:
        if self._client is not None:
            pybullet.disconnect(self._client)
            self._client = None

    # ── Simulation stepping ────────────────────────────────────
    def step(self) -> None:
        pybullet.stepSimulation(physicsClientId=self._client)

    def sync_viewer(self) -> None:
        pass  # PyBullet GUI updates automatically

    # ── Robot queries ──────────────────────────────────────────
    def get_base_position(self) -> np.ndarray:
        pos, _ = pybullet.getBasePositionAndOrientation(self._robot_id, physicsClientId=self._client)
        return np.array(pos)

    def get_base_orientation(self) -> np.ndarray:
        _, ori = pybullet.getBasePositionAndOrientation(self._robot_id, physicsClientId=self._client)
        return np.array(ori)

    def get_base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        lin, ang = pybullet.getBaseVelocity(self._robot_id, physicsClientId=self._client)
        return np.array(lin), np.array(ang)

    def get_joint_state(self, joint_id: int) -> tuple[float, float]:
        state = pybullet.getJointState(self._robot_id, joint_id, physicsClientId=self._client)
        return state[0], state[1]  # position, velocity

    def get_joint_infos(self) -> list[JointInfo]:
        infos = []
        for i in range(pybullet.getNumJoints(self._robot_id, physicsClientId=self._client)):
            ji = pybullet.getJointInfo(self._robot_id, i, physicsClientId=self._client)
            if ji[2] == pybullet.JOINT_REVOLUTE:
                name = ji[1].decode("utf-8")
                category = name.split("_")[-1]
                infos.append(JointInfo(
                    name=name,
                    joint_id=i,
                    link_id=ji[0],
                    joint_type=category,
                    limits=(ji[8], ji[9]),
                ))
        return infos

    def get_link_position(self, link_id: int) -> np.ndarray:
        state = pybullet.getLinkState(self._robot_id, link_id, physicsClientId=self._client)
        return np.array(state[0])

    def get_rotation_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        return np.array(pybullet.getMatrixFromQuaternion(quaternion)).reshape(3, 3)

    def euler_from_quaternion(self, quaternion: np.ndarray) -> np.ndarray:
        return np.array(pybullet.getEulerFromQuaternion(quaternion))

    def quaternion_from_euler(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        return np.array(pybullet.getQuaternionFromEuler([roll, pitch, yaw]))

    # ── Robot control ──────────────────────────────────────────
    def apply_joint_controls(self, joint_ids: list[int], positions: list[float], max_torques: list[float]) -> None:
        for jid, pos, torque in zip(joint_ids, positions, max_torques):
            pybullet.setJointMotorControl2(
                bodyUniqueId=self._robot_id,
                jointIndex=jid,
                controlMode=pybullet.POSITION_CONTROL,
                targetPosition=pos,
                force=torque,
                physicsClientId=self._client,
            )

    def reset_robot(self, position: np.ndarray, orientation: np.ndarray,
                    joint_ids: list[int], joint_positions: list[float]) -> None:
        pybullet.resetBasePositionAndOrientation(
            self._robot_id, position.tolist(), orientation.tolist(),
            physicsClientId=self._client,
        )
        pybullet.resetBaseVelocity(
            self._robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0],
            physicsClientId=self._client,
        )
        for jid, jpos in zip(joint_ids, joint_positions):
            pybullet.resetJointState(
                self._robot_id, jid,
                targetValue=jpos, targetVelocity=0.0,
                physicsClientId=self._client,
            )

    # ── Contact detection ──────────────────────────────────────
    def get_feet_contacts(self, foot_link_ids: list[int]) -> set[int]:
        contact_points = pybullet.getContactPoints(
            bodyA=self._robot_id, bodyB=self._terrain_id,
            physicsClientId=self._client,
        )
        feet_in_contact: set[int] = set()
        foot_set = set(foot_link_ids)
        for contact in contact_points:
            link_idx = contact[3]  # linkIndexA
            # PyBullet contact link indices are shifted by 1 vs stored link_id
            if (link_idx - 1) in foot_set:
                feet_in_contact.add(link_idx - 1)
        return feet_in_contact

    # ── Extra accessors used by legacy Terrain / Env code ──────
    @property
    def client(self):
        return self._client

    @property
    def terrain_id(self):
        return self._terrain_id
