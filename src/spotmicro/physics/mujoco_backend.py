import numpy as np
import mujoco
import mujoco.viewer
import time

from spotmicro.physics.backend import PhysicsBackend, JointInfo



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


class MujocoBackend(PhysicsBackend):
    """PhysicsBackend implementation backed by MuJoCo."""

    def __init__(self, use_gui: bool = False, sim_frequency: int = 240):
        self._model = None
        self._data = None
        self._viewer = None
        self._use_gui = use_gui
        self._sim_frequency = sim_frequency
        # Mapping from joint name to actuator id (populated in load_model)
        self._joint_to_actuator: dict[str, int] = {}
        self._ground_geom_id: int = -1

    @property
    def engine_name(self) -> str:
        return "mujoco"

    # ── Lifecycle ──────────────────────────────────────────────
    def load_model(self, model_path: str, position: np.ndarray, orientation: np.ndarray) -> None:
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        self._ground_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

        # Build joint_name -> actuator_id mapping
        self._joint_to_actuator = {}
        for i in range(self._model.nu):
            act_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if act_name:
                joint_name = act_name.replace("_ctrl", "")
                self._joint_to_actuator[joint_name] = i

        if self._use_gui:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ── Simulation stepping ────────────────────────────────────
    def step(self) -> None:
        mujoco.mj_step(self._model, self._data)


    def sync_viewer(self) -> None:
        if self._viewer is not None:
            self._viewer.sync()

    # ── Robot queries ──────────────────────────────────────────
    def get_base_position(self) -> np.ndarray:
        return self._data.xpos[1].copy()

    def get_base_orientation(self) -> np.ndarray:
        return self._data.xquat[1].copy()

    def get_base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        lin_vel = self._data.cvel[1, :3].copy()
        ang_vel = self._data.cvel[1, 3:].copy()
        return lin_vel, ang_vel

    def get_joint_state(self, joint_id: int) -> tuple[float, float]:
        return float(self._data.qpos[joint_id]), float(self._data.qvel[joint_id])

    def get_joint_infos(self) -> list[JointInfo]:
        infos = []
        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name is None:
                continue
            if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                category = name.split("_")[-1]
                actuator_id = self._joint_to_actuator.get(name, i)
                infos.append(JointInfo(
                    name=name,
                    joint_id=actuator_id,
                    link_id=i,
                    joint_type=category,
                    limits=(float(self._model.jnt_range[i, 0]), float(self._model.jnt_range[i, 1])),
                    qposadr=int(self._model.jnt_qposadr[i]),
                ))
        return infos

    def get_link_position(self, link_id: int) -> np.ndarray:
        body_id = self._model.jnt_bodyid[link_id]
        return self._data.xpos[body_id].copy()

    def get_rotation_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        return _quaternion_to_matrix(quaternion)

    def euler_from_quaternion(self, quaternion: np.ndarray) -> np.ndarray:
        return _quaternion_to_euler(quaternion)

    def quaternion_from_euler(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        return _quaternion_from_euler(roll, pitch, yaw)

    # ── Robot control ──────────────────────────────────────────
    def apply_joint_controls(self, joint_ids: list[int], positions: list[float], max_torques: list[float]) -> None:
        for jid, pos, _ in zip(joint_ids, positions, max_torques):
            self._data.ctrl[jid] = pos

    def reset_robot(self, position: np.ndarray, orientation: np.ndarray,
                    joint_ids: list[int], joint_positions: list[float]) -> None:
        self._data.qpos[:] = 0
        self._data.qvel[:] = 0
        self._data.ctrl[:] = 0

        # Base position (free joint: first 3 = pos, next 4 = quat)
        self._data.qpos[0:3] = position
        self._data.qpos[3:7] = orientation

        # Joint homing (resolve qposadr from model)
        for jid, jpos in zip(joint_ids, joint_positions):
            # jid here is the actuator id; we need qposadr.
            # We use the stored JointInfo which has qposadr, but reset_robot
            # is called with actuator ids. We search for the matching joint.
            for i in range(self._model.njnt):
                name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if name is None:
                    continue
                act_id = self._joint_to_actuator.get(name, i)
                if act_id == jid:
                    self._data.qpos[self._model.jnt_qposadr[i]] = jpos
                    break

    # ── Contact detection ──────────────────────────────────────
    def get_feet_contacts(self, foot_link_ids: list[int]) -> set[int]:
        feet_in_contact: set[int] = set()

        foot_body_ids = {
            self._model.jnt_bodyid[link_id]: link_id
            for link_id in foot_link_ids
        }

        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            if self._ground_geom_id not in (geom1, geom2):
                continue

            other_geom = geom2 if geom1 == self._ground_geom_id else geom1
            other_body_id = self._model.geom_bodyid[other_geom]
            foot_link_id = foot_body_ids.get(other_body_id)

            if foot_link_id is not None:
                feet_in_contact.add(foot_link_id)

        return feet_in_contact

    # ── Extra accessors ────────────────────────────────────────
    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    # ── Terrain management ──────────────────────────────────────
    def spawn_terrain(self, heightmap_data: np.ndarray, scale: list[float], origin: list[float]) -> int:
        """Spawn terrain from heightmap data.

        Modifies the hfield data directly without reloading the model.

        Returns:
            int: Hfield ID of the terrain
        """
        if self._model is None:
            raise RuntimeError("Model must be loaded before spawning terrain. Call load_model() first.")

        # Find the hfield
        hfield_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        if hfield_id < 0:
            raise RuntimeError("No hfield named 'terrain' found in model. Add <hfield name='terrain' .../> to XML.")

        # Get hfield dimensions
        nrow = self._model.hfield_nrow[hfield_id]
        ncol = self._model.hfield_ncol[hfield_id]

        # Resize heightmap to match hfield dimensions
        from scipy.ndimage import zoom
        scale_factors = (nrow / heightmap_data.shape[0], ncol / heightmap_data.shape[1])
        resized = zoom(heightmap_data, scale_factors)

        # Normalize to [0, 1] for MuJoCo
        normalized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)

        # Get data slice for this hfield
        hfield_adr = self._model.hfield_adr[hfield_id]
        hfield_size = nrow * ncol

        # Modify hfield data directly (no reload!)
        self._model.hfield_data[hfield_adr:hfield_adr + hfield_size] = normalized.flatten()

        # Update physics constants
        mujoco.mj_setConst(self._model, self._data)

        # Update rendering if viewer exists
        if self._viewer is not None:
            self._viewer.update_hfield(hfield_id)
            self._viewer.sync()

        # Update ground geom id for contact detection
        self._ground_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        return hfield_id

    def remove_terrain(self, terrain_handle: int) -> None:
        """Reset terrain to flat plane.

        Sets all hfield values to 0.5 (middle height = flat).
        """
        if self._model is None:
            return

        hfield_id = terrain_handle
        nrow = self._model.hfield_nrow[hfield_id]
        ncol = self._model.hfield_ncol[hfield_id]
        hfield_adr = self._model.hfield_adr[hfield_id]
        hfield_size = nrow * ncol

        # Reset to flat (0.5 = middle value)
        self._model.hfield_data[hfield_adr:hfield_adr + hfield_size] = 0.5

        # Update physics
        mujoco.mj_setConst(self._model, self._data)

        # Update rendering
        if self._viewer is not None:
            self._viewer.update_hfield(hfield_id)
            self._viewer.sync()
